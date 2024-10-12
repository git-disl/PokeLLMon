import json
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import transformers
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import re

SYSTEM_PROMPT = """System: You are an assistant. Given below dialogue history with human, output one response following the user's instructions accurately."""

class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)


def get_point_data(split: str ,silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    # with open("battle_data/self_play_heuristic/heuristic_player3.jsonl", "r") as f:
    #     dataset1 = f.readlines()
    # with open("battle_data/self_play_heuristic/heuristic_player4.jsonl", "r") as f:
    #     dataset2 = f.readlines()
    # dataset = dataset1 + dataset2
    with open("battle_data/self_play_llm_noBC/self_play_rft_iter1.json", "r") as f:
        dataset = f.readlines()

    with TemporarilySeededRandom(42):
        random.shuffle(dataset)
        if split == "train":
            dataset = dataset[:int(len(dataset)*0.95)]
        else:
            dataset = dataset[int(len(dataset)*0.95):]

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing self-play data', disable=silent):
        row = json.loads(row)
        prompt = row["prompt"]
        output = row["output"]
        data[prompt]['sft_target'].append(output)

    return data

def get_pair_data(split: str ,silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    with open("battle_data/self_play_llm_noBC/self_play_dpo_iter1.json", "r") as f:
        dataset = json.load(f)

    with TemporarilySeededRandom(42):
        random.shuffle(dataset)
        if split == "train":
            dataset = dataset[:int(len(dataset)*0.95)]
        else:
            dataset = dataset[int(len(dataset)*0.95):]

    data = []
    for row in tqdm.tqdm(dataset, desc='Processing self-play data', disable=silent):
        data.append([row["winner_prompt"], row["winner_output"], row["loser_prompt"], row["loser_output"]])

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int,
                           max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.

       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    # check if there is "<>" for reward
    batch = {}
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens[
        'input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens,
                    'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

def new_tokenize_batch_element(chosen_prompt: str, chosen_output: str, rejected_prompt: str, rejected_output: str, truncation_mode: str, tokenizer, max_length: int,
                           max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.

       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    # check if there is "<>" for reward
    batch = {}
    chosen_prompt_tokens = tokenizer(chosen_prompt, add_special_tokens=False)
    rejected_prompt_tokens = tokenizer(rejected_prompt, add_special_tokens=False)

    chosen_tokens = tokenizer(chosen_output, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected_output, add_special_tokens=False)

    assert tokenizer.eos_token_id not in chosen_prompt_tokens['input_ids'], f"Prompt contains EOS token: {chosen_prompt_tokens}"
    assert tokenizer.eos_token_id not in rejected_prompt_tokens['input_ids'], f"Prompt contains EOS token: {rejected_prompt_tokens}"

    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Prompt contains EOS token: {chosen_tokens}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Prompt contains EOS token: {rejected_tokens}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    # Create labels
    chosen_sequence_tokens = {k: chosen_prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: rejected_prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(chosen_prompt_tokens['input_ids'])] = [-100] * len(chosen_prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(rejected_prompt_tokens['input_ids'])] = [-100] * len(rejected_prompt_tokens['input_ids'])

    batch['chosen'] = chosen_prompt + chosen_output
    batch['rejected'] = rejected_prompt + rejected_output
    batch['chosen_response_only'] = chosen_output
    batch['rejected_response_only'] = rejected_output

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_pair_iterator(tokenizer,
                      split: str = 'train',
                      batch_size: int = 1,
                      shuffle: bool = True,
                      max_length: int = 512,
                      max_prompt_length: int = 128,
                      n_epochs: Optional[int] = None,
                      n_examples: Optional[int] = None,
                      seed: int = 0,
                      silent: bool = False,
                      cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        type: for point or pair data
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2 ** 32, size=1000000))
        truncation_mode = 'keep_end'
        flat_data = []
        dataset = get_pair_data(split, silent=silent, cache_dir=cache_dir)
        for data in dataset:
            data.append(truncation_mode)
            flat_data.append(data)

    print(f"flat_data #:{len(flat_data)}")
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
                print(f"current epoch_idx: {epoch_idx}")
                print(f"current example_idx: {example_idx}")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for chosen_prompt, chosen_output, rejected_prompt, rejected_output, truncation_mode in flat_data:
            if done:
                break
            batch_element = new_tokenize_batch_element(chosen_prompt, chosen_output, rejected_prompt, rejected_output, truncation_mode,
                                                       tokenizer, max_length, max_prompt_length)
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:  # this must be put right after batch.append()
                yield collate_fn(batch)
                if n_examples is not None and example_idx >= n_examples:
                    if not silent:
                        print(f'Finished generating {n_examples} examples on {split} split')
                    done = True
                batch = []

        if done:
            break
        epoch_idx += 1
        print("Epoch +=1 !!!!!")


def get_data_iterator(mode,
                      tokenizer,
                      split: str = 'train',
                      batch_size: int = 1,
                      shuffle: bool = True,
                      max_length: int = 512,
                      max_prompt_length: int = 128,
                      n_epochs: Optional[int] = None,
                      n_examples: Optional[int] = None,
                      seed: int = 0,
                      silent: bool = False,
                      cache_dir: Optional[str] = None
                      ):

    if mode == "point":
        return get_point_iterator(tokenizer, split, batch_size, shuffle,
                                  max_length, max_prompt_length, n_epochs,
                                  n_examples, seed, silent, cache_dir)
    elif mode == "pair":
        return get_pair_iterator(tokenizer, split, batch_size, shuffle,
                                  max_length, max_prompt_length, n_epochs,
                                  n_examples, seed, silent, cache_dir)

def get_point_iterator(tokenizer,
                      split: str = 'train',
                      batch_size: int = 1,
                      shuffle: bool = True,
                      max_length: int = 512,
                      max_prompt_length: int = 128,
                      n_epochs: Optional[int] = None,
                      n_examples: Optional[int] = None,
                      seed: int = 0,
                      silent: bool = False,
                      cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        type: for point or pair data
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2 ** 32, size=1000000))
        flat_data = []
        truncation_mode = 'keep_end'
        for prompt, data in get_point_data(split=split, silent=silent, cache_dir=cache_dir).items():
            flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'][0], truncation_mode))

    print(f"flat_data #:{len(flat_data)}")
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
                print(f"current epoch_idx: {epoch_idx}")
                print(f"current example_idx: {example_idx}")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break

            batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer,
                                                   max_length, max_prompt_length)
            batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:
                yield collate_fn(batch)
                if n_examples is not None and example_idx >= n_examples:
                    if not silent:
                        print(f'Finished generating {n_examples} examples on {split} split')
                    done = True
                batch = []

        if done:
            break
        epoch_idx += 1
        print("Epoch +=1 !!!!!")


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True

if __name__ == '__main__':

    tokenizer_name_or_path = "meta-llama/Llama-2-7b-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir="ckpt_dir")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_iterator = get_data_iterator("pair", tokenizer=tokenizer, split="train", n_epochs=1, batch_size=8)
    for batch in train_iterator:
        print("pause")






