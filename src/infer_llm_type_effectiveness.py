"""
Generate pairs of types to ask the LLM to predict whether it has advantage or disadvantges
"""
import argparse
from tqdm import tqdm
from src.data.gen_data import GenData
import json
from peft import PeftModel
import transformers
import torch
from sklearn.metrics import classification_report


###############
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="battle_log/type_chart_predict_task1")
args = parser.parse_args()

def generate_task1_pairs():

    gen = GenData.from_format("gen8randombattle")
    type_chat = gen.type_chart
    pair_list = []
    label_list = []

    for pokemon_type, chart_dict in type_chat.items():
        for move_type, damage_multiplier in chart_dict.items():
            pair_list.append([move_type, pokemon_type])
            label_list.append(damage_multiplier)

    return pair_list, label_list

def main():

    model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir="./ckpt_dir",
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir="./ckpt_dir",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="balanced",
    )

    print("Loading finished...")
    model.eval()
    pair_list, label_list = generate_task1_pairs()
    llm_output_list = []
    answer_list = []
    for idx in tqdm(range(len(pair_list))):
        pair = pair_list[idx]
        label = label_list[idx]
        if label == 1:
            answer = "B"
        elif label == 2:
            answer = "A"
        elif label == 0.5:
            answer = "C"
        elif label == 0:
            answer = "D"
        else:
            raise Exception("Unknown label")

        answer_list.append(answer)

        prompt = '[INST] <<SYS>> You are an assistant to answer the following multi-choice question. The output should be a JSON in the format {"answer":"<your answer>"}. No additional text is allowed. <</SYS>>\n' \
                 f'In Pokémon battles, a {pair[0].lower()} attack is ____ against a {pair[1].lower()} Pokémon?\nA. Super Effective (2x Damage)\nB. Standard (1x Damage)\nC. Not Effective (0.5x Damage)\nD. Zero Effect (0x Damage)\nYour output: [/INST]'

        # prompt = '<start_of_turn>You are an assistant to answer the following multi-choice question. The output should be a JSON in the format {"answer":"<your answer>"}. No additional text is allowed.\n' \
        #          f'In Pokémon battles, a {pair[0].lower()} attack is ____ against a {pair[1].lower()} Pokémon?\nA. Super Effective (2x Damage)\nB. Standard (1x Damage)\nC. Not Effective (0.5x Damage)\nD. Zero Effect (0x Damage)\nYour output:<end_of_turn>'

        try:
            input_dict = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_ids = input_dict["input_ids"]

            with torch.no_grad():
                generation_output = model.generate(
                                    inputs=input_ids,
                                    # top_p=1,
                                    temperature=0,
                                    do_sample=False,
                                    # do_sample=True,
                                    max_new_tokens=50,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id # self.tokenizer.pad_token_id
                                )

            s = generation_output[0]
            llm_output = tokenizer.decode(s, skip_special_tokens=True)

            llm_output = llm_output.split("Your output:")[-1]
            print(f"Move: {pair[0]}; Pokemon: {pair[1]}; Label: {answer}")
            print("LLM output:", llm_output)

            json_start = llm_output.find('{')
            json_end = llm_output.find('}') + 1  # find the first }
            json_content = llm_output[json_start:json_end]
            llm_answer = json.loads(json_content)["answer"]
            if llm_answer.startswith("A"):
                llm_output_list.append("A")
            elif llm_answer.startswith("B"):
                llm_output_list.append("B")
            elif llm_answer.startswith("C"):
                llm_output_list.append("C")
            elif llm_answer.startswith("D"):
                llm_output_list.append("D")
            else:
                llm_output_list.append("E")
        except Exception as e:
            print(e)
            llm_output_list.append("B")

    report = classification_report(answer_list, llm_output_list, digits=4)
    print(report)


if __name__ == '__main__':
    main()