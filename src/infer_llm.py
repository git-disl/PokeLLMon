import asyncio
from tqdm import tqdm
import torch
import numpy as np
import os
import hydra
import logging
import random
import pickle as pkl
import json
import transformers
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from src.utils.llm_utils import get_local_dir, get_local_run_dir, disable_dropout
from src.client.account_configuration import AccountConfiguration

from src.player import LLMPlayer, HeuristicsPlayer, MaxBasePowerPlayer

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

@hydra.main(version_base=None, config_path="config", config_name="config")
def entry_point(config: DictConfig):
    logging.getLogger().setLevel(logging.WARNING)
    # Running the async main function
    asyncio.get_event_loop().run_until_complete(main(config))
#
async def main(config: DictConfig):
    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    print(OmegaConf.to_yaml(config))

    save_replay_dir = os.path.join("./battle_log/", config.exp_name)
    os.makedirs(save_replay_dir, exist_ok=True)

    # load llm as policy model
    model_kwargs = {'device_map': 'balanced'}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, output_hidden_states=True, cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(
            f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])

    # Initialize Players
    index = str(random.randint(0, 10000))
    llm_player = LLMPlayer(model=policy,
                           config=config,
                           # save_replay_dir=save_replay_dir,
                           save_replay_dir="",
                           # account_configuration=AccountConfiguration("test_player926", "123456"),
                           account_configuration=AccountConfiguration("LLM" + index, "XXX"),
                           )

    opponent_player = HeuristicsPlayer(battle_format=config.battle_format, account_configuration=AccountConfiguration("Player" + index, "XXX"),)
    # dynamax is disabled for local battles.
    opponent_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    # Now, let's evaluate our player
    for i in tqdm(range(5000)):
        # print(f'I am here~')
        await llm_player.battle_against(opponent_player, n_battles=1)
        # with open(f"{save_replay_dir}/all_battles.pkl", "wb") as f:
        #     pkl.dump(llm_player.battles, f)

        print("index=", i)
        print("except turns:", llm_player.except_cnt / llm_player.total_cnt)

        # summarize battles:
        win_cnt = 0
        total_cnt = 0
        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        turn_list = []
        for tag, battle in llm_player.battles.items():
            if battle.finished:
                beat_score = 0
                for mon in battle.opponent_team.values():
                    beat_score += (1 - mon.current_hp_fraction)

                beat_list.append(beat_score)
                remain_score = 0

                for mon in battle.team.values():
                    remain_score += mon.current_hp_fraction
                remain_list.append(remain_score)
                if battle.won:
                    win_list.append(1)
                tag_list.append(tag)
                turn_list.append(battle.turn)

        print("battle score:", np.mean(beat_list) + np.mean(remain_list))
        print("win rate:", llm_player.win_rate)
        print("turn:", np.mean(turn_list))

    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    entry_point()
    # asyncio.get_event_loop().run_until_complete(main())
