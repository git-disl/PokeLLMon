import asyncio
from tqdm import tqdm
import numpy as np
import os
import hydra
import logging
import random
import pickle as pkl
from omegaconf import OmegaConf, DictConfig
import argparse
from typing import Optional, Set
from client.account_configuration import AccountConfiguration
from src.utils.llm_utils import get_local_dir, get_local_run_dir
from src.player import GPTPlayer, HeuristicsPlayer

# parser = argparse.ArgumentParser()
# parser.add_argument("--backend", type=str, default="gpt-4o-mini", choices=["gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4-1106", "gpt-4-0125"])
# parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
# args = parser.parse_args()

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

@hydra.main(version_base=None, config_path="config", config_name="config")
def entry_point(config: DictConfig):
    logging.getLogger().setLevel(logging.WARNING)
    # Running the async main function
    asyncio.get_event_loop().run_until_complete(main(config))

async def main(config: DictConfig):

    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    print(OmegaConf.to_yaml(config))

    save_replay_dir = os.path.join("./battle_log/", config.exp_name)
    os.makedirs(save_replay_dir, exist_ok=True)

    # Initialize Players
    index = str(random.randint(0, 10000))
    heuristic_player = HeuristicsPlayer(battle_format=config.battle_format, account_configuration=AccountConfiguration("Heuristic" + index, "XXX"),)
    os.makedirs(save_replay_dir, exist_ok=True)
    gpt_player = GPTPlayer(config=config,
                           api_key=os.getenv("OPENAI_API_KEY"),
                           backend="gpt-4o-2024-08-06",
                           # backend="gpt-4-1106-preview",
                           # backend="gpt-3.5-turbo",
                           # backend="gpt-4o-mini-2024-07-18",
                           reason_algo=config.reason_algo,
                           save_replay_dir=save_replay_dir,
                           account_configuration=AccountConfiguration("test_player926", "123456"),
                           )

    # dynamax is disabled for local battles.
    heuristic_player._dynamax_disable = True
    gpt_player._dynamax_disable = True

    # play against bot for five battles

    for i in range(101):
        await gpt_player.battle_against(heuristic_player, n_battles=1)
        with open(f"{save_replay_dir}/all_battles_{index}.pkl", "wb") as f:
            pkl.dump(gpt_player.battles, f)

        # summarize battles:
        win_cnt = 0
        total_cnt = 0
        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        turn_list = []
        for tag, battle in gpt_player.battles.items():
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

        print("battle #", i+1)
        print("except #:", gpt_player.except_cnt)
        print("accumulated input token #:", gpt_player.prompt_tokens)
        print("accumulated output token #:", gpt_player.completion_tokens)
        print("accumulated cost", gpt_player.prompt_tokens/1000000 * 2.5 + gpt_player.completion_tokens/1000000 * 10)

        print("battle score:", np.mean(beat_list) + np.mean(remain_list))
        print("win rate:", gpt_player.win_rate)
        print("turn:", np.mean(turn_list))

    print("Finished")

if __name__ == "__main__":
    entry_point()
