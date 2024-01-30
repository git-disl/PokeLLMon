import asyncio
import time
from tqdm import tqdm
import numpy as np
from poke_env import AccountConfiguration, ShowdownServerConfiguration
import os
import pickle as pkl
import argparse

from poke_env.player import LLMPlayer, SimpleHeuristicsPlayer

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
args = parser.parse_args()


async def main():

    heuristic_player = SimpleHeuristicsPlayer(battle_format="gen8randombattle",
                                              # account_configuration=AccountConfiguration("Heuristic Bot", "XXX"),
                                              # server_configuration=ShowdownServerConfiguration,
                                              )

    # backend = "gpt-4-1106-preview"
    backend = "gpt-4-0125-preview"
    os.makedirs(args.log_dir, exist_ok=True)
    llm_player = LLMPlayer(battle_format="gen8randombattle",
                           backend=backend,
                           temperature=0.8,
                           log_dir=args.log_dir,
                           w_reason=True,
                           account_configuration=AccountConfiguration("Your Account", "Your Password"),
                           # server_configuration=ShowdownServerConfiguration,
                           save_replays=args.log_dir
                           )

    for i in tqdm(range(10)):
        x = np.random.randint(0, 100)
        if x > 50:
            await heuristic_player.battle_against(llm_player, n_battles=1)
        else:
            await llm_player.battle_against(heuristic_player, n_battles=1)
        for battle_id, battle in llm_player.battles.items():
            with open(f"{args.log_dir}/{battle_id}.pkl", "wb") as f:
                pkl.dump(battle, f)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
