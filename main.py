import asyncio
import time
from tqdm import tqdm
import numpy as np
from env import AccountConfiguration
import os
import pickle as pkl
import argparse

from env.player import LLMPlayer, HeuristicsPlayer

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="gpt-4-0125", choices=["gpt-3.5-turbo-0125", "gpt-4-1106", "gpt-4-0125"])
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
args = parser.parse_args()

async def main():

    heuristic_player = HeuristicsPlayer(battle_format="gen8randombattle")

    os.makedirs(args.log_dir, exist_ok=True)
    llm_player = LLMPlayer(battle_format="gen8randombattle",
                           api_key=os.getenv("OPENAI_API_KEY"),
                           backend=args.backend,
                           temperature=args.temperature,
                           prompt_algo=args.prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration("Your_account", "Your_password"),
                           save_replays=args.log_dir
                           )

    # dynamax is disabled for local battles.
    heuristic_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    # play against bot for five battles
    for i in tqdm(range(5)):
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
