import asyncio
from tqdm import tqdm
import numpy as np
import os
import pickle as pkl
import json
import argparse
import random
from src.client.account_configuration import AccountConfiguration

from src.player import LLMPlayer, HeuristicsPlayer, RandomPlayer

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="gpt-4o-2024-08-06", choices=["gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4-1106", "gpt-4-0125"])
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--prompt_algo", default="io", choices=["io", "sc", "cot", "tot"])
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_bot")
args = parser.parse_args()

async def main():

    os.makedirs(args.log_dir, exist_ok=True)
    index = str(random.randint(0, 10000))
    heuristic_player1 = HeuristicsPlayer(battle_format="gen8randombattle",
                                         account_configuration=AccountConfiguration("HeuristicPlayer2WA", ""))
    # heuristic_player2 = HeuristicsPlayer(battle_format="gen8randombattle",
    #                                      account_configuration=AccountConfiguration("HeuristicPlayer2WB", ""))
    heuristic_player2 = RandomPlayer(battle_format="gen8randombattle",
                                         account_configuration=AccountConfiguration("RandomPlayer2WB", ""))
    # dynamax is disabled for local battles.
    heuristic_player1._dynamax_disable = True
    heuristic_player2._dynamax_disable = True

    # play against bot for five battles
    for i in tqdm(range(2000)):
        x = np.random.randint(0, 100)
        if x > 50:
            await heuristic_player1.battle_against(heuristic_player2, n_battles=1)
        else:
            await heuristic_player2.battle_against(heuristic_player1, n_battles=1)

    # for battle_id, battle in heuristic_player1.battles.items():
    #     with open(f"battle_data/self_play_heuristic/label_2w.jsonl", "a") as f:
    #         if battle.won:
    #             result_log = {"battle_id": battle_id, "winner": heuristic_player1.username, "loser": heuristic_player2.username}
    #         elif battle.lost:
    #             result_log = {"battle_id": battle_id, "winner": heuristic_player2.username, "loser": heuristic_player1.username}
    #         else:
    #             continue
    #         f.write(json.dumps(result_log) + "\n")

    print(heuristic_player1.win_rate)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
