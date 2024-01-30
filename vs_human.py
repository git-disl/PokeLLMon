import asyncio
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import LLMPlayer
import pickle as pkl
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./battle_log/pokellmon_vs_human")
args = parser.parse_args()

async def main():

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

    # Playing 5 games on the ladder
    for i in tqdm(range(50)):
        try:
            await llm_player.ladder(1)
            for battle_id, battle in llm_player.battles.items():
                with open(f"{args.log_dir}/{battle_id}.pkl", "wb") as f:
                    pkl.dump(battle, f)
        except:
            continue

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())