import asyncio
from tqdm import tqdm
import random
from src.client.account_configuration import AccountConfiguration
from src.player import HeuristicsPlayer

async def main():

    index = str(random.randint(0, 10000))
    player1 = HeuristicsPlayer(battle_format="gen6randombattle", account_configuration=AccountConfiguration(f"Player1_{index}", ""))
    player2 = HeuristicsPlayer(battle_format="gen6randombattle", account_configuration=AccountConfiguration(f"Player2_{index}", ""))
    # dynamax is disabled for local battles.
    player1._dynamax_disable = True
    player2._dynamax_disable = True

    # play against bot for five battles
    for i in tqdm(range(2000)):
        await player1.battle_against(player2, n_battles=1)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
