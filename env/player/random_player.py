"""This module defines a random players baseline
"""

from env.environment import AbstractBattle
from env.player.battle_order import BattleOrder
from env.player.player import Player


class RandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self.choose_random_move(battle)
