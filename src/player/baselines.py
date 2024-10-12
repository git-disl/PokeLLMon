from typing import List
import json
import os

from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.environment.move_category import MoveCategory
from src.environment.pokemon import Pokemon
from src.environment.side_condition import SideCondition
from src.player.player import Player
from src.data.gen_data import GenData
from src.player.battle_order import BattleOrder


def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

    move_type_damage_multiplier_list = []

    if type_2:
        for type in TYPE_list:
            move_type_damage_multiplier_list.append(type_chart[type_1][type] * type_chart[type_2][type])
        move_type_damage_multiplier_dict = dict(zip(TYPE_list, move_type_damage_multiplier_list))
    else:
        move_type_damage_multiplier_dict = type_chart[type_1]

    effective_type_list = []
    extreme_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    for type, value in move_type_damage_multiplier_dict.items():
        if value == 2:
            effective_type_list.append(type)
        elif value == 4:
            extreme_type_list.append(type)
        elif value == 1 / 2:
            resistant_type_list.append(type)
        elif value == 1 / 4:
            extreme_resistant_type_list.append(type)
        elif value == 0:
            immune_type_list.append(type)
        else:  # value == 1
            continue

    if constraint_type_list:
        extreme_type_list = list(set(extreme_type_list).intersection(set(constraint_type_list)))
        effective_type_list = list(set(effective_type_list).intersection(set(constraint_type_list)))
        resistant_type_list = list(set(resistant_type_list).intersection(set(constraint_type_list)))
        extreme_resistant_type_list = list(set(extreme_resistant_type_list).intersection(set(constraint_type_list)))
        immune_type_list = list(set(immune_type_list).intersection(set(constraint_type_list)))

    return extreme_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list


def move_type_damage_wraper(pokemon_name, type_1, type_2, type_chart, constraint_type_list=None):

    move_type_damage_prompt = ""
    extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list)

    if effective_type_list or resistant_type_list or immune_type_list:

        move_type_damage_prompt = f"{pokemon_name}"
        if extreme_effective_type_list:
            move_type_damage_prompt = move_type_damage_prompt + " can be super-effectively attacked by " + ", ".join(
                extreme_effective_type_list) + " moves"
        if effective_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", can be effectively attacked by " + ", ".join(
                effective_type_list) + " moves"
        if resistant_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is resistant to " + ", ".join(
                resistant_type_list) + " moves"
        if extreme_resistant_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is super-resistant to " + ", ".join(
                extreme_resistant_type_list) + " moves"
        if immune_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is immuned to " + ", ".join(
                immune_type_list) + " moves"

    return move_type_damage_prompt


class RandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self.choose_random_move(battle)


class MaxBasePowerPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)

class HeuristicsPlayer(Player):
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_dynamax(self, battle: AbstractBattle, n_remaining_mons: int):
        if battle.can_dynamax and self._dynamax_disable is False:
            # Last full HP mon
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # Matchup advantage and full hp on full hp
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False

    def _should_switch_out(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        # If there is a decent switch in...
        if [
            m
            for m in battle.available_switches
            if self._estimate_matchup(m, opponent) > 0
        ]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                self._estimate_matchup(active, opponent)
                < self.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        return False

    def _stat_estimation(self, mon: Pokemon, stat: str):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    def check_status(self, status):
        if status:
            if status.value == 1:
                return "burnt"
            elif status.value == 2:
                return "fainted"
            elif status.value == 3:
                return "frozen"
            elif status.value == 4:
                return "paralyzed"
            elif status.value == 5:
                return "poisoned"
            elif status.value == 7:
                return "toxic"
            elif status.value == 6:
                return "asleep"
        else:
            return ""

    def boost_multiplier(self, state, level):
        if state == "accuracy":
            if level == 0:
                return 1.0
            if level == 1:
                return 1.33
            if level == 2:
                return 1.66
            if level == 3:
                return 2.0
            if level == 4:
                return 2.5
            if level == 5:
                return 2.66
            if level == 6:
                return 3.0
            if level == -1:
                return 0.75
            if level == -2:
                return 0.6
            if level == -3:
                return 0.5
            if level == -4:
                return 0.43
            if level == -5:
                return 0.36
            if level == -6:
                return 0.33
        else:
            if level == 0:
                return 1.0
            if level == 1:
                return 1.5
            if level == 2:
                return 2.0
            if level == 3:
                return 2.5
            if level == 4:
                return 3.0
            if level == 5:
                return 3.5
            if level == 6:
                return 4.0
            if level == -1:
                return 0.67
            if level == -2:
                return 0.5
            if level == -3:
                return 0.4
            if level == -4:
                return 0.33
            if level == -5:
                return 0.29
            if level == -6:
                return 0.25

    def state_translate(self, battle: AbstractBattle):

        n_turn = 0
        if "p1" in list(battle.team.keys())[0]:
            context_prompt = (f"Historical turns:\n" + "\n".join(
                battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).
                                          replace("p1a: ", "").
                                          replace("p2a:","opposing").
                                          replace("Player1", "You").
                                          replace("Player2", "Opponent"))
        else:
            context_prompt = (f"Historical turns:\n" + "\n".join(
                battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).
                              replace("p2a: ", "").
                              replace("p1a:", "opposing").
                              replace("Player2", "You").
                              replace("Player1", "Opponent"))

        if n_turn:
            battle_prompt = context_prompt + " (Current turn):\n"
                             # + "\nCurrent battle state:\n"
        else:
            battle_prompt = ""

        # number of fainted pokemon
        opponent_fainted_num = 0
        opponent_unfaint_inactive_pokemons = []
        for _, opponent_pokemon in battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1
            elif opponent_pokemon.active is False:
                opponent_unfaint_inactive_pokemons.append(opponent_pokemon.species)
        opponent_unfaint_inactive_pokemons = ",".join(opponent_unfaint_inactive_pokemons)

        opponent_unfainted_num = 6 - opponent_fainted_num
        opponent_hp_fraction = round(battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp * 100)
        opponent_stats = battle.opponent_active_pokemon.calculate_stats()
        opponent_boosts = battle.opponent_active_pokemon._boosts
        active_stats = battle.active_pokemon.stats
        active_boosts = battle.active_pokemon._boosts
        opponent_status = battle.opponent_active_pokemon.status
        # opponent_is_dynamax = battle.opponent_active_pokemon.is_dynamaxed

        # Type information
        opponent_type = ""

        opponent_type_list = []
        if battle.opponent_active_pokemon.type_1:
            type_1 = battle.opponent_active_pokemon.type_1.name
            opponent_type += type_1
            opponent_type_list.append(type_1)

            if battle.opponent_active_pokemon.type_2:
                type_2 = battle.opponent_active_pokemon.type_2.name
                opponent_type = opponent_type + "&" + type_2
                opponent_type_list.append(type_2)

        opponent_prompt = (
                f"Opponent has {opponent_unfainted_num} pokemons left." +
                (f" Opponent's known pokemon off the field:{opponent_unfaint_inactive_pokemons}\n" if len(opponent_unfaint_inactive_pokemons) else "\n") +
                f"Opponent current pokemon:{battle.opponent_active_pokemon.species}:Type:{opponent_type},HP:{opponent_hp_fraction}%," +
                (f"Status:{self.check_status(opponent_status)}," if self.check_status(opponent_status) else "") +
                (f"Atk:{opponent_stats['atk']}," if opponent_boosts['atk']==0 else f"Atk:{round(opponent_stats['atk'] * self.boost_multiplier('atk', opponent_boosts['atk']))}({opponent_boosts['atk']} stage),") +
                (f"Def:{opponent_stats['def']}," if opponent_boosts['def']==0 else f"Def:{round(opponent_stats['def'] * self.boost_multiplier('def', opponent_boosts['def']))}({opponent_boosts['def']} stage),") +
                (f"Spa:{opponent_stats['spa']}," if opponent_boosts['spa']==0 else f"Spa:{round(opponent_stats['spa'] * self.boost_multiplier('spa', opponent_boosts['spa']))}({opponent_boosts['spa']} stage),") +
                (f"Spd:{opponent_stats['spd']}," if opponent_boosts['spd']==0 else f"Spd:{round(opponent_stats['spd'] * self.boost_multiplier('spd', opponent_boosts['spd']))}({opponent_boosts['spd']} stage),") +
                (f"Spe:{opponent_stats['spe']}" if opponent_boosts['spe'] == 0 else f"Spe:{round(opponent_stats['spe'] * self.boost_multiplier('spe', opponent_boosts['spe']))}({opponent_boosts['spe']} stage)")
        )

        team_move_type = []
        for move in battle.available_moves:
            if move.base_power > 0:
                team_move_type.append(move.type.name)

        for pokemon in battle.available_switches:
            for move in pokemon.moves.values():
                if move.base_power > 0:
                    team_move_type.append(move.type.name)

        # Opponent active pokemon move
        opponent_move_prompt = ""
        if battle.opponent_active_pokemon.moves:
            for move_id, opponent_move in battle.opponent_active_pokemon.moves.items():
                if opponent_move.base_power == 0:
                    continue # only show attack move

                opponent_move_prompt += f"[{opponent_move.id},{opponent_move.type.name.capitalize()},Power:{opponent_move.base_power}],"
                opponent_type_list.append(opponent_move.type.name)

        opponent_side_condition_list = []
        for side_condition in battle.opponent_side_conditions:
            opponent_side_condition_list.append(" ".join(side_condition.name.lower().split("_")))

        opponent_side_condition = ",".join(opponent_side_condition_list)
        if opponent_side_condition:
            opponent_prompt = opponent_prompt + ",Opponent side condition:" + opponent_side_condition

        opponent_prompt += "\n"

        # The active pokemon
        active_hp_fraction = round(battle.active_pokemon.current_hp / battle.active_pokemon.max_hp * 100)
        active_status = battle.active_pokemon.status

        active_type = ""
        if battle.active_pokemon.type_1:
            active_type += battle.active_pokemon.type_1.name
            if battle.active_pokemon.type_2:
                active_type = active_type + "&" + battle.active_pokemon.type_2.name

        active_pokemon_prompt = (
            f"Your current pokemon:{battle.active_pokemon.species},Type:{active_type},HP:{active_hp_fraction}%," +
            (f"Status:{self.check_status(active_status)}," if self.check_status(active_status) else "" ) +
            (f"Atk:{active_stats['atk']}," if active_boosts['atk'] == 0 else f"Atk:{round(active_stats['atk'] * self.boost_multiplier('atk', active_boosts['atk']))}({active_boosts['atk']} stage),") +
            (f"Def:{active_stats['def']}," if active_boosts['def'] == 0 else f"Def:{round(active_stats['def'] * self.boost_multiplier('def', active_boosts['def']))}({active_boosts['def']} stage),") +
            (f"Spa:{active_stats['spa']}," if active_boosts['spa'] == 0 else f"Spa:{round(active_stats['spa'] * self.boost_multiplier('spa', active_boosts['spa']))}({active_boosts['spa']} stage),") +
            (f"Spd:{active_stats['spd']}," if active_boosts['spd'] == 0 else f"Spd:{round(active_stats['spd'] * self.boost_multiplier('spd', active_boosts['spd']))}({active_boosts['spd']} stage),") +
            (f"Spe:{active_stats['spe']}" if active_boosts['spe']==0 else f"Spe:{round(active_stats['spe']*self.boost_multiplier('spe', active_boosts['spe']))}({active_boosts['spe']} stage)")
        )

        side_condition_list = []
        for side_condition in battle.side_conditions:

            side_condition_name = " ".join(side_condition.name.lower().split("_"))
            if side_condition == SideCondition.SPIKES:
                effect = " (cause damage to your pokémon when switch in except flying type)"
            elif side_condition == SideCondition.STEALTH_ROCK:
                effect = " (cause rock-type damage to your pokémon when switch in)"
            elif side_condition == SideCondition.STICKY_WEB:
                effect = " (reduce the speed stat of your pokémon when switch in)"
            elif side_condition == SideCondition.TOXIC_SPIKES:
                effect = " (cause your pokémon toxic when switch in)"
            else:
                effect = ""

            # if knowledge:
                # side_condition_name = side_condition_name + effect
            side_condition_list.append(side_condition_name)

        side_condition_prompt = ",".join(side_condition_list)

        if side_condition_prompt:
            active_pokemon_prompt = active_pokemon_prompt + ", your side condition: " + side_condition_prompt + "\n"
        else:
            active_pokemon_prompt += "\n"

        # Move
        move_prompt = f"Your {battle.active_pokemon.species} has {len(battle.available_moves)} moves can take:\n"
        for i, move in enumerate(battle.available_moves):

            if move.category.name == "SPECIAL":
                active_spa = active_stats["spa"] * self.boost_multiplier("spa", active_boosts["spa"])
                opponent_spd = opponent_stats["spd"] * self.boost_multiplier("spd", active_boosts["spd"])
                power = round(active_spa / opponent_spd * move.base_power)
                move_category = move.category.name.capitalize()
            elif move.category.name == "PHYSICAL":
                active_atk = active_stats["atk"] * self.boost_multiplier("atk", active_boosts["atk"])
                opponent_def = opponent_stats["def"] * self.boost_multiplier("def", active_boosts["def"])
                power = round(active_atk / opponent_def * move.base_power)
                move_category = move.category.name.capitalize()
            else:
                move_category = move.category.name.capitalize()
                power = 0

            move_prompt += (f"{move.id}:Type:{move.type.name}," +
                            (f"Cate:{move_category}," if move_category else "") +
                            f"Power:{power},Acc:{round(move.accuracy * self.boost_multiplier('accuracy', active_boosts['accuracy'])*100)}%"
                            )
            # add knowledge
            # if self.config.knowledge:
            #     try:
            #         effect = self.move_effect[move.id]
            #     except:
            #         effect = ""
            #     move_prompt += f",$Effect:{effect}$\n"
            # else:
            move_prompt += "\n"

        # add knowledge
        # if self.config.knowledge:
        #     opponent_move_type_damage_prompt = move_type_damage_wraper(battle.opponent_active_pokemon,
        #                                                                self.gen.type_chart,
        #                                                                team_move_type)
        #     if opponent_move_type_damage_prompt:
        #         opponent_prompt = opponent_prompt + "$" + opponent_move_type_damage_prompt + "$" + "\n"
        #
        #     active_move_type_damage_prompt = move_type_damage_wraper(battle.active_pokemon, self.gen.type_chart,
        #                                                              opponent_type_list)
        #     if active_move_type_damage_prompt:
        #         active_pokemon_prompt = active_pokemon_prompt + "$" + active_move_type_damage_prompt+ "$" + "\n"

        # Switch
        if len(battle.available_switches) > 0:
            switch_prompt = f"You have {len(battle.available_switches)} pokemons can switch:\n"
        else:
            switch_prompt = f"You have no pokemon can switch:\n"

        for i, pokemon in enumerate(battle.available_switches):

            type = ""
            if pokemon.type_1:
                type_1 = pokemon.type_1.name
                type += type_1
                if pokemon.type_2:
                    type_2 = pokemon.type_2.name
                    type = type + "&" + type_2

            hp_fraction = round(pokemon.current_hp / pokemon.max_hp * 100)

            stats = pokemon.stats
            switch_move_list = []
            for _, move in pokemon.moves.items():
                if move.base_power == 0:
                    continue # only output attack move

                switch_move_list.append(f"[{move.id},{move.type.name}]")
            switch_move_prompt = ",".join(switch_move_list)

            switch_prompt += (
                        f"{pokemon.species}:Type:{type},HP:{hp_fraction}%," +
                        (f"Status:{self.check_status(pokemon.status)}, " if self.check_status(pokemon.status) else "") +
                        f"Atk:{stats['atk']},Def:{stats['def']},Spa:{stats['spa']},Spd:{stats['spd']}," +
                        (f"Spe:{stats['spe']}" + f",Moves:{switch_move_prompt}" if switch_move_prompt else "") +
                        "\n")

        system_prompt = "You are playing a Pokemon battle and the goal is to win\n"
        if battle.active_pokemon.fainted: # forced switch
            state_prompt = battle_prompt + opponent_prompt + switch_prompt
            return system_prompt, state_prompt

        else: # take a move or active switch
            state_prompt = battle_prompt + opponent_prompt + active_pokemon_prompt + move_prompt + switch_prompt
            return system_prompt, state_prompt


    def choose_move(self, battle: AbstractBattle):
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)

        system_prompt, state_prompt = self.state_translate(battle)
        dump_log = {"prompt": system_prompt + state_prompt + "Output:", "battle_tag": battle.battle_tag, "turn": battle.turn, "player": self.username}

        self.gen = GenData.from_format(self.format)
        for mon in battle.team.values():
            self.move_set = self.move_set.union(set(mon.moves.keys()))
            self.item_set.add(mon.item)
            self.ability_set.add(mon.ability)
            try:
                self.pokemon_item_dict[mon.species].add(mon.item)
            except:
                self.pokemon_item_dict[mon.species] = set()
                self.pokemon_item_dict[mon.species].add(mon.item)
            try:
                self.pokemon_ability_dict[mon.species].add(mon.ability)
            except:
                self.pokemon_ability_dict[mon.species] = set()
                self.pokemon_ability_dict[mon.species].add(mon.ability)
            for name, move in mon.moves.items():
                try:
                    self.pokemon_move_dict[mon.species][name][3] += 1
                except:
                    try:
                        self.pokemon_move_dict[mon.species][name] = [name, move.type.name, move.base_power, 1]
                    except:
                        self.pokemon_move_dict[mon.species] = {}
                        self.pokemon_move_dict[mon.species][name] = [name, move.type.name, move.base_power, 1]

        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
            opponent, "def"
        )
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
            opponent, "spd"
        )

        next_action = None
        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )
            n_opp_remaining_mons = 6 - len(
                [m for m in battle.opponent_team.values() if m.fainted is True]
            )

            # Entry hazard...
            for move in battle.available_moves:
                # ...setup
                if (
                    n_opp_remaining_mons >= 3
                    and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    next_action = self.create_order(move)
                    break

                # ...removal
                elif (
                    battle.side_conditions
                    and move.id in self.ANTI_HAZARDS_MOVES
                    and n_remaining_mons >= 2
                ):
                    next_action = self.create_order(move)
                    break

            # Setup moves
            if (
                next_action is None
                and active.current_hp_fraction == 1
                and self._estimate_matchup(active, opponent) > 0
            ):
                for move in battle.available_moves:
                    if (
                        self._boost_disable is False
                        and move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            [active.boosts[s] for s, v in move.boosts.items() if v > 0]
                        )
                        < 6
                    ):
                        next_action = self.create_order(move)
                        break

            if next_action is None:
                move = max(
                    battle.available_moves,
                    key=lambda m: m.base_power
                    * (1.5 if m.type in active.types else 1)
                    * (
                        physical_ratio
                        if m.category == MoveCategory.PHYSICAL
                        else special_ratio
                    )
                    * m.accuracy
                    * m.expected_hits
                    * opponent.damage_multiplier(m),
                )
                next_action = self.create_order(
                    move, dynamax=self._should_dynamax(battle, n_remaining_mons)
                )

        if next_action is None and battle.available_switches:
            switches: List[Pokemon] = battle.available_switches
            next_action = self.create_order(
                max(
                    switches,
                    key=lambda s: self._estimate_matchup(s, opponent),
                )
            )

        if next_action:
            action = next_action.message.split(" ")[1]
            object = next_action.message.split(" ")[2]

            if action == "switch":
                dump_log.update({"output": '{"' + action + '": "' + object + '"}'})
            if action == "move":
                # dump_log.update({"output": '{"' + action + '": "' + object + '", "dynamax": "' + str(next_action.dynamax) + '"}'})
                dump_log.update({"output": '{"' + action + '": "' + object + '"}'})

            # dump_log_dir = os.path.join("battle_data/self_play_heuristic")
            # os.makedirs(dump_log_dir, exist_ok=True)
            # with open(os.path.join(dump_log_dir, self.username + ".jsonl"), "a") as f:
            #     f.write(json.dumps(dump_log) + "\n")

        else:
            next_action = self.choose_random_move(battle)

        return next_action
