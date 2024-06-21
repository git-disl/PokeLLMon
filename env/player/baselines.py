from typing import List
import json
import os

from env.environment.abstract_battle import AbstractBattle
from env.environment.double_battle import DoubleBattle
from env.environment.move_category import MoveCategory
from env.environment.pokemon import Pokemon
from env.environment.side_condition import SideCondition
from env.player.player import Player
from env.data.gen_data import GenData

with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
    move_effect = json.load(f)

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

    def calc_reward(
            self, current_battle: AbstractBattle
    ) -> float:
        # Calculate the reward
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def choose_move(self, battle: AbstractBattle):
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)

        # calculate reward for the last step
        # last_action_reward = self.calc_reward(battle)
        self.w_reason = False
        self.gen = GenData.from_format(self.format)
        # with open("./poke_env/data/static/moves/gen8moves_effect.json", "r") as f:
        #     self.move_effect = json.load(f)
        # with open("./poke_env/data/static/moves/gen8_pokemon_move_dict.json", "r") as f:
        #     self.pokemon_move_dict = json.load(f)

        # state_prompt = self.state_translate(battle) # for dump data
        # dump_log = {"prompt":state_prompt.lower() + "Output:"}
        # abilities

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

        # try:
        #     self.pokemon_move_dict[mon.species]
        # except:

        # pokemon_ability_dict = {}
        # for pokemon_name, ability_set in self.pokemon_ability_dict.items():
        #     pokemon_ability_dict[pokemon_name] = list(ability_set)
        #
        # with open("./poke_env/data/static/abilities/gen7pokemon_ability_dict.json", "w") as f:
        #     json.dump(pokemon_ability_dict, f)
        #
        # pokemon_item_dict = {}
        # for pokemon_name, item_set in self.pokemon_item_dict.items():
        #     pokemon_item_dict[pokemon_name] = list(item_set - {'', None})
        #
        # with open("./poke_env/data/static/items/gen7pokemon_item_dict.json", "w") as f:
        #     json.dump(pokemon_item_dict, f)
        #
        # with open("./poke_env/data/static/moves/gen7pokemon_move_dict.json", "w") as f:
        #     json.dump(self.pokemon_move_dict, f, indent=4)

        with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
            self.move_effect = json.load(f)
        with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
            self.ability_effect = json.load(f)
        with open("./poke_env/data/static/items/item_effect.json", "r") as f:
            self.item_effect = json.load(f)

        set(self.move_effect.keys())

        set(self.ability_effect.keys())

        self.item_set - set(self.item_effect.keys())
        self.ability_set - set(self.ability_effect.keys())
        self.move_set - set(self.move_effect.keys())

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
            # action = next_action.message.split(" ")[1]
            # object = next_action.message.split(" ")[2]
            #
            # if action == "switch":
            #     dump_log.update({"output": '{"' + action + '": "' + object + '"}'})
            # if action == "move":
            #     dump_log.update(
            #         {"output": '{"' + action + '": "' + object + '", "dynamax": "' + str(next_action.dynamax) + '"}'})
            #
            # dump_log_dir = "/Users/husihao/Documents/PokemonProject/PokeLLMon/battle_log"
            # if dump_log_dir:
            #     with open(os.path.join(dump_log_dir, "heuristic_battle_log.jsonl"), "a") as f:
            #         f.write(json.dumps(dump_log) + "\n")
            pass

        else:
            next_action = self.choose_random_move(battle)

        return next_action

    def state_translate(self, battle: AbstractBattle):

        system_prompt = "You are a pokemon master that targets to win the pokemon battle.\n"
        n_turn = 5
        if "p1" in list(battle.team.keys())[0]:
            context_prompt = f"Historical turns:\n" + "\n".join(battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).replace("p1a: ", "").replace("p2a:","opposing").replace("Player1", "You").replace("Player2", "Opponent")
        else:
            context_prompt = f"Historical turns:\n" + "\n".join(battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).replace("p2a: ", "").replace("p1a:","opposing").replace("Player2", "You").replace("Player1", "Opponent")

        if battle.active_pokemon.fainted:
            battle_prompt = system_prompt + context_prompt + f" Your {battle.active_pokemon.species} fainted. You need to decide which pokemon to switch.\nCurrent battle state:\n"
        else:
            battle_prompt = system_prompt + context_prompt + " You need to decide which action to take.\nCurrent battle state:\n"

        # number of fainted pokemon
        opponent_fainted_num = 0
        for _, opponent_pokemon in battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1

        opponent_unfainted_num = 6 - opponent_fainted_num
        opponent_hp_fraction = round(battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp * 100)
        opponent_base_states = battle.opponent_active_pokemon._base_stats
        opponent_boosts = battle.opponent_active_pokemon._boosts
        opponent_status = battle.opponent_active_pokemon.status
        opponent_is_dynamax = battle.opponent_active_pokemon.is_dynamaxed

        # Type information
        opponent_type = ""

        type_1 = None
        type_2 = None
        opponent_type_list = []
        if battle.opponent_active_pokemon.type_1:
            type_1 = battle.opponent_active_pokemon.type_1.name
            opponent_type += type_1
            opponent_type_list.append(type_1)

            if battle.opponent_active_pokemon.type_2:
                type_2 = battle.opponent_active_pokemon.type_2.name
                opponent_type = opponent_type + " and " + type_2
                opponent_type_list.append(type_2)

        opponent_prompt = (
                f"Opponent has {opponent_unfainted_num} unfainted pokemons. " +
                f"Opponent current pokemon: {battle.opponent_active_pokemon.species}, {opponent_type}, HP: {opponent_hp_fraction}%, Is dynamax: {opponent_is_dynamax}, Status: {self.check_status(opponent_status)}. " +
                f"Attack: {opponent_base_states['atk']}, Defense: {opponent_base_states['def']}, Special attack: {opponent_base_states['spa']}, Special defense: {opponent_base_states['spd']}, Speed: {opponent_base_states['spe']}."
        )

        ability_list = ["atk", "def", "spa", "spd", "spe"]
        opponent_boost_list = []
        for ability in ability_list:
            if opponent_boosts[ability] != 0:
                multiplier = str(int(self.boost_multiplier(ability, opponent_boosts[ability]) * 100))
                if ability == "atk":
                    opponent_boost_list.append(f"attack: {opponent_boosts[ability]} (*{multiplier}%)")
                elif ability == "def":
                    opponent_boost_list.append(f"defense: {opponent_boosts[ability]} (*{multiplier}%)")
                elif ability == "spa":
                    opponent_boost_list.append(f"special attack: {opponent_boosts[ability]} (*{multiplier}%)")
                elif ability == "spd":
                    opponent_boost_list.append(f"speical defense: {opponent_boosts[ability]} (*{multiplier}%)")
                elif ability == "spe":
                    opponent_boost_list.append(f"speed: {opponent_boosts[ability]} (*{multiplier}%)")

        opponent_boost_prompt = ", ".join(opponent_boost_list)

        if opponent_boost_prompt:
            opponent_prompt = opponent_prompt + " Boosts: " + opponent_boost_prompt + "."

        opponent_move_type_damage_prompt = move_type_damage_wraper(battle.opponent_active_pokemon.species, type_1, type_2, self.gen.type_chart, None)

        if opponent_move_type_damage_prompt:
            opponent_prompt = opponent_prompt + " " + opponent_move_type_damage_prompt + ".\n"

        # Opponent active pokemon move
        if battle.opponent_active_pokemon.moves:
            opponent_move_prompt = f"Moves already used by {battle.opponent_active_pokemon.species}:"
            for move_id, opponent_move in battle.opponent_active_pokemon.moves.items():
                if opponent_move.base_power == 0:
                    continue # only count attack move
                opponent_move_prompt += f" [{opponent_move.id}, {opponent_move.type.name}, Power: {opponent_move.base_power}],"
                opponent_type_list.append(opponent_move.type.name)
            opponent_prompt = opponent_prompt + opponent_move_prompt + "\n"

        opponent_side_condition_list = [] # I should add the description for the side condition. and the status.
        for side_condition in battle.opponent_side_conditions:
            opponent_side_condition_list.append(" ".join(side_condition.name.lower().split("_")))

        opponent_side_condition = ",".join(opponent_side_condition_list)
        if opponent_side_condition:
            opponent_prompt = opponent_prompt + "Opponent team's side condition: " + opponent_side_condition + "\n"

        # The active pokemon
        active_hp_fraction = round(battle.active_pokemon.current_hp / battle.active_pokemon.max_hp * 100)
        active_status = battle.active_pokemon.status
        active_base_states = battle.active_pokemon._base_stats
        active_boosts = battle.active_pokemon._boosts

        active_type = ""
        type_1 = None
        type_2 = None

        if battle.active_pokemon.type_1:
            type_1 = battle.active_pokemon.type_1.name
            active_type += type_1

            if battle.active_pokemon.type_2:
                type_2 = battle.active_pokemon.type_2.name
                active_type = active_type + " and " + type_2

        active_move_type_damage_prompt = move_type_damage_wraper(battle.active_pokemon.species, type_1, type_2, self.gen.type_chart, opponent_type_list)

        active_pokemon_prompt = (f"Your current pokemon: {battle.active_pokemon.species}, {active_type}, HP: {active_hp_fraction}%, Status: {self.check_status(active_status)}. "
                                 f"Attack: {active_base_states['atk']}, Defense: {active_base_states['def']}, Special attack: {active_base_states['spa']}, Special defense: {active_base_states['spd']}, Speed: {active_base_states['spe']}.")

        rela_attack = active_base_states['atk'] * self.boost_multiplier('atk', active_boosts['atk']) / (opponent_base_states['def'] * self.boost_multiplier('def', opponent_boosts['def']))
        rela_defense = active_base_states['def'] * self.boost_multiplier('def', active_boosts['def']) / (opponent_base_states['atk'] * self.boost_multiplier('atk', opponent_boosts['atk']))
        rela_spe_attack = active_base_states['spa'] * self.boost_multiplier('spa', active_boosts['spa']) / (opponent_base_states['spd'] * self.boost_multiplier('spd', opponent_boosts['spd']))
        rela_spe_defense = active_base_states['spd'] * self.boost_multiplier('spd', active_boosts['spd']) / (opponent_base_states['spa'] * self.boost_multiplier('spa', opponent_boosts['spa']))
        rela_speed = active_base_states['spe'] * self.boost_multiplier('spe', active_boosts['spe']) / (opponent_base_states['spe'] * self.boost_multiplier('spe', opponent_boosts['spe']))

        ability_list = ["atk", "def", "spa", "spd", "spe"]
        active_boost_list = []
        for ability in ability_list:
            if active_boosts[ability]!=0:
                multiplier = str(int(self.boost_multiplier(ability, active_boosts[ability]) * 100))
                if ability == "atk":
                    active_boost_list.append(f"attack: {active_boosts[ability]} (*{multiplier}%)")
                elif ability == "def":
                    active_boost_list.append(f"defense: {active_boosts[ability]} (*{multiplier}%)")
                elif ability == "spa":
                    active_boost_list.append(f"special attack: {active_boosts[ability]} (*{multiplier}%)")
                elif ability == "spd":
                    active_boost_list.append(f"special defense: {active_boosts[ability]} (*{multiplier}%)")
                elif ability == "spe":
                    active_boost_list.append(f"speed: {active_boosts[ability]} (*{multiplier}%)")

        active_boost_prompt = ", ".join(active_boost_list)

        if active_boost_prompt:
            active_pokemon_prompt = active_pokemon_prompt + " Boost: " + active_boost_prompt + ". Note that all the boost will be reset when pokemon switch out."

        if active_move_type_damage_prompt:
            active_pokemon_prompt = active_pokemon_prompt + " " + active_move_type_damage_prompt + ".\n"

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

            side_condition_name = side_condition_name + effect
            side_condition_list.append(side_condition_name)

        side_condition_prompt = ",".join(side_condition_list)

        if side_condition_prompt:
            active_pokemon_prompt = active_pokemon_prompt + "Your team's side condition: " + side_condition_prompt + "\n"

        # Move
        move_prompt = f" Your {battle.active_pokemon.species} has {len(battle.available_moves)} moves:\n"
        for i, move in enumerate(battle.available_moves):
            try:
                effect = self.move_effect[move.id]
            except:
                effect = ""

            move_prompt += f"Move: {move.id}, {move.type.name}, Class: {move.category.name.lower()}, Power: {move.base_power}, Accuracy: {round(move.accuracy * self.boost_multiplier('accuracy', active_boosts['accuracy'])*100)}%"
            if effect:
                move_prompt += f", Effect: {effect}\n"
            else:
                move_prompt += "\n"

        # Switch
        if battle.active_pokemon.fainted:
            switch_prompt = f"You have {len(battle.available_switches)} pokemons that can be switched:\n"
        else:
            switch_prompt = f"Besides taking moves, you have {len(battle.available_switches)} pokemons that can be switched:\n"

        for i, pokemon in enumerate(battle.available_switches):

            type_1 = None
            type_2 = None
            type = ""
            if pokemon.type_1:
                type_1 = pokemon.type_1.name
                type += type_1
                if pokemon.type_2:
                    type_2 = pokemon.type_2.name
                    type = type + " and " + type_2

            hp_fraction = round(pokemon.current_hp / pokemon.max_hp * 100)

            base_states = pokemon._base_stats
            rela_attack = base_states['atk'] / (opponent_base_states['def'] * self.boost_multiplier('def', opponent_boosts['def']))
            rela_defense = base_states['def'] / (opponent_base_states['atk'] * self.boost_multiplier('atk', opponent_boosts['atk']))
            rela_spe_attack = base_states['spa'] / (opponent_base_states['spd'] * self.boost_multiplier('spd', opponent_boosts['spd']))
            rela_spe_defense = base_states['spd'] / (opponent_base_states['spa'] * self.boost_multiplier('spa', opponent_boosts['spa']))
            rela_speed = base_states['spe'] / (opponent_base_states['spe'] * self.boost_multiplier('spe', opponent_boosts['spe']))

            switch_move_prompt = f" Moves:"
            for _, move in pokemon.moves.items():
                if move.base_power == 0:
                    continue # only output attack move
                switch_move_prompt += f" [{move.id}, {move.type.name}, Power: {move.base_power}],"
                # switch_prompt += switch_move_prompt

            switch_prompt += (f"Pokemon: {pokemon.species}, {type}, HP: {hp_fraction}%, Status: {self.check_status(pokemon.status)}, " +
                              f"Attack: {base_states['atk']}, Defense: {base_states['def']}, Special attack: {base_states['spa']}, Special defense: {base_states['spd']}, Speed: {base_states['spe']}."
                              + switch_move_prompt)

                              # f" Ability (times): attack: {round(rela_attack,2)}, defense: {round(rela_defense,2)}, special attack: {round(rela_spe_attack,2)}, special defense: {round(rela_spe_defense,2)}, speed: {round(rela_speed,2)}.")

            pokemon_move_type_damage_prompt = move_type_damage_wraper(pokemon.species, type_1, type_2,self.gen.type_chart, opponent_type_list)

            if pokemon_move_type_damage_prompt:
                switch_prompt = switch_prompt + " " + pokemon_move_type_damage_prompt + "\n"
            else:
                switch_prompt += "\n"

        if battle.active_pokemon.fainted:
            if self.w_reason:
                constraint_prompt = '''Your output MUST strictly adhere the JSON format: {"switch":"<switch_pokemon_name>", "reason":"<reason>"}\n'''
            else:
                constraint_prompt = '''Your output MUST strictly adhere the JSON format: {"switch":"<switch_pokemon_name>"}\n'''
            state_prompt = battle_prompt + opponent_prompt + switch_prompt + constraint_prompt
        else:
            dynamax_prompt = ""
            if battle.can_dynamax and not battle.active_pokemon.is_dynamaxed:
                dynamax_prompt = f"If choose move, you can Dynamax {battle.active_pokemon.species} to boost its moves for three turns. Dynamax is a powerful one-time option, so use it strategically.\n"
                if self.w_reason:
                    constraint_prompt = '''You should choose the best action and provide reasoning by thinking step by step. Your output MUST strictly adhere to the format: {"move":"<move_name>", "dynamax":"<true_or_false>", "reason":"<reason>"} or {"switch":"<switch_pokemon_name>", "reason":"<reason>"}\n'''
                else:
                    constraint_prompt = '''You should choose the best action and the output MUST strictly adhere to the format: {"move":"<move_name>", "dynamax":"<true_or_false>"} or {"switch":"<switch_pokemon_name>"}\n'''
            else:
                if self.w_reason:
                    constraint_prompt = '''You should choose the best action and provide reasoning by thinking step by step. Your output MUST strictly adhere to the format: {"move":"<move_name>", "reason":"<reason>"} or {"switch":"<switch_pokemon_name>", "reason":"<reason>"}\n'''
                else:
                    constraint_prompt = '''You should choose the best action and the output MUST strictly adhere to the format: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''

            state_prompt = battle_prompt + opponent_prompt + active_pokemon_prompt + move_prompt + switch_prompt + dynamax_prompt + constraint_prompt

        return state_prompt

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
                return "sleeping"
        else:
            return "healthy"

