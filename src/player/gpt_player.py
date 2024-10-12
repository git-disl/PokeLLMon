import json
import os
import random
from typing import List
from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.environment.move_category import MoveCategory
from src.environment.pokemon import Pokemon
from src.environment.side_condition import SideCondition
from src.player.player import Player, BattleOrder
from typing import Dict, List, Optional, Union
from src.environment.move import Move
import time
import json
from openai import OpenAI
from src.data.gen_data import GenData

SYSTEM_PROMPT = "You are playing a Pokemon battle and the goal is to win\n"
def prompt_template(reason_algo, n_shot=0, forced_switch=False):
    if forced_switch:
        if reason_algo == "io":
            if n_shot == 1:
                with open("src/prompt/normal_forced_switch_1shot", "r") as f:
                    template = f.read()
                    return template
            elif n_shot == 3:
                with open("src/prompt/normal_forced_switch_3shot", "r") as f:
                    template = f.read()
                    return template
        elif reason_algo == "cot":
            if n_shot == 1:
                with open("src/prompt/cot_forced_switch_1shot", "r") as f:
                    template = f.read()
                    return template
            elif n_shot == 3:
                with open("src/prompt/cot_forced_switch_3shot", "r") as f:
                    template = f.read()
                    return template
        else:
            raise ValueError("algorithm error.")
    else:
        if reason_algo == "io":
            if n_shot == 1:
                with open("src/prompt/normal_act_1shot", "r") as f:
                    template = f.read()
                    return template
            elif n_shot == 3:
                with open("src/prompt/normal_act_3shot", "r") as f:
                    template = f.read()
                    return template

        elif reason_algo == "cot":
            if n_shot == 1:
                with open("src/prompt/cot_act_1shot", "r") as f:
                    template = f.read()
                    return template
            elif n_shot == 3:
                with open("src/prompt/cot_act_3shot", "r") as f:
                    template = f.read()
                    return template
        else:
            raise ValueError("algorithm error.")

    return "[INPUT]"

def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

    move_type_damage_multiplier_list = []

    if type_2: # being attack
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

    # return (list(map(lambda x: x.capitalize(), extreme_type_list)),
    #        list(map(lambda x: x.capitalize(), effective_type_list)),
    #        list(map(lambda x: x.capitalize(), resistant_type_list)),
    #        list(map(lambda x: x.capitalize(), extreme_resistant_type_list)),
    #        list(map(lambda x: x.capitalize(), immune_type_list)))

def move_type_damage_wraper(pokemon, defense_chart, attack_chart, constraint_type_list=None):

    type_1 = None
    type_2 = None
    if pokemon.type_1:
        type_1 = pokemon.type_1.name
        if pokemon.type_2:
            type_2 = pokemon.type_2.name

    # defense
    x4_types, x2_types, x1_2_types, x1_4_types, x0_types = calculate_move_type_damage_multipier(type_1,
                                                                                                type_2,
                                                                                                defense_chart,
                                                                                                constraint_type_list)

    move_type_damage_prompt = ""
    # defense
    descriptions = []
    if x4_types:
        descriptions.append(",".join(x4_types) + f" deal 4x damage")

    if x2_types:
        descriptions.append(",".join(x2_types) + f" deal 2x damage")

    if x1_2_types:
        descriptions.append(",".join(x1_2_types) + f" only deal 0.5x damage")

    if x1_4_types:
        descriptions.append(",".join(x1_4_types) + f" only deal 0.25x damage")

    if x0_types:
        descriptions.append(",".join(x0_types) + f" have no effect")

    if descriptions:
        move_type_damage_prompt += (f"{pokemon.species} as defender, " + "; ".join(descriptions) + f" to {pokemon.species}\n")

    # attack
    descriptions = []
    for attack_type in [type_1, type_2]:
        if attack_type is None:
            continue
        _, x2_types, x1_2_types, _, x0_types = calculate_move_type_damage_multipier(attack_type,
                                                                                    None,
                                                                                    attack_chart,
                                                                                    constraint_type_list)
        if x2_types:
            descriptions.append(f"{attack_type} deal 2x damage to " + ",".join(x2_types) + f" pokemon")

        if x1_2_types:
            descriptions.append(f"{attack_type} deal 0.5x damage to " + ",".join(x1_2_types) + f" pokemon")

        if x0_types:
            descriptions.append(f"{attack_type} have no effect to " + ",".join(x0_types) + f" pokemon")

    if descriptions:
        move_type_damage_prompt += (f"{pokemon.species} as attacker, " + "; ".join(descriptions) + "\n")

    return move_type_damage_prompt


class GPTPlayer(Player):
    def __init__(self,
                 config,
                 api_key="",
                 backend="",
                 reason_algo="",
                 save_replay_dir="",
                 account_configuration=None,
                 server_configuration=None):

        super().__init__(battle_format=config.battle_format,
                         team=None,
                         save_replays=save_replay_dir,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = config.temperature
        self.save_replay_dir = save_replay_dir
        self.api_key = api_key
        self.reason_algo = reason_algo
        self.config = config
        self.gen = GenData.from_format(config.battle_format)
        self.defense_chart = self.gen.type_chart
        self.attack_chart = {}
        for defense_type, defense_dict in self.defense_chart.items():
            for attack_type, multiplier in defense_dict.items():
                try:
                    self.attack_chart[attack_type][defense_type] = multiplier
                except KeyError:
                    self.attack_chart[attack_type] = {defense_type: multiplier}

        self.except_cnt = 0
        self.total_cnt = 0
        self.last_thought = ""
        self.last_state = ""
        self.last_step = "" # force switch, active switch, active action
        self.last_state_prompt = ""
        self.last_action = ""

        with open("src/data/static/moves/moves_effect.json", "r") as f:
            self.move_effect = json.load(f)
        with open("src/data/static/moves/gen8pokemon_move_dict.json", "r") as f:
            self.pokemon_move_dict = json.load(f)
        with open("src/data/static/abilities/ability_effect.json", "r") as f:
            self.ability_effect = json.load(f)
        with open("src/data/static/abilities/gen8pokemon_ability_dict.json", "r") as f:
            self.pokemon_ability_dict = json.load(f)
        with open("src/data/static/items/item_effect.json", "r") as f:
            self.item_effect = json.load(f)
        with open("src/data/static/items/gen8pokemon_item_dict.json", "r") as f:
            self.pokemon_item_dict = json.load(f)

        self.SPEED_TIER_COEFICIENT = 0.1
        self.HP_FRACTION_COEFICIENT = 0.4

    def chatgpt(self, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200) -> str:
        client = OpenAI(api_key=self.api_key)
        if json_format:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                stream=False,
                # seed=seed,
                stop=stop,
                max_tokens=max_tokens
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                stream=False,
                # seed=seed,
                max_tokens=max_tokens,
                stop=stop
            )
        outputs = response.choices[0].message.content
        # log completion tokens
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens

        return outputs

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

    def _should_dynamax(self, battle: AbstractBattle):
        n_remaining_mons = len(
            [m for m in battle.team.values() if m.fainted is False]
        )
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


    def state_translate(self, battle: AbstractBattle):

        n_turn = self.config.T
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
            if self.config.knowledge:
                try:
                    effect = self.move_effect[move.id]
                except:
                    effect = ""
                move_prompt += f",Effect:{effect}\n"
            else:
                move_prompt += "\n"

        # add knowledge
        if self.config.knowledge:
            opponent_move_type_damage_prompt = move_type_damage_wraper(battle.opponent_active_pokemon,
                                                                       self.defense_chart,
                                                                       self.attack_chart,
                                                                       team_move_type)
            if opponent_move_type_damage_prompt:
                opponent_prompt = opponent_prompt + opponent_move_type_damage_prompt

            # active_move_type_damage_prompt = move_type_damage_wraper(battle.active_pokemon, self.gen.type_chart,
            #                                                          opponent_type_list)
            # if active_move_type_damage_prompt:
            #     active_pokemon_prompt = active_pokemon_prompt + active_move_type_damage_prompt

        # Switch
        if len(battle.available_switches) > 0:
            switch_prompt = f"You have {len(battle.available_switches)} pokemons can switch:\n"
        else:
            switch_prompt = f"You have no pokemon left\n"

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


        if battle.force_switch:
            switch_prompt = f"This is a force switch, " + switch_prompt
            state_prompt = battle_prompt + opponent_prompt + switch_prompt
            return state_prompt

        else: # take a move or active switch
            state_prompt = battle_prompt + opponent_prompt + active_pokemon_prompt + move_prompt + switch_prompt
            return state_prompt

    def parse(self, llm_output, battle):
        json_start = llm_output.find('{')
        json_end = llm_output.find('}') + 1 # find the first }
        json_content = llm_output[json_start:json_end]
        llm_action_json = json.loads(json_content)
        next_action = None
        thought = ""
        if "move" in llm_action_json.keys():
            llm_move_id = llm_action_json["move"]
            llm_move_id = llm_move_id.replace(" ","").replace("-", "")
            for i, move in enumerate(battle.available_moves):
                if move.id.lower() == llm_move_id.lower():
                    next_action = self.create_order(move, dynamax=self._should_dynamax(battle))
                    self.last_step = "active_move"

        elif "switch" in llm_action_json.keys():
            llm_switch_species = llm_action_json["switch"]
            for i, pokemon in enumerate(battle.available_switches):
                if pokemon.species.lower() == llm_switch_species.lower():
                    next_action = self.create_order(pokemon)
                    if battle.force_switch:
                        self.last_step = "force_switch"
                    else:
                        self.last_step = "active_switch"

        if "thought" in llm_action_json.keys():
            thought = llm_action_json["thought"]

        if next_action is None:
            raise ValueError("Value Error")
        return next_action, thought

    def parse_new(self, llm_output, battle):
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}') + 1 # find the first }
        json_content = llm_output[json_start:json_end]
        llm_action_json = json.loads(json_content)
        next_action = None
        action = llm_action_json["decision"]["action"]
        target = llm_action_json["decision"]["target"]
        target = target.replace(" ", "").replace("_", "")
        if action.lower() == "move":
            for i, move in enumerate(battle.available_moves):
                if move.id.lower() == target.lower():
                    next_action = self.create_order(move, dynamax=self._should_dynamax(battle))

        elif action.lower() == "switch":
            for i, pokemon in enumerate(battle.available_switches):
                if pokemon.species.lower() == target.lower():
                    next_action = self.create_order(pokemon)

        if next_action is None:
            raise ValueError("Value Error")

        return next_action

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
    def gen_with_io(self, battle: AbstractBattle, state_prompt):

        if battle.force_switch:
            constraint_prompt = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            if self.config.n_shot == 1:
                with open("src/prompt/normal_forced_switch_1shot", "r") as f:
                    template = f.read()
            elif self.config.n_shot == 3:
                with open("src/prompt/normal_forced_switch_3shot", "r") as f:
                    template = f.read()
            else: # n_shot = 0
                template = "[INPUT]"
        else:
            constraint_prompt = '''Choose the best action. Your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''
            if self.config.n_shot == 1:
                with open("src/prompt/normal_act_1shot", "r") as f:
                    template = f.read()
            elif self.config.n_shot == 3:
                with open("src/prompt/normal_act_3shot", "r") as f:
                    template = f.read()
            else:
                template = "[INPUT]"

        state_prompt = template.replace("[INPUT]", state_prompt) + constraint_prompt

        next_action = None
        for i in range(2):
            try:
                llm_output = self.chatgpt(user_prompt=state_prompt,
                                          model=self.backend,
                                          temperature=self.temperature,
                                          max_tokens=self.config.max_output_length,
                                          json_format=True)
                print("LLM output:", llm_output)
                next_action, _ = self.parse(llm_output, battle)
                if self.save_replay_dir:
                    with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "algo": self.reason_algo,
                                            "system_prompt": SYSTEM_PROMPT,
                                            "user_prompt": state_prompt,
                                            "llm_output": llm_output,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                break
            except Exception as e:
                continue
        if next_action is None:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
        self.total_cnt += 1
        return next_action

    def gen_with_cot(self, battle: AbstractBattle, state_prompt):

        if battle.force_switch:
            constraint_prompt = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            if self.config.n_shot == 1:
                with open("src/prompt/cot_forced_switch_1shot", "r") as f:
                    template = f.read()
            elif self.config.n_shot == 3:
                with open("src/prompt/cot_forced_switch_3shot", "r") as f:
                    template = f.read()
        else:
            constraint_prompt = '''Choose the best action. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            if self.config.n_shot == 1:
                with open("src/prompt/cot_act_1shot", "r") as f:
                    template = f.read()
            elif self.config.n_shot == 3:
                with open("src/prompt/cot_act_3shot", "r") as f:
                    template = f.read()

        state_prompt = template.replace("[INPUT]", state_prompt) + constraint_prompt
        next_action = None
        for i in range(2):
            try:
                llm_output = self.chatgpt(user_prompt=state_prompt,
                                          model=self.backend,
                                          temperature=self.temperature,
                                          max_tokens=self.config.max_output_length,
                                          json_format=True)
                print("LLM output:", llm_output)
                next_action, thought = self.parse(llm_output, battle)
                # self.last_thought = thought
                if self.save_replay_dir:
                    with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "algo": self.reason_algo,
                                            "system_prompt": SYSTEM_PROMPT,
                                            "user_prompt": state_prompt,
                                            "llm_output": llm_output,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                break
            except Exception as e:
                continue
        if next_action is None:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
            # self.last_thought = ""
        self.total_cnt += 1
        return next_action

    def gen_with_sc_cot(self, battle: AbstractBattle, state_prompt):

        if battle.force_switch:
            constraint_prompt = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            with open("src/prompt/sc_forced_switch_1shot_1", "r") as f:
                template_1 = f.read()
            with open("src/prompt/sc_forced_switch_1shot_2", "r") as f:
                template_2 = f.read()
            with open("src/prompt/sc_forced_switch_1shot_3", "r") as f:
                template_3 = f.read()

        else:
            constraint_prompt = '''Choose the best action. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            with open("src/prompt/sc_act_1shot_1", "r") as f:
                template_1 = f.read()
            with open("src/prompt/sc_act_1shot_2", "r") as f:
                template_2 = f.read()
            with open("src/prompt/sc_act_1shot_3", "r") as f:
                template_3 = f.read()

        # shuffle three templates
        templates = [template_1, template_2, template_3]
        random.shuffle(templates)
        state_prompt_1 = templates[0].replace("[INPUT]", state_prompt) + constraint_prompt
        state_prompt_2 = templates[1].replace("[INPUT]", state_prompt) + constraint_prompt
        state_prompt_3 = templates[2].replace("[INPUT]", state_prompt) + constraint_prompt

        next_action1 = None
        next_action2 = None
        for i in range(2):
            try:
                llm_output1 = self.chatgpt(user_prompt=state_prompt_1,
                                           model=self.backend,
                                           temperature=self.temperature,
                                           max_tokens=self.config.max_output_length,
                                           json_format=True)
                print("llm_output1:", llm_output1)
                next_action1, _ = self.parse(llm_output1, battle)
                break
            except:
                continue

        for i in range(2):
            try:
                llm_output2 = self.chatgpt(user_prompt=state_prompt_2,
                                           model=self.backend,
                                           temperature=self.temperature,
                                           max_tokens=self.config.max_output_length,
                                           json_format=True)
                print("llm_output2:", llm_output2)
                next_action2, _ = self.parse(llm_output2, battle)
                break
            except:
                continue
        if next_action1 and next_action2:
            if next_action1.message == next_action2.message:
                with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                    f.write(json.dumps({"turn": battle.turn,
                                        "system_prompt": SYSTEM_PROMPT,
                                        "user_prompt": state_prompt_1,
                                        "llm_output1": llm_output1,
                                        "llm_output2": llm_output2,
                                        "battle_tag": battle.battle_tag
                                        }) + "\n")
                self.total_cnt += 1
                return next_action1
            else:
                next_action3 = None
                for i in range(2):
                    try:
                        llm_output3 = self.chatgpt(user_prompt=state_prompt_3,
                                                   model=self.backend,
                                                   temperature=self.temperature,
                                                   max_tokens=self.config.max_output_length,
                                                   json_format=True)
                        print("llm_output3:", llm_output3)
                        next_action3, _ = self.parse(llm_output3, battle)
                        break
                    except:
                        continue
                if next_action3:
                    with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "system_prompt": SYSTEM_PROMPT,
                                            "user_prompt": state_prompt_3,
                                            "llm_output1": llm_output1,
                                            "llm_output2": llm_output2,
                                            "llm_output3": llm_output3,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                    self.total_cnt += 1
                    return next_action3
                else:
                    self.total_cnt += 1
                    return next_action1
        next_action = self.choose_max_damage_move(battle)
        self.total_cnt += 1
        return next_action
    def gen_with_tot(self, battle: AbstractBattle, state_prompt):

        if battle.force_switch:
            with open("src/prompt/tot_forced_switch_step1_1shot", "r") as f:
                step1_template = f.read()
            with open("src/prompt/tot_forced_switch_step2_1shot", "r") as f:
                step2_template = f.read()
        else:
            with open("src/prompt/tot_act_step1_1shot", "r") as f:
                step1_template = f.read()
            with open("src/prompt/tot_act_step2_1shot", "r") as f:
                step2_template = f.read()

        # Step one: propose solution
        state_step1_prompt = step1_template.replace("[INPUT]", state_prompt)

        proposal = ""
        next_action = None
        for i in range(2):
            try:
                proposal = self.chatgpt(user_prompt=state_step1_prompt,
                                       model=self.backend,
                                       temperature=self.temperature,
                                       max_tokens=self.config.max_output_length,
                                       json_format=True)
                print("Phase 1 output:", proposal)
                break
            except:
                continue

        # Separate the llm_output without parse.
        if proposal == "":
            self.except_cnt += 1
            return self.choose_max_damage_move(battle)

        # Step 2: Evaluation and Critique
        state_step2_prompt = step2_template.replace("[INPUT]", state_prompt).replace("[PROPOSAL]", proposal)

        for i in range(2):
            try:
                llm_output2 = self.chatgpt(user_prompt=state_step2_prompt,
                                           model=self.backend,
                                           temperature=self.temperature,
                                           max_tokens=self.config.max_output_length,
                                           json_format=True)

                print("Phase 2 output:", llm_output2)
                next_action, evaluation = self.parse(llm_output2, battle)
                with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                    f.write(json.dumps({"turn": battle.turn,
                                        "system_prompt": SYSTEM_PROMPT,
                                        "user_prompt1": state_step1_prompt,
                                        "user_prompt2": state_step2_prompt,
                                        "llm_output1": proposal,
                                        "llm_output2": llm_output2,
                                        "battle_tag": battle.battle_tag
                                        }) + "\n")
                break
            except Exception as e:
                continue

        if next_action is None:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
        self.total_cnt += 1
        return next_action

    def gen_with_reflexion(self, battle: AbstractBattle, state_prompt):
        # check if the last step is active move, active switch, or force switch
        # if the last step is force switch, then skip reflexion because nothing happened.
        # if the last step is active move, then use act template
        # if the last step is active switch, then use switch template

        if self.last_step not in ["", "force_switch"]:
            with open("src/prompt/reflexion", "r") as f:
                reflexion_template = f.read()

            last_outcome = battle.battle_msg_history.split("[sep]")[-2].replace("p2a: ", "").replace("p1a:", "opposing").replace("Player2", "You").replace("Player1", "Opponent")

            reflexion_prompt = (reflexion_template.
                                replace("[INPUT]", self.last_state_prompt).
                                replace("[ACTION]", self.last_action).
                                replace("[OUTCOME]", last_outcome))

            reflexion = ""
            global SYSTEM_PROMPT
            SYSTEM_PROMPT = "Make a reflection on a pokemon battle state, action and its outcome"
            print("=+=+=+=+=+=+=+=+=+=+=+=+=")
            print(reflexion_prompt)
            for i in range(2):
                try:
                    reflexion = self.chatgpt(user_prompt=reflexion_prompt,
                                             model=self.backend,
                                             temperature=self.temperature,
                                             max_tokens=self.config.max_output_length,
                                             json_format=True)
                    json_start = reflexion.find('{')
                    json_end = reflexion.find('}') + 1  # find the first }
                    reflexion = reflexion[json_start:json_end]
                    reflexion = json.loads(reflexion)["reflection"]
                    print("Refection:", reflexion)
                    break
                except Exception as e:
                    continue

            if reflexion == "":
                self.except_cnt += 1
        else:
            reflexion = "" #no reflection if the last step is beginning or force switch

        # Output the result with reflexion
        if battle.force_switch:
            constraint_prompt = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
        else:
            constraint_prompt = '''Choose the best action. Your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''

        if reflexion:
            all_state_prompt = state_prompt + f"Refection on the last step: {reflexion}\n" + constraint_prompt
        else:
            all_state_prompt = state_prompt + constraint_prompt

        next_action = None
        SYSTEM_PROMPT = "You are playing a Pokemon battle and the goal is to win\n" # change back to the original system prompt
        print("=+=+=+=+=+=+=+=+=+=+=+=+=")
        print(all_state_prompt)
        for i in range(2):
            try:
                llm_output = self.chatgpt(user_prompt=all_state_prompt,
                                           model=self.backend,
                                           temperature=self.temperature,
                                           max_tokens=100,
                                           json_format=True)

                print("LLM output:", llm_output)
                next_action, _ = self.parse(llm_output, battle)
                with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                    f.write(json.dumps({"turn": battle.turn,
                                        "system_prompt": SYSTEM_PROMPT,
                                        "user_prompt": all_state_prompt,
                                        "last_user_prompt": self.last_state_prompt,
                                        "reflexion": reflexion,
                                        "llm_output": llm_output,
                                        "battle_tag": battle.battle_tag
                                        }) + "\n")
                break
            except Exception as e:
                continue

        self.last_state_prompt = state_prompt

        if next_action is None:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
        self.total_cnt += 1

        action = next_action.message.split(" ")[1]
        object = next_action.message.split(" ")[2]
        self.last_action = action + ":" + object

        return next_action

    def choose_move(self, battle: AbstractBattle):

        # state_prompt = self.state_translate(battle)
        # return self.choose_random_move(battle)

        if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            next_action = BattleOrder(battle.available_switches[0])
            return next_action

        # state_prompt = self.state_translate(battle)
        state_prompt = self.state_translate(battle) # add lower case

        if self.last_thought:
            state_prompt = state_prompt + f"Your thought from last step: {self.last_thought}\n"
        else:
            state_prompt = state_prompt

        # print("===================")
        # print(state_prompt)

        if self.reason_algo == "io":
            next_action = self.gen_with_io(battle, state_prompt)

        elif self.reason_algo == "cot":
            self.config.max_output_length = 500
            next_action = self.gen_with_cot(battle, state_prompt)

        elif self.reason_algo == "sc":
            self.config.max_output_length = 500
            next_action = self.gen_with_sc_cot(battle, state_prompt)

        elif self.reason_algo == "tot":
            self.config.max_output_length = 500
            next_action = self.gen_with_tot(battle, state_prompt)

        elif self.reason_algo == "reflexion":
            self.config.max_output_length = 500
            next_action = self.gen_with_reflexion(battle, state_prompt)

        else:
            raise ValueError("Unknown reason algo")

        return next_action

    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)