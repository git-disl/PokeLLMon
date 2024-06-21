import json
import re
import json

move_name_list = []
move_effect_list = []
with open("gen8_raw.txt", "r") as f:
    idx = 0
    for i in range(2184):
        data = f.readline()
        if idx%3 == 0:
            move_name = data.split("	")[0]
            move_name_list.append(move_name)
        elif idx%3 == 1:
            effect = data[:-1]
            move_effect_list.append(effect)

        idx += 1

move2effect = dict(zip(move_name_list, move_effect_list))

with open("gen8moves.json", "r") as f:
    gen8moves = json.load(f)

move2effect_new = dict()
for move, info in gen8moves.items():
    try:
        effect = move2effect[info['name']]
        move2effect_new[move] = effect
    except:
        continue


with open("gen8moves_effect.json", "w") as f:
    json.dump(move2effect_new, f, indent=4)
