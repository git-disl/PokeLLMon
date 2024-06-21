import pandas as pd
import json

X = pd.read_csv("raw.txt", "\t")

name = list(X.Name.values)
description = list(X.Description.values)
name_new = list(map(lambda x: x.lower().replace(" ", ""), name))

ability_dict = {}

for i in range(len(name)):
    ability_dict[name_new[i]] = {"name": name[i], "effect": description[i]}

print("pause")
with open("ability_effect.json", "w") as f:
    json.dump(ability_dict, f, indent=4)

print("pause")


