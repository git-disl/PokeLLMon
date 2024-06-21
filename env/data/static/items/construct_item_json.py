import pandas as pd
import json

X = pd.read_csv("raw.txt", "\t")

name = list(X.Name.values)
effect = list(X.Effect.values)
category = list(X.Category.values)

item_dict = {}

for i in range(len(name)):
    new_name = name[i].split(" icon ")[0]

    if str(effect[i]) != "nan":
        item_dict[new_name.lower().replace(" ", "")] = {"name":new_name, "effect":effect[i]}

print("pause")
with open("item_effect.json", "w") as f:
    json.dump(item_dict, f, indent=4)

print("pause")


