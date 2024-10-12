"""
Generate pairs of types to ask the LLM to predict whether it has advantage or disadvantges
"""
import argparse
from src.data.gen_data import GenData
import json
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import classification_report

TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="battle_log/type_chart_predict_task1")
args = parser.parse_args()
from openai import OpenAI

completion_tokens = 0
prompt_tokens = 0


def generate_task1_pairs():

    gen = GenData.from_format("gen8randombattle")
    type_chat = gen.type_chart
    pair_list = []
    label_list = []

    for pokemon_type, chart_dict in type_chat.items():
        for move_type, damage_multiplier in chart_dict.items():
            pair_list.append([move_type, pokemon_type])
            label_list.append(damage_multiplier)

    return pair_list, label_list

def main():

    # backend = "gpt-3.5-turbo"
    # backend = "gpt-4-1106-preview"
    # backend = "gpt-4o-2024-08-06"
    backend = "gpt-4o-mini-2024-07-18"

    correct_cnt = 0
    wrong_cnt = 0
    pair_list, label_list = generate_task1_pairs()
    llm_output_list = []
    answer_list = []
    for idx in tqdm(range(len(pair_list))):
        pair = pair_list[idx]
        label = label_list[idx]
        if label == 1:
            answer = "B"
        elif label == 2:
            answer = "A"
        elif label == 0.5:
            answer = "C"
        elif label == 0:
            answer = "D"
        else:
            raise Exception("Unknown label")
        answer_list.append(answer)

        # prompt = f'"What is the outcome when a {pair[1]} type Pokémon is hit by a {pair[0]} type move?" A. Super Effective (Double Damage); B. Standard (Normal Damage); C. Not Effective (Half Damage); D. Zero Effect (Immunity); E. I Do Not Know' \
        #           'The output should only be a json: {"answer":"<your answer>"}. \nOutput:'

        prompt = f'In Pokémon battles, a {pair[0].lower()} attack is ____ against a {pair[1].lower()} Pokémon?\nA. Super Effective (2x Damage)\nB. Standard (1x Damage)\nC. Not Effective (0.5x Damage)\nD. Zero Effect (0x Damage)\nYour output:'

        try:
            client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                model=backend,
                messages=[
                    {"role": "system", "content": 'You are an assistant to answer the following multi-choice question. The output should be a JSON in the format {"answer":"<your answer>"}. No additional text is allowed.'},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=False,
                # seed=seed,
                max_tokens=100,
            )
            llm_output = response.choices[0].message.content

            llm_output = llm_output.split("Output:")[-1]
            print(f"Move: {pair[0]}; Pokemon: {pair[1]}; Label: {answer}")
            print("LLM output:", llm_output)

            json_start = llm_output.find('{')
            json_end = llm_output.find('}') + 1  # find the first }
            json_content = llm_output[json_start:json_end]
            llm_answer = json.loads(json_content)["answer"]
            if llm_answer.startswith("A"):
                llm_output_list.append("A")
            elif llm_answer.startswith("B"):
                llm_output_list.append("B")
            elif llm_answer.startswith("C"):
                llm_output_list.append("C")
            elif llm_answer.startswith("D"):
                llm_output_list.append("D")
            else:
                llm_output_list.append("E")
        except Exception as e:
            print(e)
            llm_output_list.append("B")

    report = classification_report(answer_list, llm_output_list, digits=4)
    print(report)


    #
    # with open(f"{args.log_dir}/{backend}_t0_uppercase.jsonl", "a") as f:
    #     for i in range(len(llm_output_list)):
    #         llm_output = llm_output_list[i]
    #         pair = pair_list[i]
    #         label = label_list[i]
    #         json_start = llm_output.find('{')
    #         json_end = llm_output.find('}') + 1
    #         json_content = llm_output[json_start:json_end]
    #         X = json.loads(json_content)
    #         if label == 1:
    #             answer = "B"
    #         elif label == 2:
    #             answer = "A"
    #         elif label == 0.5:
    #             answer = "C"
    #         else:
    #             answer = "D"
    #
    #         X.update({
    #             "move": pair[0],
    #             "pokemon": pair[1],
    #             "label": answer
    #         })
    #         f.write(json.dumps(X) + "\n")


if __name__ == '__main__':
    main()