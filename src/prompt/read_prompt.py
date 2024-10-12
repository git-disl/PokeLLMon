

def prompt_template(n_shot=0):
    if n_shot == 1:
        with open("src/prompt/vanilla_1shot", "r") as f:
            template = f.read()
            
    elif n_shot == 3:
        with open("src/prompt/vanilla_3shot", "r") as f:
            template = f.read()

    else:
        template = ""
    return template


if __name__ == '__main__':
    template = prompt_template(n_shot=1)
    template = template.replace("[SYSTEM]", "aaa")
    print("pause")
