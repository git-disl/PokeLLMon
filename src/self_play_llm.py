import asyncio
from tqdm import tqdm
import torch
import numpy as np
import os
import hydra
import logging
import random
import pickle as pkl
import time
import json
import transformers
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from src.utils.llm_utils import get_local_dir, get_local_run_dir, disable_dropout
from src.client.account_configuration import AccountConfiguration
from src.client.server_configuration import ServerConfiguration

from src.player import LLMPlayer, HeuristicsPlayer

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

@hydra.main(version_base=None, config_path="config", config_name="config")
def entry_point(config: DictConfig):
    logging.getLogger().setLevel(logging.WARNING)
    # Running the async main function
    asyncio.get_event_loop().run_until_complete(main(config))
#
async def main(config: DictConfig):
    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    print(OmegaConf.to_yaml(config))

    # load llm as policy model
    model_kwargs = {'device_map': 'balanced'}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, output_hidden_states=True, cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(
            f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])

    # Initialize Players
    index = str(random.randint(0, 100000))
    llm_player1 = LLMPlayer(model=policy,
                           config=config,
                           save_replay_dir="", # no replay
                           account_configuration=AccountConfiguration("llm_player1_" + index, "")) # 加index是为了防止重连时, 触发 choose_move
    llm_player2 = LLMPlayer(model=policy,
                            config=config,
                            save_replay_dir="", # no replay
                            account_configuration=AccountConfiguration("llm_player2_" + index, ""))

    # dynamax is disabled for local battles.
    llm_player1._dynamax_disable = True
    llm_player2._dynamax_disable = True

    time.sleep(10)
    # Now, let's evaluate our player
    for i in tqdm(range(2750)):

        x = np.random.randint(0, 100)
        if x > 50:
            await llm_player1.battle_against(llm_player2, n_battles=1)
        else:
            await llm_player2.battle_against(llm_player1, n_battles=1)

        # 每打完一场，检查双方的胜负并记录
        for battle_id, battle in llm_player1.battles.items():
            with open(f"battle_data/self_play_llm/label_{index}.json", "a") as f:
                if battle.won:
                    result_log = {"battle_id": battle_id, "winner": llm_player1.username, "loser": llm_player2.username}
                elif battle.lost:
                    result_log = {"battle_id": battle_id, "winner": llm_player2.username, "loser": llm_player1.username}
                else:
                    continue
                f.write(json.dumps(result_log) + "\n")
        # 清空已记录的对战数据，为下一轮对战准备
        llm_player1.battles.clear()
        llm_player2.battles.clear()


if __name__ == "__main__":
    entry_point()
    # asyncio.get_event_loop().run_until_complete(main())
