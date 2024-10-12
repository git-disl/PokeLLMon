import asyncio
import argparse
from src.player import RandomPlayer, MaxBasePowerPlayer, HeuristicsPlayer, RLPlayer, LLMACAgent
import pickle as pkl
from distutils.util import strtobool
import numpy as np
import wandb
import os
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from poke_env import AccountConfiguration
import random
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='local_test')
    parser.add_argument("--log_dir", type=str, default="battle_log/llama_7b_lowercase_t0.8_gpt4_imitation_reason_action_epoch10_vs_heuristic_0113")
    parser.add_argument("--oppo", type=str, default="heuristic", choices=["random", "max_damage", "heuristic"])
    parser.add_argument("--policy-learning-rate", type=float, default=1e-5, help="the learning rate of the policy network")
    parser.add_argument("--value-learning-rate", type=float, default=1e-5, help="the learning rate of the value network")
    parser.add_argument("--num-steps", type=int, default=32, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="total timesteps of the experiments")

    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,  help="the lambda for the general advantage estimation")
    parser.add_argument("--policy-num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--value-num-minibatches", type=int, default=8, help="the number of mini-batches")

    parser.add_argument('--gradient-checkpointing-steps', action='store', type=int, default=8, help='The number of steps for gradient checkpointing')
    parser.add_argument('--critic-warm-up-steps', action='store', type=int, default=0, help='The number of time steps to warm up critic')

    parser.add_argument("--update-epochs", type=int, default=1, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")

    # dir
    parser.add_argument('--ckpt_dir', action='store', type=str, default="saved_models")
    parser.add_argument('--init_checkpoint', action='store', type=str, default="saved_models")
    parser.add_argument('--record-path', action='store', type=str, default="exp_gemma_2b", help='The path to save the tensorboard results')

    args = parser.parse_args()

    args.batch_size = int(args.num_steps)
    args.policy_minibatch_size = int(args.batch_size // args.policy_num_minibatches)
    print("policy_minibatch_size:", args.policy_minibatch_size)
    args.value_minibatch_size = int(args.batch_size // args.value_num_minibatches)
    print("value_minibatch_size:", args.value_minibatch_size)

    return args

async def main():

    index = str(random.randint(0, 10000))
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    run_name = f"{time_str}"
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}"

    if args.oppo == "random":
        opponent = RandomPlayer(battle_format="gen8randombattle")
        # name_prefix = "random"
    elif args.oppo == "heuristic":
        opponent = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
        # name_prefix = "heuristic"
    elif args.oppo == "max_damage":
        opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
        # name_prefix = "max_damage"
    else:
        raise ValueError(f"Please choose correct opponent modes")

    opponent._dynamax_disable = True

    wandb.init(
        entity=None,
        project="PokeLLMon",
        dir="wandb",
        name=args.exp_name,
    )

    envs = RLPlayer(battle_format="gen8randombattle",
                    opponent=opponent,
                    start_challenging=True,
                    account_configuration=AccountConfiguration("PokeLLM" + index, "")
                    )

    # initialize Agent
    lora_weights = ""

    # lora_weights = ""
    agent = LLMACAgent(battle_format="gen8randombattle",
                       # model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
                       model_name_or_path="google/gemma-2b-it",
                       lora_weights=lora_weights,
                       w_knowledge=True,
                       log_dir=args.log_dir
                       )

    policy_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, agent.actor.parameters()),
                                     lr=args.policy_learning_rate, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0)

    value_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, agent.critic.parameters()),
                                    lr=args.value_learning_rate, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0)

    # ALGO Logic: Storage setup
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    obs = [None] * args.num_steps
    actions = torch.zeros(args.num_steps).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)
    steps = torch.zeros(args.num_steps).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()
    envs.reset()
    next_obs = agent.battle2prompt(envs.current_battle)
    next_done = False
    num_updates = args.total_timesteps // args.batch_size
    num_critic_warm_up_updates = args.critic_warm_up_steps // args.batch_size

    is_warmup = True
    # is_warmup = False # for testing
    step_counter = 0
    pbar = tqdm(total=args.total_timesteps)
    for update in range(1, num_updates + 1 + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False

        # Annealing the rate if instructed to do so.
        if args.anneal_lr and not is_warmup:
            frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
            policy_optimizer.param_groups[0]["lr"] = frac * args.policy_learning_rate
            value_optimizer.param_groups[0]["lr"] = frac * args.value_learning_rate

        for step in range(0, args.num_steps):
            global_step += 1
            pbar.update(1)
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value([next_obs])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.

            _, reward, next_done, truncated, info = envs.step(action.cpu().numpy()[0]) # here I should modify the choose action function
            # print("info", info)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            if next_done:
                step_counter = 0
                envs.reset()
            else:
                step_counter += 1

            next_obs = agent.battle2prompt(envs.current_battle)
            # print(next_obs)
            steps[step] = step_counter # step # is not equal to turn #

            # for item in info: # How to include these information into info?
            #     if "episode" in item.keys():
            #         print(
            #             f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value([next_obs]).reshape(1, -1) # the input should be list
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                discount = torch.pow(args.gamma, steps[t])
                delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + discount * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

    #     # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        pg_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        old_approx_kl = torch.tensor(0)
        approx_kl = torch.tensor(0)
        total_approx_kl = torch.tensor(0)

        for epoch in range(args.update_epochs):
            if kl_explode:
                break
            # update value
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.value_minibatch_size):
                end = start + args.value_minibatch_size
                mb_inds = b_inds[start:end]
                newvalue = agent.get_value([b_obs[x] for x in mb_inds])

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * args.vf_coef

                value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                value_optimizer.step()

            if is_warmup:
                continue

            policy_optimizer.zero_grad()
            # update policy
            for start in range(0, args.batch_size, args.policy_minibatch_size):
                if policy_update_steps % args.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + args.policy_minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value([b_obs[x] for x in mb_inds], b_actions.long()[mb_inds],
                                                                              is_warmup, return_value=False)

                logratio = newlogprob - b_logprobs[
                    mb_inds]  # here is off-policy, because the batch is 128, might update many times
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / args.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss
                loss /= args.gradient_checkpointing_steps

                loss.backward()

                if policy_update_steps % args.gradient_checkpointing_steps == 0:
                    if args.target_kl is not None:
                        if total_approx_kl > args.target_kl:
                            policy_optimizer.zero_grad()
                            kl_explode = True
                            policy_update_steps -= args.gradient_checkpointing_steps
                            break

                    nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                    policy_optimizer.step()
                    policy_optimizer.zero_grad()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        wandb.log({"charts/policy_learning_rate": policy_optimizer.param_groups[0]["lr"], "global_step": global_step})
        wandb.log({"charts/value_learning_rate": value_optimizer.param_groups[0]["lr"], "global_step": global_step})
        wandb.log({"losses/value_loss": v_loss.item(), "global_step": global_step})
        wandb.log({"losses/policy_loss": pg_loss.item(), "global_step": global_step})
        wandb.log({"losses/entropy": entropy_loss.item(), "global_step": global_step})
        wandb.log({"losses/old_approx_kl": old_approx_kl.item(), "global_step": global_step})
        wandb.log({"losses/approx_kl": approx_kl.item(), "global_step": global_step})
        wandb.log({"losses/total_approx_kl": total_approx_kl.item(), "global_step": global_step})
        wandb.log({"losses/policy_update_times": policy_update_steps // args.gradient_checkpointing_steps,
                   "global_step": global_step})
        wandb.log({"losses/clipfrac": num_clipfracs, "global_step": global_step})
        wandb.log({"losses/explained_variance": explained_var, "global_step": global_step})

        current_time = time.time()
        print("SPS:", global_step, (current_time - start_time))
        wandb.log({"charts/SPS": global_step / (current_time - start_time), "global_step": global_step})

        # if global_step // 3000 != pre_global_step // 3000:
        #     agent.save(global_step // 3000, f"{args.record_path}/{run_name}/{args.ckpt_dir}")
        # pre_global_step = global_step

    # agent.save(global_step // 3000 + 1, f"{args.record_path}/{run_name}/{args.ckpt_dir}")

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())