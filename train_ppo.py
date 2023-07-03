# reference: ClearRL repository https://github.com/vwxyzjn/cleanrl
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# %%
import argparse
import os
import random
import time
import datetime
import yaml
from distutils.util import strtobool
import sys
sys.path.append('/Users/zahrapadar/Desktop/DL-LAB/project/air_hockey_challenge_local_warmup/')

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
# from air_hockey_agent.agent_builder import build_agent_ppo
from air_hockey_agent.ppo import PPO_Agent
# # %%
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="3dof-hit",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    print("Arguments",  args)
    return args

def reward_c(self, state, action, next_state, absorbing):
    r = 0
    action_penalty = 1e-3
    has_hit = self._check_collision("puck", "robot_1/ee")

    # if not self.has_bounce:
        # self.has_bounce = self._check_collision("puck", "rim_short_sides")
    goal = np.array([0.98, 0])
    puck_pos, puck_vel = self.get_puck(next_state)
    #print("puck_pos",puck_pos)
    #print("puck_vel",puck_vel)
    # width of table minus radius of puck
    effective_width = 0.51 - 0.03165

    # Calculate bounce point by assuming incoming angle = outgoing angle
    w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
        0] - effective_width *
            goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)
    #print("w",w)

    side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
    #print("side_point",side_point)

    vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(side_point - puck_pos[:2])
    vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])
    # If puck is out of bounds
    if absorbing:
        # If puck is in the opponent goal
        if (puck_pos[0] - self.env_info['table']['length'] / 2) > 0 and \
                (np.abs(puck_pos[1]) - self.env_info['table']['goal_width']) < 0:
            r = 200

    else:
        if not has_hit:
            ee_pos = self.get_ee()[0][:2]

            dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)

            vec_ee_puck = (puck_pos[:2] - ee_pos) / dist_ee_puck

            cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

            # Reward if vec_ee_puck and vec_puck_goal have the same direction
            cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
            cos_ang = np.max([cos_ang_goal, cos_ang_side])

            r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
        else:
            r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

            r_goal = 0
            if puck_pos[0] > 0.7:
                sig = 0.1
                r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

            r = 2 * r_hit + 10 * r_goal

    r -= action_penalty * np.linalg.norm(action)
    return r



def cust_rewards(policy, state, done, episode_timesteps):

    reward = 0.0
    ee_pos = policy.get_ee_pose(state)[0]                               
    puck_pos = policy.get_puck_pos(state)
    dist = np.linalg.norm(ee_pos-puck_pos)
    reward += np.exp(-5*dist) * (puck_pos[0]<=1.51)
    # reward+=policy.get_puck_vel(state)[0]
    # # reward -= episode_timesteps*0.01
    # # if policy.get_puck_vel(state)[0]>0.06 and ((dist>0.16)):
    # #     reward+=0
    # reward += np.exp(puck_pos[0]-2.484)*policy.get_puck_vel(state)[0]*(policy.get_puck_vel(state)[0]>0)
    # reward += np.exp(0.536-puck_pos[0])*policy.get_puck_vel(state)[0] *(policy.get_puck_vel(state)[0]<0)
    reward += policy.get_puck_vel(state)[0]
    reward+= done*100

    if abs(policy.get_ee_pose(state)[0][1])>0.519:
        reward -=1 

    if (policy.get_ee_pose(state)[0][0])<0.536:
        reward -=1 

    # what if
    # reward = -distance_to_target - 0.5 * np.linalg.norm(end_effector_velocity)

    return reward

def make_env(env_id, seed, custom_reward_function=None):
    env = AirHockeyChallengeWrapper(env_id, custom_reward_function=custom_reward_function, interpolation_order=3)
    env.seed = seed
    return env

if __name__ == "__main__":
    args = parse_args()
    # args = vars(args)

    # with open('config_ppo.yaml', 'w') as yaml_file:
    #     yaml.dump(args, yaml_file)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
   
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # observation, reward, done, info = env.step(action)

    envs = make_env(args.env_id, args.seed, custom_reward_function=reward_c)
    env_info = envs.env_info

    state_dim = env_info["rl_info"].observation_space.low.shape[0]
    state_dim_ = env_info["rl_info"].observation_space.low.shape

    action_dim = 2 * (env_info["rl_info"].action_space.low.shape[0])
    action_dim_ = (6,)
    state, done = envs.reset(), False #initial_state

    agent = PPO_Agent(envs.env_info, agent_id=1).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (state_dim_)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_dim_).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    state = envs.reset() # second reset, WHY?????
    # next_obs = state
    state = torch.Tensor(state).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    # next_done[global_step] = done

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # linearly decreases to 0
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps): # each batch basically
            global_step += 1 * args.num_envs
            obs[step] = state

            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad(): #because we ar ein rollout phase
                action, logprob, _, value = agent.get_action_and_value(state)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, infos = envs.step(action.cpu().numpy().reshape(2,3))

            writer.add_scalar("charts/custom reward", reward, global_step)

            # custom_reward_function(self.base_env,state, action,next_state, absorbing)?????

            next_done = np.array(done) 
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            state = next_obs

            # CONSTRAINTS ???
            # Get the joint position and velocity from the observation
            q = next_obs[env_info['joint_pos_ids']]
            dq = next_obs[env_info['joint_vel_ids']]
            dist_constr = env_info['constraints'].fun(q,dq)["joint_pos_constr"]
            vel_constr = env_info['constraints'].fun(q,dq)["joint_vel_constr"]
            c_ee = env_info['constraints'].get('ee_constr').fun(q, dq)
            jac_vel = env_info['constraints'].get('joint_vel_constr').jacobian(q, dq)

            state, next_done = torch.Tensor(state).to(device), torch.Tensor(next_done).to(device)
            

            print(f"global_step{global_step}: reward {rewards[step], reward}")

            #infos: 'constraints_value', 'jerk', 'success'

    
        # implementing GAE for PPO:
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(torch.Tensor(next_obs.reshape(1, -1)))
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                # print(t)
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]#TD-error
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam 
            returns = advantages + values
         
        # flatten the batch
        b_obs = obs.reshape((-1,) + state_dim_)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_dim_)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size): #each minibatch
                end = start + args.minibatch_size
                # print(start, end)
                mb_inds = b_inds[start:end]

                # forward pass on minibatch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl: #an early stopping 0.015 is a good threshold
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    # current_date = datetime.now()
    # formatted_date = current_date.strftime("%m-%d-%Y")
    agent.save(f"./models/agent")
    envs.stop()
    writer.close()
# %%
