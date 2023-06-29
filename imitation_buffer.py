# %%
import numpy as np
import torch
import gym
import argparse
import os

import utils
# import OurDDPG
# import DDPG
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
# from air_hockey_agent.agent_builder_ddpg_hit import build_agent
# from air_hockey_challenge.environments.planar.hit import AirHockeyHit
from tensorboard_evaluation import *
from baseline.baseline_agent.baseline_agent import build_agent

# %%
def cust_rewards(policy,state,done,episode_timesteps):
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
    des_z = 0.1645
    reward +=policy.get_puck_vel(state)[0]
    reward+=done*100
    # print(done,reward)
    tolerance = 0.02
    if abs(policy.get_ee_pose(state)[0][1])>0.519:
        reward -=1 
    if (policy.get_ee_pose(state)[0][0])<0.536:
        reward -=1 
    if (policy.get_ee_pose(state)[0][2]-0.1)<des_z-tolerance or (policy.get_ee_pose(state)[0][2]-0.1)>des_z+tolerance:
        reward -=1

    # print(reward)
    return reward

# %%
env = AirHockeyChallengeWrapper(env="7dof-hit",\
    interpolation_order=3, debug=False)


state_dim = env.env_info['rl_info'].shape[0]
action_dim = env.env_info['rl_info'].shape[1]

pos_max = env.env_info['robot']['joint_pos_limit'][1]
vel_max = env.env_info['robot']['joint_vel_limit'][1] 
max_ = np.stack([pos_max,vel_max])
max_action  =   max_.reshape(14,)

# kwargs = {
# 	"state_dim": state_dim,
# 	"action_dim": action_dim,
# 	"max_action": max_action,
# 	"discount": args.discount,
# 	"tau": args.tau,
# }

# Initialize policy

policy = build_agent(env.env_info)               ## TO REFORMATE


replay_buffer = utils.ReplayBuffer(state_dim, 14)

# Evaluate untrained policy
# evaluations = [eval_policy(policy, env, args.seed)]
evaluations=[0]
state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0
intermediate_t=0
for t in range(int(1e6)):
    critic_loss = np.nan
    actor_loss = np.nan
    episode_timesteps += 1
    intermediate_t+=1
    # Select action randomly or according to policy
    
    action = policy.draw_action(np.array(state))
        
    # Perform action
    next_state, reward, done, _ = env.step(action) 
    # print(next_state[3])
    # env.render()
    # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0   ###MAX EPISODE STEPS
    done_bool = float(done) 
    reward = cust_rewards(policy,state,done,episode_timesteps)
    # Store data in replay buffer
    replay_buffer.add(state, action.reshape(-1,), next_state, reward, done)
    # print(intermediate_t,reward)
    state = next_state
    episode_reward += reward

    # Train agent 0ng sufficient data
    # if t >= 0:
    #     critic_loss,actor_loss=policy.train(replay_buffer, args.batch_size)

    if done or intermediate_t > 200: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
    # Reset environment

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 
        intermediate_t=0
# print(t)
# Evaluate episode
    if (t + 1) % 1e3 == 0:
        replay_buffer.save("replay_buffer/data")


