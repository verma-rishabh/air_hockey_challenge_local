import numpy as np
import torch
import gym
import argparse
import os

import utils
# import OurDDPG
# import DDPG
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_ddpg_hit import build_agent
# from air_hockey_challenge.environments.planar.hit import AirHockeyHit
from tensorboard_evaluation import *
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


# def custom_rewards(base_env, state, action, next_state, absorbing):
#     reward = 0
#     reward +=   (next_state[3] - state[3])*100 if next_state[3]>state[3] else 0 
#     return reward

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
    tolerance = 0.02
    if abs(policy.get_ee_pose(state)[0][1])>0.519:
        reward -=1 
    if (policy.get_ee_pose(state)[0][0])<0.536:
        reward -=1 
    if (policy.get_ee_pose(state)[0][2]-0.1)<des_z-tolerance or (policy.get_ee_pose(state)[0][2]-0.1)>des_z+tolerance:
        reward -=1
    # print (reward)


    return reward

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


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    # eval_env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, debug=False)
    # eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        # print(_)
        state, done = eval_env.reset(), False
        episode_timesteps=0
        while not done and episode_timesteps<200:
            # print("ep",episode_timesteps)
            action = policy.draw_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            # done_bool = float(_["success"]) 
            # reward = cust_rewards(policy,state,done_bool,episode_timesteps)
            print(reward)
            eval_env.render()
            avg_reward += reward
            episode_timesteps+=1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG-v0")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="air-hockey")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=200, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5000, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e7, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256,    type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.05)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=10, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name
    
    args, unknown = parser.parse_known_args()
    main_dir = "/run/media/luke/Data/uni/SS2023/DL Lab/Project/qualifying/DDPG"
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("-----------------------------render----------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(main_dir+"/results"):
        os.makedirs(main_dir+"/results")

    if args.save_model and not os.path.exists(main_dir+"/models"):
        os.makedirs(main_dir+"/models")
    tensorboard_dir=main_dir + "/tensorboard/"
    tensorboard = Evaluation(tensorboard_dir, "train", ["critic_loss","actor_loss","total_reward"])

    # env = gym.make(args.env)
    env = AirHockeyChallengeWrapper(env="7dof-hit",\
         interpolation_order=3, debug=False)
    # env = AirHockeyHit(moving_init=False,horizon=180)
    # Set seeds
    env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(main_dir + f"/models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, 14)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, env, args.seed)]
    evaluations=[0]
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    intermediate_t=0
    for t in range(int(args.max_timesteps)):
        critic_loss = np.nan
        actor_loss = np.nan
        episode_timesteps += 1
        intermediate_t+=1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            # action = env.action_space.sample()
            action = np.random.uniform(-max_action,max_action,(14,)).reshape(2,7)
        else:
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

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            critic_loss,actor_loss=policy.train(replay_buffer, args.batch_size)

        if done or intermediate_t > 50: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            tensorboard.write_episode_data(t, eval_dict={ "critic_loss" : critic_loss,\
                "actor_loss":actor_loss,\
                    "total_reward":episode_reward})
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            intermediate_t=0
        # print(t)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, env, args.seed))
            np.save(main_dir +f"/results/{file_name}", evaluations)
            if 1: policy.save(main_dir + f"/models/{file_name}")
            

main()