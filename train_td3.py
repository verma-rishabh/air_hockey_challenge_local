import numpy as np
import torch
import gym
import argparse
import os

import utils
# import OurDDPG
# import DDPG
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder import build_agent


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def custom_rewards(base_env, state, action, next_state, absorbing):
    print(base_env, state, action, next_state, absorbing)

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, debug=True,custom_reward_function=custom_rewards)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        print(_)
        state, done = eval_env.reset(), False
        episode_timesteps=0
        while not done and episode_timesteps<200:
            print("ep",episode_timesteps)
            action = policy.draw_action(np.array(state))
            state, reward, done, _ = eval_env.step(action.reshape(2,3))
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
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=20, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e3, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args, unknown = parser.parse_known_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # env = gym.make(args.env)
    env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, debug=True,custom_reward_function=custom_rewards)


    # Set seeds
    env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.env_info['rl_info'].shape[0]
    action_dim = env.env_info['rl_info'].shape[1] 
    max_action = float(np.max(abs(env.env_info['robot']['joint_pos_limit'])))

    # kwargs = {
    # 	"state_dim": state_dim,
    # 	"action_dim": action_dim,
    # 	"max_action": max_action,
    # 	"discount": args.discount,
    # 	"tau": args.tau,
    # }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        # kwargs["policy_noise"] = args.policy_noise * max_action
        # kwargs["noise_clip"] = args.noise_clip * max_action
        # kwargs["policy_freq"] = args.policy_freq
        policy = build_agent(env.env_info)               ## TO REFORMATE

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim*2)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]
    evaluations=[0]
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            # action = env.action_space.sample()
            action = np.random.uniform(-1, 1, (2, env.env_info['robot']['n_joints'])).reshape(6,) * 3
        else:
            action = (
                policy.draw_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
          
        # Perform action
        next_state, reward, done, _ = env.step(action.reshape(2,3)) 
        env.render()
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0   ###MAX EPISODE STEPS
        done_bool = float(done) if episode_timesteps < 200 else 0
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
        # print(t)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
            

main()