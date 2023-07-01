import numpy as np
import torch
import os
import utils
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_exp1_hit import build_agent
from tensorboard_evaluation import *
from omegaconf import OmegaConf
from air_hockey_challenge.utils.kinematics import inverse_kinematics


class train(AirHockeyChallengeWrapper):
    def __init__(self, env=None, custom_reward_function=None, interpolation_order=3, **kwargs):
        # Load config file
        self.conf = OmegaConf.load('train.yaml')
        env = self.conf.env
        # base env
        super().__init__(env, custom_reward_function, interpolation_order, **kwargs)
        # seed
        self.seed(self.conf.agent.seed)
        torch.manual_seed(self.conf.agent.seed)
        np.random.seed(self.conf.agent.seed)
        # env variables
        self.action_shape = self.env_info["rl_info"].action_space.shape[0]
        self.observation_shape = self.env_info["rl_info"].observation_space.shape[0]
        # policy
        self.policy = build_agent(self.env_info)
        # action_space.high
        pos_max = self.env_info['robot']['joint_pos_limit'][1]
        vel_max = self.env_info['robot']['joint_vel_limit'][1] 
        max_ = np.stack([pos_max,vel_max])
        self.max_action  = max_.reshape(14,)
        # make dirs 
        self.make_dir()
        tensorboard_dir=self.conf.agent.dump_dir + "/tensorboard/"
        self.tensorboard = Evaluation(tensorboard_dir, "train", ["critic_loss","actor_loss","total_reward"])
        # load model if defined
        if self.conf.agent.load_model!= "":
            policy_file = self.conf.agent.file_name if self.conf.agent.load_model == "default" else self.conf.agent .load_model
            self.policy.load(self.conf.agent.dump_dir + f"/models/{policy_file}")
        
        self.replay_buffer = utils.ReplayBuffer(self.observation_shape, self.action_shape)

    def make_dir(self):
        if not os.path.exists(self.conf.agent.dump_dir+"/results"):
            os.makedirs(self.conf.agent.dump_dir+"/results")

        if not os.path.exists(self.conf.agent.dump_dir+"/models"):
            os.makedirs(self.conf.agent.dump_dir+"/models")

    def _loss(self,next_state,action,reward):
        desired_action = np.zeros((2,7))
        des_z = self.env_info['robot']['ee_desired_height']
        ee_pos = self.policy.get_ee_pose(next_state)[0] 
        ee_pos[2] = des_z
        # angles 
        success,desired_angles = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,ee_pos)
        if success:                                         # if the confg. is possible
            desired_action[0,:] = desired_angles  
            loss = - np.square(np.subtract(action, desired_action)).reshape(-1,)/self.max_action    #because its a reward and hence should be -ve                
        else:
            loss = desired_action.reshape(-1,)
            loss[:] = -1                                   # have to think about this
        loss+=reward

        # loss[6] = reward
        return loss

    def cust_rewards(self,state,done):
        reward = 0.0
        ee_pos = self.policy.get_ee_pose(state)[0]                               
        puck_pos = self.policy.get_puck_pos(state)
        dist = np.linalg.norm(ee_pos-puck_pos)
        reward += np.exp(-5*dist) * (puck_pos[0]<=1.51)
        # reward+=policy.get_puck_vel(state)[0]
        # # reward -= episode_timesteps*0.01
        # # if policy.get_puck_vel(state)[0]>0.06 and ((dist>0.16)):
        # #     reward+=0
        # reward += np.exp(puck_pos[0]-2.484)*policy.get_puck_vel(state)[0]*(policy.get_puck_vel(state)[0]>0)
        # reward += np.exp(0.536-puck_pos[0])*policy.get_puck_vel(state)[0] *(policy.get_puck_vel(state)[0]<0)
        des_z = self.env_info['robot']['ee_desired_height']
        reward +=self.policy.get_puck_vel(state)[0]
        reward+=done*100
        tolerance = 0.02

        if abs(self.policy.get_ee_pose(state)[0][1])>0.519:         # should replace with env variables some day
            reward -=1 
        if (self.policy.get_ee_pose(state)[0][0])<0.536:
            reward -=1 
        if (self.policy.get_ee_pose(state)[0][2]-0.1)<des_z-tolerance or (self.policy.get_ee_pose(state)[0][2]-0.1)>des_z+tolerance:
            reward -=1
        # print (reward)


        return reward




    def eval_policy(self,eval_episodes=10):
        # eval_env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, debug=False)
        # eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            # print(_)
            state, done = self.reset(), False
            episode_timesteps=0
            while not done and episode_timesteps<50:
                # print("ep",episode_timesteps)
                action = self.policy.draw_action(np.array(state))
                next_state, reward, done, _ = self._step(state,action)
                # done_bool = float(_["success"]) 
                # reward = cust_rewards(policy,state,done_bool,episode_timesteps)
                print(reward)
                self.render()
                avg_reward += reward.mean()
                episode_timesteps+=1
                state = next_state

        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def _step(self,state,action):
        next_state, reward, done, info = self.step(action)
        reward = self.cust_rewards(state,done)
        reward = self._loss(next_state,action,reward)
        return next_state, reward, done, info

    def _monte_carlo(self,rewards):
        pre_value = 0
        for i in reversed(range(rewards.shape[0])):
            pre_value = self.conf.agent.discount * pre_value + rewards[i]
            rewards[i] = -pre_value/100                                          # loss = -ve of rewards
        return rewards

    def train_model(self):
        evaluations=[0]
        state, done = self.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        intermediate_t=0
        for t in range(int(self.conf.agent.max_timesteps)):
            critic_loss = np.nan
            actor_loss = np.nan
            episode_timesteps += 1
            intermediate_t+=1
            # Select action randomly or according to policy
            if t < self.conf.agent.start_timesteps:
                # action = env.action_space.sample()
                action = np.random.uniform(-self.max_action,self.max_action,(14,)).reshape(2,7)
            else:
                action = self.policy.draw_action(np.array(state))
            
            # Perform action
            next_state, reward, done, _ = self._step(state,action) 
            # print(next_state[3])
            # env.render()
            # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0   ###MAX EPISODE STEPS
            done_bool = float(done) 
            # reward = cust_rewards(policy,state,done,episode_timesteps)
            # Store data in replay buffer
            self.replay_buffer.add(state, action.reshape(-1,), next_state, reward.reshape(-1,), done)
            # print(intermediate_t,reward)
            state = next_state
            episode_reward += reward.mean()

            # Train agent after collecting sufficient data
            if t >= self.conf.agent.start_timesteps:
                critic_loss,actor_loss=self.policy.train(self.replay_buffer, self.conf.agent.batch_size)

            if done or intermediate_t > 50: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                self.tensorboard.write_episode_data(t, eval_dict={ "critic_loss" : critic_loss,\
                    "actor_loss":actor_loss,\
                        "total_reward":episode_reward})
                state, done = self.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                intermediate_t=0
            # print(t)
            # Evaluate episode
            if (t + 1) % self.conf.agent.eval_freq == 0:
                evaluations.append(self.eval_policy())
                np.save(self.conf.agent.dump_dir +f"/results/{self.conf.agent.file_name}", evaluations)
                if 1: self.policy.save(self.conf.agent.dump_dir + f"/models/{self.conf.agent.file_name}")

x = train()
x.train_model()
