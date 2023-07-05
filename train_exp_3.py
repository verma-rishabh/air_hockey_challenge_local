import numpy as np
import torch
import os
import utils
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_ddpg_exp3_hit import build_agent
from tensorboard_evaluation import *
from omegaconf import OmegaConf
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian
import copy

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
        # self.action_shape = self.env_info["rl_info"].action_space.shape[0]
        self.action_shape = 4
        self.observation_shape = self.env_info["rl_info"].observation_space.shape[0]
        # policy
        self.policy = build_agent(self.env_info)
        # action_space.high
        pos_max = self.env_info['robot']['joint_pos_limit'][1]
        vel_max = self.env_info['robot']['joint_vel_limit'][1] 
        max_ = np.stack([pos_max,vel_max])
        # self.max_action  = max_.reshape(14,)
        self.max_action  = np.array([1.5,0.5,3.0,3.0])                          # from replay buffer
        # make dirs 
        self.make_dir()
        tensorboard_dir=self.conf.agent.dump_dir + "/tensorboard/"
        self.tensorboard = Evaluation(tensorboard_dir, "train", ["critic_loss","actor_loss","total_reward"])
        # load model if defined
        if self.conf.agent.load_model!= "":
            policy_file = self.conf.agent.file_name if self.conf.agent.load_model == "default" else self.conf.agent .load_model
            # self.policy.load(self.conf.agent.dump_dir + f"/models/{policy_file}")

        ################CAUTION######################################    
        # self.policy.load(self.conf.agent.dump_dir + f"/models/offline")
        ##################################################################
        self.replay_buffer = utils.ReplayBuffer(self.observation_shape, self.action_shape)
        self.replay_buffer.load("/run/media/luke/Data/uni/SS2023/DL Lab/Project/qualifying/DDPG_exp2/replay/data_4actions.npz")

    def make_dir(self):
        if not os.path.exists(self.conf.agent.dump_dir+"/results"):
            os.makedirs(self.conf.agent.dump_dir+"/results")

        if not os.path.exists(self.conf.agent.dump_dir+"/models"):
            os.makedirs(self.conf.agent.dump_dir+"/models")
    
    def reward_mushroomrl(self, state, action, next_state):
        
        r = 0
        reset = 0
        mod_next_state = next_state                            # changing frame of puck pos (wrt origin)
        mod_next_state[:3]  = mod_next_state[:3] - [1.51,0,0.1]
        absorbing = self.base_env.is_absorbing(mod_next_state)
        puck_pos, puck_vel = self.base_env.get_puck(mod_next_state)                     # extracts from obs therefore robot frame


        ###################################################
        goal = np.array([0.974, 0])
        effective_width = 0.519 - 0.03165

        # Calculate bounce point by assuming incoming angle = outgoing angle
        w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
            0] - effective_width *
             goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)


        side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
        #print("side_point",side_point)

        vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(side_point - puck_pos[:2])
        vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])
        has_hit = self.base_env._check_collision("puck", "robot_1/ee")

        
        ###################################################
        
        

        # If puck is out of bounds
        if absorbing:
            # If puck is in the opponent goal
            if (puck_pos[0] - self.env_info['table']['length'] / 2) > 0 and \
                    (np.abs(puck_pos[1]) - self.env_info['table']['goal_width']) < 0:
                    print("puck_pos",puck_pos,"absorbing",absorbing)
                    r = 200

        else:
            if not has_hit:
                ee_pos = self.base_env.get_ee()[0]

                dist_ee_puck = np.linalg.norm(puck_pos - ee_pos)

                vec_ee_puck = (puck_pos[:2] - ee_pos[:2]) / dist_ee_puck

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
        # r-= np.exp(-8 * np.abs(puck_pos[2] - 0.1645))                      #'ee_desired_height': 0.1645
        r -= 1e-3 * np.linalg.norm(action)
        des_z = self.env_info['robot']['ee_desired_height']
        tolerance = 0.02
        if (self.policy.get_ee_pose(next_state)[0][2]-0.1)<des_z-tolerance*10 or (self.policy.get_ee_pose(next_state)[0][2]-0.1)>des_z+tolerance*10:
            r -=10
            reset = 1
        # print(r)
        return r,reset
    # def _loss(self,next_state,action,reward):
    #     desired_action = np.zeros((2,7))
    #     des_z = self.env_info['robot']['ee_desired_height']
    #     ee_pos = self.policy.get_ee_pose(next_state)[0] 
    #     ee_pos[2] = des_z
    #     # angles 
    #     success,desired_angles = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,ee_pos)
    #     if success:                                         # if the confg. is possible
    #         desired_action[0,:] = desired_angles  
    #         loss = - np.square(np.subtract(action, desired_action)).reshape(-1,)/self.max_action    #because its a reward and hence should be -ve                
    #     else:
    #         loss = desired_action.reshape(-1,)
    #         loss[:] = -1                                   # have to think about this
    #     loss+=reward

    #     # loss[6] = reward
    #     return loss
    

    def cust_rewards(self,next_state,action,done):
        reward = 0.0
        reset = 0
        ee_pos = self.policy.get_ee_pose(next_state)[0]                               
        puck_pos = self.policy.get_puck_pos(next_state)
        dist = np.linalg.norm(ee_pos-puck_pos)
        reward += np.exp(-5*dist) * (puck_pos[0]<=1.51)
        # reward+=policy.get_puck_vel(state)[0]
        # # reward -= episode_timesteps*0.01
        # # if policy.get_puck_vel(state)[0]>0.06 and ((dist>0.16)):
        # #     reward+=0
        # reward += np.exp(puck_pos[0]-2.484)*policy.get_puck_vel(state)[0]*(policy.get_puck_vel(state)[0]>0)
        # reward += np.exp(0.536-puck_pos[0])*policy.get_puck_vel(state)[0] *(policy.get_puck_vel(state)[0]<0)
        des_z = self.env_info['robot']['ee_desired_height']
        reward +=self.policy.get_puck_vel(next_state)[0]
        reward+=done*100

        tolerance = 0.02

        if abs(self.policy.get_ee_pose(next_state)[0][1])>0.519:         # should replace with env variables some day
            reward -=1 
        if (self.policy.get_ee_pose(next_state)[0][0])<0.536:
            reward -=1 
        if (self.policy.get_ee_pose(next_state)[0][2]-0.1)<des_z-tolerance*10 or (self.policy.get_ee_pose(next_state)[0][2]-0.1)>des_z+tolerance:
            reward -=1
            # reset = 1
        reward -= 1e-3 * np.linalg.norm(action)
        # print (reward)


        return reward,reset




    def eval_policy(self,eval_episodes=10):
        # eval_env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, debug=False)
        # eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            # print(_)
            state, done = self.reset(), False
            episode_timesteps=0
            while not done and episode_timesteps<100:
                # print("ep",episode_timesteps)
                action = self.policy.draw_action(np.array(state))
                next_state, reward, done, _ = self._step(state,action)
                # done_bool = float(_["success"]) 
                # reward = cust_rewards(policy,state,done_bool,episode_timesteps)
                print(reward)    # def _loss(self,next_state,action,reward):
    #     desired_action = np.zeros((2,7))
    #     des_z = self.env_info['robot']['ee_desired_height']
    #     ee_pos = self.policy.get_ee_pose(next_state)[0] 
    #     ee_pos[2] = des_z
    #     # angles 
    #     success,desired_angles = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,ee_pos)
    #     if success:                                         # if the confg. is possible
    #         desired_action[0,:] = desired_angles  
    #         loss = - np.square(np.subtract(action, desired_action)).reshape(-1,)/self.max_action    #because its a reward and hence should be -ve                
    #     else:
    #         loss = desired_action.reshape(-1,)
    #         loss[:] = -1                                   # have to think about this
    #     loss+=reward

    #     # loss[6] = reward
    #     return loss
                self.render()
                avg_reward += reward
                episode_timesteps+=1
                state = next_state

        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def _step(self,state,action):
        des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645
        _,x = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,des_pos)
        des_v = np.array([action[2],action[3],0.0])
        jac = jacobian(self.policy.robot_model, self.policy.robot_data,self.policy.get_joint_pos(state))
        inv_jac = np.linalg.pinv(jac)
        joint_vel = des_v@inv_jac.T[:3,:]
        # if (_):
        action = np.zeros((2,7))
        action[0,:] = x
        action[1:] = joint_vel
        next_state, reward, done, info = self.step(action)
        next_state_copy = copy.deepcopy(next_state)
        # reward,reset= self.reward_mushroomrl(next_state_copy, action, next_state)
        reward,reset= self.cust_rewards(next_state_copy, action,done)    
            # reward = self._loss(next_state,action,reward)
        # else:
        if (reset):
            # reward -= 1.0
            next_state, done = self.reset(), False
            info = None
        return next_state, reward, done, info

    # def _monte_carlo(self,rewards):
    #     pre_value = 0
    #     for i in reversed(range(rewards.shape[0])):
    #         pre_value = self.conf.agent.discount * pre_value + rewards[i]
    #         rewards[i] = -pre_value/100                                          # loss = -ve of rewards
    #     return rewards

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
                action = np.random.uniform(-self.max_action,self.max_action,(self.action_shape))
            else:
                action = self.policy.draw_action(np.array(state))
            
            # Perform action
            next_state, reward, done, _ = self._step(state,action) 
            # print(next_state[3])
            # self.render()
            # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0   ###MAX EPISODE STEPS
            done_bool = float(done) 
            # reward = cust_rewards(policy,state,done,episode_timesteps)
            # Store data in replay buffer
            self.replay_buffer.add(state, action.reshape(-1,), next_state, reward, done_bool)
            # print(intermediate_t,reward)
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= self.conf.agent.start_timesteps:
                critic_loss,actor_loss=self.policy.train(self.replay_buffer, self.conf.agent.batch_size)

            if done or intermediate_t > 100: 
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
