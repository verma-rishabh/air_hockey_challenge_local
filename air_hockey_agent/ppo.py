import os
import random
import time
from distutils.util import strtobool
import sys
sys.path.append('/Users/zahrapadar/Desktop/DL-LAB/project/air_hockey_challenge_local_warmup/')

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from air_hockey_challenge.framework.agent_base import AgentBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

class PPO_Agent(AgentBase, nn.Module):

    def __init__(self, env, agent_id):
        super(PPO_Agent, self).__init__(env.env_info, agent_id)
        nn.Module.__init__(self)

        self.env = env
        self.state_dim = env.env_info["rl_info"].observation_space.low.shape[0]
        self.action_dim = 2 * env.env_info["rl_info"].action_space.low.shape[0]
        self.action_dim_ = 2 * env.env_info["rl_info"].action_space.low.shape[0]

        
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(self.state_dim), 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(self.state_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, self.action_dim), std=0.01),
        )

        # standard dev. of the components of the action
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        # the parameters declared with nn.Parameter are automatically registered as trainable parameters 
        # of the model. The optimizer is responsible for updating these parameters based on the 
        # computed gradients
    

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

    def get_value(self, x):
        return self.critic(x)
    
    def draw_action(self, x):
        # x being the observation
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(self.action_mean_)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample().reshape(2,3)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x) # unnormalized action probabilities
        action_logstd = self.actor_logstd #.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std) #in discrete we use Categorical ~ softmax
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor_mean.state_dict(), filename + "_actor_mean")
        torch.save(self.actor_logstd, filename + "_actor_logstd")

    def load(self, filename):
        self.critic.load_state_dict(filename + "_critic")
        self.actor_mean.load_state_dict(filename + "_actor_mean")
        self.actor_logstd.load_state_dic(filename + "_actor_logstd")

    def reset(self):
        self.env.reset()
