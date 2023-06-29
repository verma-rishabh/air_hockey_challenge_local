import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from air_hockey_challenge.framework.agent_base import AgentBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		# self.l3 = nn.Linear(64, 64)
		# self.l4 = nn.Linear(64, 32)
		self.l5 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		# a = F.relu(self.l3(a))
		# a = F.relu(self.l4(a))
		return self.max_action * torch.tanh(self.l5(a))


# class Critic(nn.Module):
# 	def __init__(self, state_dim, action_dim):
# 		super(Critic, self).__init__()

# 		self.l1 = nn.Linear(state_dim + action_dim, 256)
# 		self.l2 = nn.Linear(256, 256)
# 		# self.l3 = nn.Linear(64, 32)
# 		self.l4 = nn.Linear(256, 1)


# 	def forward(self, state, action):
# 		q = F.relu(self.l1(torch.cat([state, action], 1)))
# 		q = F.relu(self.l2(q))
# 		# q = F.relu(self.l3(q))
# 		return self.l4(q)


class exp1_agent(AgentBase):
	def __init__(self, env_info,agent_id, discount=0.99, tau=0.005):
		super().__init__(env_info, agent_id)
		state_dim = self.env_info["rl_info"].observation_space.shape[0]
		action_dim = self.env_info["rl_info"].action_space.shape[0]
		pos_max = env_info['robot']['joint_pos_limit'][1]
		vel_max = env_info['robot']['joint_vel_limit'][1] 
		max_ = np.stack([pos_max,vel_max],dtype=np.float32)
		max_action  = max_.reshape(action_dim,)
		max_action = torch.from_numpy(max_action).to(device)
		self.max_action = max_action

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.actor_loss = torch.nn.MSELoss(reduction='none')
		# self.critic = Critic(state_dim, action_dim).to(device)
		# self.critic_target = copy.deepcopy(self.critic)
		# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau


	def draw_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).clamp(-self.max_action, self.max_action).cpu().data.numpy()\
			.flatten().reshape(2,7)
		# return self.actor(state).cpu().data.numpy().flatten()


	def train(self,state,loss,reward):
		# Sample replay buffer 
		

		# # Compute the target Q value
		# target_Q = self.critic_target(next_state, self.actor_target(next_state).clamp(-self.max_action, self.max_action))
		# target_Q = reward + (not_done * self.discount * target_Q).detach()

		# # Get current Q estimate
		# current_Q = self.critic(state, action)

		# # Compute critic loss
		# critic_loss = F.mse_loss(current_Q, target_Q)
		# _critic_loss = critic_loss.item()
		# # Optimize the critic
		# self.critic_optimizer.zero_grad()
		# critic_loss.backward()
		# self.critic_optimizer.step()

		# Compute actor loss
		_loss = self.actor_loss(self.actor(state), loss).to(device)/self.max_action-reward				# have to check if works as desired
		# _loss[:,6] = loss[:,6]													# last joint pos
		# actor_loss = -self.critic(state, self.actor(state)).mean()
		_loss = torch.mean(_loss,axis=0)
		_actor_loss = _loss.cpu().data.numpy().flatten().sum()
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		_loss.backward(torch.ones((14)).to(device)*1.0)				# have to check if works as desired
		self.actor_optimizer.step()

		# Update the frozen target models
		# for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
		# 	target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return _actor_loss

	def save(self, filename):
		# torch.save(self.critic.state_dict(), filename + "_critic")
		# torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		# self.critic.load_state_dict(torch.load(filename + "_critic"))
		# self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		# self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
	
	def reset(self):
		pass