# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from .model import DQN


class Agent():

	def __init__(self, args, env):
		try:
			self.action_space = int(env.action_space.n)
		except:
			self.action_space = int(env) # action space 
		self.atoms = args.atoms 
		self.Vmin = args.V_min 
		self.Vmax = args.V_max 
		self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  
		self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1) 
		self.batch_size = args.batch_size 
		self.n = args.multi_step
		self.discount = args.discount
		self.norm_clip = args.norm_clip

		self.net = DQN(args, self.action_space).to(device=args.device)
		self.target_net = DQN(args, self.action_space).to(device=args.device)
		self.net.train()
		self.update_target_net()
		self.target_net.train()    
		
		for param in self.target_net.parameters():
			param.requires_grad = False 

		self.optimiser = optim.Adam(self.net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

	def reset_noise(self):
		self.net.reset_noise()

	def act(self, state): # action 
		with torch.no_grad():
			return (self.net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

	def learn(self, mem):
		idxs, states, actions, returns, next_states, done_masks , weights = mem.sample(self.batch_size) # replay buffer
		log_p_s = self.net(states, log=True)  # log probability 
		log_p_sa = log_p_s[range(self.batch_size), actions]  # log probability 

		with torch.no_grad():
			p_s = self.net(next_states) # n-step 
			d_s = self.support.expand_as(p_s) * p_s  # distribution
			argmax_indices_ns = d_s.sum(2).argmax(1)  # argmax index
			self.target_net.reset_noise()  
			p_s = self.target_net(next_states)  # target Q (probability) head 
			p_sa = p_s[range(self.batch_size), argmax_indices_ns]  # target Q probability (policy)

			# Distribution RL 
			V = returns.unsqueeze(1) + done_masks * (self.discount ** self.n) * self.support.unsqueeze(0) # Bellman equation
			V = V.clamp(min=self.Vmin, max=self.Vmax)  # Distribution clamping

			#  distribution  bin L2 projection histogram
			b = (V - self.Vmin) / self.delta_z  
			l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64) # lower, upper bound

			l[(u > 0) * (l == u)] -= 1 
			u[(l < (self.atoms - 1)) * (l == u)] += 1
			m = states.new_zeros(self.batch_size, self.atoms)
			offset = torch.linspace(0, ( (self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
			m.view(-1).index_add_(0, (l + offset).view(-1), (p_sa * (u.float() - b)).view(-1))  
			m.view(-1).index_add_(0, (u + offset).view(-1), (p_sa * (b - l.float())).view(-1))  

		loss = -torch.sum(m * log_p_sa, 1)  # m : target, log_p_sa : input distribution -> cross entropy
		self.net.zero_grad()
		(weights * loss).mean().backward()  # importance weight
		clip_grad_norm_(self.net.parameters(), self.norm_clip)  # Gradient clipping.(gradient bad policy.)
		self.optimiser.step()

		mem.update_priorities(idxs, loss.detach().cpu().numpy())  # PER TD-error update| sampling importance update

	def update_target_net(self): # target network update
		self.target_net.load_state_dict(self.net.state_dict())

	def save(self, path, name='model.pth'): # save model
		torch.save(self.net.state_dict(), os.path.join(path, name))

	def load(self, path, name='model.pth'): # load model for test
		self.net.load_state_dict(torch.load(os.path.join(path,name)))

	def train(self): 
		self.net.train()

	def eval(self):
		self.net.eval()
