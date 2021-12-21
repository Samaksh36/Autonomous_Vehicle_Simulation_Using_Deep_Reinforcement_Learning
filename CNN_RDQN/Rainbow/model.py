# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

class NoisyLinear(nn.Module): # Noisy networks
	"""
	Noisy networks 가 구현되어 있는 모듈이다.
	"""
	def __init__(self, in_features, out_features, std_init=0.5): # std  0.5  exploration
		super(NoisyLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.std_init = std_init
		self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
		self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
		self.bias_mu = nn.Parameter(torch.empty(out_features))
		self.bias_sigma = nn.Parameter(torch.empty(out_features))
		self.register_buffer('weight_epsilon', torch.empty(out_features, in_features)) # register_buffer parameter-> update
		self.register_buffer('bias_epsilon', torch.empty(out_features)) # noise VAE
		self.reset_parameters()
		self.reset_noise()
		

	def reset_parameters(self):
		mu_range = 1 / math.sqrt(self.in_features)  # xavier
		self.weight_mu.data.uniform_(-mu_range, mu_range)
		self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
		self.bias_mu.data.uniform_(-mu_range, mu_range)
		self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

	def _scale_noise(self, size): 
		x = torch.randn(size, device=self.weight_mu.device) # normal distribution  reset
		return x.sign().mul_(x.abs().sqrt_()) # root scale

	def reset_noise(self):
		epsilon_in = self._scale_noise(self.in_features) # root normal distribution noise
		epsilon_out = self._scale_noise(self.out_features) 
		self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in)) # outer product  weight epsilon
		self.bias_epsilon.copy_(epsilon_out)

	def forward(self, input):
		if self.training: # .train() network noisy net 
			return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
		else: # 아닌 것은 그냥
			return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module): # RainbowDQN
	"""
		Main DQN 모듈이다.
		Dueling DQN, Noisy networks 가 주로 구현되어있는 모듈이다.
	"""
	def __init__(self, args, action_space):
		super(DQN, self).__init__()
		self.atoms = args.atoms
		self.action_space = action_space

		self.convs = nn.Sequential(nn.Conv2d(args.history_length, 16, 3, stride=1, padding=1), nn.ReLU(), # 16X160X64
									nn.MaxPool2d(2,2), # 16X80X32
									nn.Conv2d(16,32,3,1, padding=1), nn.ReLU(), # 32X80X32
									nn.MaxPool2d(2,2), # 32X40X16
									nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), # 64X40X16
									nn.MaxPool2d(2,2), # 64X20X8
									nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),  # 128X20X8
									nn.MaxPool2d(2,2) # 128X10X4
									)
		self.conv_output_size = 128*10*4

		self.fc_h1_v = NoisyLinear(self.conv_output_size, 4*args.hidden_size, std_init=args.noisy_std) # value head 1
		self.fc_h2_v = NoisyLinear(4*args.hidden_size, args.hidden_size, std_init=args.noisy_std) # value head 2
		self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std) # value head final

		self.fc_h1_a = NoisyLinear(self.conv_output_size, 4*args.hidden_size, std_init=args.noisy_std) # advantage head 1
		self.fc_h2_a = NoisyLinear(4*args.hidden_size, args.hidden_size, std_init=args.noisy_std)
		self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std) # advantage head final

	def forward(self, x, log=False):
		x = self.convs(x)
		x = x.view(-1, self.conv_output_size)

		v = F.relu(self.fc_h1_v(x))
		v = self.fc_z_v(F.relu(self.fc_h2_v(v)))  # value

		a = F.relu(self.fc_h1_a(x))
		a = self.fc_z_a(F.relu(self.fc_h2_a(a)))  # advantage

		v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
		q = v + a - a.mean(1, keepdim=True)  # Dueling DQN
		
		if log:  
			q = F.log_softmax(q, dim=2)
		else: 
			q = F.softmax(q, dim=2)
		return q

	def reset_noise(self): # noise reset -> noise 
		for name, module in self.named_children():
			if 'fc' in name:
				module.reset_noise()
