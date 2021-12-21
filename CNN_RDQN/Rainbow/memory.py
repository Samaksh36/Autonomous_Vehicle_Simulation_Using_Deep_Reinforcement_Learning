# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch


Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)

def chage_global(new_dtype, new_blank): # input image size가 바뀔 경우 다음 함수를 사용한다.
	global Transition_dtype, blank_trans
	Transition_dtype = new_dtype
	blank_trans = new_blank


class SegmentTree():
	"""
	BST 구조
	sum_tree 는 Transition_dtype 의 priority
	data는 실제 Transition_dtype들의 Tree
	priority가 높은 곳에서 더 많이 sampling 되도록 짜여진 자료구조
	"""
	def __init__(self, size):
		self.index = 0
		self.size = size # PER 사이즈와 동일
		self.full = False
		self.tree_start = 2**(size-1).bit_length()-1  # Binary tree의 dummy leaf 개수
		self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32) # dummy leaf + 전체 사이즈
		self.data = np.array([blank_trans] * size, dtype=Transition_dtype) # element는 Transition dtype와 같은 datatype으로 들어간다. default는 blank trans
		self.max = 1  # Initial max value to return (1 = 1^ω)

	def _update_nodes(self, indices):
		children_indices = indices * 2 + np.expand_dims([1, 2], axis=1) # binary tree에서 children 의 index
		self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0) # children 두개 더한걸로 parent 값 update

	def _propagate(self, indices): # index 여러개에 대해서 한번에
		parents = (indices - 1) // 2 # BT에서 parent
		unique_parents = np.unique(parents) # 여러개면 하나만 (같은 부모를 가진 노드인 경우)
		self._update_nodes(unique_parents) 
		if parents[0] != 0: # 위쪽으로 쭉 recursion
			self._propagate(parents)

	def _propagate_index(self, index): # index 하나에 대해서 propagate recursion
		parent = (index - 1) // 2 
		left, right = 2 * parent + 1, 2 * parent + 2
		self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
		if parent != 0:
			self._propagate_index(parent)

	def update(self, indices, values): # index 여러개
		self.sum_tree[indices] = values  # node 하나의 값 update
		self._propagate(indices)  # propagate
		current_max_value = np.max(values) 
		self.max = max(current_max_value, self.max) # 지금까지 들어왔던 values 중 최대값 저장 

	def _update_index(self, index, value): # index 한개
		self.sum_tree[index] = value
		self._propagate_index(index)
		self.max = max(value, self.max)

	def append(self, data, value): # 데이터 넣기
		self.data[self.index] = data  # BT 맨 뒤에다 데이터 넣기
		self._update_index(self.index + self.tree_start, value) # sum tree 업데이트
		self.index = (self.index + 1) % self.size 
		self.full = self.full or self.index == 0  # 꽉 찼는지 판단
		self.max = max(value, self.max)

	# Searches for the location of values in sum tree
	def _retrieve(self, indices, values):
		children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # 각각의 children index

		if children_indices[0, 0] >= self.sum_tree.shape[0]: #전체 사이즈를 넘어선 경우
			return indices
		elif children_indices[0, 0] >= self.tree_start: # 범위 밖으로 나간 경우
			children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)

		left_children_values = self.sum_tree[children_indices[0]]
		successor_choices = np.greater(values, left_children_values).astype(np.int32)  # value 가 left child 보다 크면 True
		successor_indices = children_indices[successor_choices, np.arange(indices.size)] # value가 left보다 크면 right, 아니면 left 선택
		successor_values = values - successor_choices * left_children_values  # value에서 left 만큼 값 빼줌, left로 가는 경우에는 안빼줌
		return self._retrieve(successor_indices, successor_values) # child 로 넘어가서 recursion

	def find(self, values):
		indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values) # root 부터 아래로 내려가면서 value에 해당하는 곳 찾기
		data_index = indices - self.tree_start # 실제 데이터 위치
		return (self.sum_tree[indices], data_index, indices)

	def get(self, data_index):
		return self.data[data_index % self.size]

	def total(self): # 총 확률 합
		return self.sum_tree[0]


class ReplayMemory():
	"""
	Prioritized memory replay 가 구현되어 있는 모듈이다.
	"""
	def __init__(self, args, capacity):
		self.device = args.device
		self.capacity = capacity
		self.history = args.history_length
		self.discount = args.discount
		self.n = args.multi_step
		self.priority_weight = args.priority_weight # importance sampling factor
		self.priority_exponent = args.priority_exponent # importance sampling factor
		self.t = 0 
		self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device) # multi step DQN 을 위한 discounting factor 미리 계산
		self.transitions = SegmentTree(capacity)

	def append(self, state, action, reward, terminal):
		state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # state 스택에서 마지막 layer만 저장한다. 어차피 들어오는 데이터가 계속 겹치기 때문에
		self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max) # append
		self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

	def _get_transitions(self, idxs): # idxs - history ~ idxs + n 까지의 데이터 얻어오는 것
		transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1) # idxs - history ~ idxs + n 까지
		transitions = self.transitions.get(transition_idxs) # 데이터 불러오기 
		transitions_firsts = transitions['timestep'] == 0  # self.t 가 0인 녀석을 찾는 것 -> 시작점 찾기
		blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
		for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
			blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # t+1 step이 transition 시작점이면 == t step이 terminal이면 True
		for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
			blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # t step 또는 t-1 step이 transition 시작점이면 True
		transitions[blank_mask] = blank_trans # blank mask 인 곳은 default data(blank trans) 로 채워놓기

		return transitions


	def _get_samples_from_segments(self, batch_size, p_total): 
		segment_length = p_total / batch_size  # 전체 확률 합 / batch size
		segment_starts = np.arange(batch_size) * segment_length # segment_length의 간격으로 batchsize개수만큼 -> 마지막 값은 p_total이 됨
		valid = False
		while not valid:
			samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # 전체 확률에서 균일하게 batch를 뽑기 위함
			probs, idxs, tree_idxs = self.transitions.find(samples)  # 일단 뽑아보고, 아래에서 valid check
			if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
				valid = True  # idxs + n 까지 sampling 할꺼고, idxs - history 까지 sampling 해야 하므로, 또한 확률도 0이면 안된다.
		transitions = self._get_transitions(idxs) # making 된 data 받아오기

		all_states = transitions['state'] # state만 가져오기
		states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255) # state 쌓아서 사용
		next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255) # n step 뒤의 state

		actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device) # states의 마지막과 같은 action
		rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device) # n step reward 전부 불러오기
		R = torch.matmul(rewards, self.n_step_scaling) # n step return으로 게산하기

		done_mask = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device) # done mask
		
		return probs, idxs, tree_idxs, states, actions, R, next_states, done_mask

	def sample(self, batch_size):
		p_total = self.transitions.total() # 전체 확률 합
		probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total) # 위의 함수로 sampling 하기
		probs = probs / p_total  # 전체 확률을 1로 normalize
		capacity = self.capacity if self.transitions.full else self.transitions.index # 꽉찼는지 확인
		weights = (capacity * probs) ** -self.priority_weight # importance weight 계산 --> 나중에 loss 에 곱해지는 값
		weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # 가장 큰 값을 1로 normalize

		return tree_idxs, states, actions, returns, next_states, nonterminals, weights

	def update_priorities(self, idxs, priorities):
		priorities = np.power(priorities, self.priority_exponent) # priority에 계속 곱해줌
		self.transitions.update(idxs, priorities) # 전체 segment tree update

	def __iter__(self): # 내부에서 뭔가 반복문이 돌 때 끝까지 돌아버려도 다시 돌 수 있게 overiding
		self.current_idx = 0

		return self

	def __next__(self):
		if self.current_idx == self.capacity:
			raise StopIteration # memory 한계 끝까지 돌면 반복 멈추도록
		# _get_samples_from_segments 에서 했던 것 반복 -> 반복 멈춘 곳에서 sampling 할 수 있도록 해줌 
		transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
		transitions_firsts = transitions['timestep'] == 0
		blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
		for t in reversed(range(self.history - 1)):
			blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])
		transitions[blank_mask] = blank_trans
		state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)
		self.current_idx += 1

		return state

	next = __next__  # python 2에서도 동작하도록 (혹시 몰라서 남겨 두자. 학습하다가 중간에 멈추면 큰일이니까...)
