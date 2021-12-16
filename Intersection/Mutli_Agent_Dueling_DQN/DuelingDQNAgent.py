import DuelingDeepQNetwork as ddqn
import ExperienceReplayMemory as erm

import numpy as np
import torch

# n_actions is number of actions for a single Agent
# We assume having 2 Agents having n_actions each


class DuelingDQNAgent(object):
    def __init__(self, learning_rate, n_actions, input_dims, gamma,
                 epsilon, batch_size, memory_size, replace_network_count,
                 dec_epsilon=1e-5, min_epsilon=0.1, checkpoint_dir='/tmp/ddqn/'):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_network_count = replace_network_count
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.checkpoint_dir = checkpoint_dir
        self.action_indices = [i for i in range(n_actions)]
        self.learn_steps_count = 0

        self.q_eval = ddqn.DuelingDeepQNetwork(learning_rate=learning_rate, n_actions=n_actions,
                                               input_dims=input_dims, name='q_eval',
                                               checkpoint_dir=checkpoint_dir)

        self.q_next = ddqn.DuelingDeepQNetwork(learning_rate=learning_rate, n_actions=n_actions,
                                               input_dims=input_dims, name='q_next',
                                               checkpoint_dir=checkpoint_dir)

        self.experience_replay_memory = erm.ExperienceReplayMemory(memory_size=memory_size,
                                                                   input_dims=input_dims)

    def decrement_epsilon(self):
        """
        Decrements the epsilon after each step till it reaches minimum epsilon (0.1)
        epsilon = epsilon - decrement (default is 0.99e-6)
        """
        self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon \
            else self.min_epsilon

    def store_experience(self, state, action_1, action_2, reward, next_state, done):
        """
        Saves the experience to the replay memory
        """
        self.experience_replay_memory.add_experience(state=state, action_1=action_1, action_2=action_2,
                                                     reward=reward, next_state=next_state,
                                                     done=done)

    def get_sample_experience(self):
        """
        Gives a sample experience from the experience replay memory
        """
        state, action_1, action_2, reward, next_state, done = self.experience_replay_memory.get_random_experience(
            self.batch_size)

        t_state = torch.tensor(state).to(self.q_eval.device)
        t_action_1 = torch.tensor(action_1).to(self.q_eval.device)
        t_action_2 = torch.tensor(action_1).to(self.q_eval.device)
        t_reward = torch.tensor(reward).to(self.q_eval.device)
        t_next_state = torch.tensor(next_state).to(self.q_eval.device)
        t_done = torch.tensor(done).to(self.q_eval.device)

        return t_state, t_action_1, t_action_2, t_reward, t_next_state, t_done

    def replace_target_network(self):
        """
        Updates the parameters after replace_network_count steps
        """
        if self.learn_steps_count % self.replace_network_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, observation):
        """
        Chooses an action with epsilon-greedy method
        """
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.q_eval.device)
            
            value, advantage, loss = self.q_eval.forward(state)
            advantage_1 = advantage[0]
            advantage_2 = advantage[1]

            action_1 = torch.argmax(advantage_1).item()
            action_2 = torch.argmax(advantage_2).item()
        else:
            action_1 = np.random.choice(self.n_actions)
            action_2 = np.random.choice(self.n_actions)

        return (action_1, action_2)

    def learn(self):
        if self.experience_replay_memory.counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action_1, action_2, reward, next_state, done = self.get_sample_experience()
        # Gets the evenly spaced batches
        batches = np.arange(self.batch_size)

        # Gets value and advantage by feed forwarding the current states and next states
        value_s, advantage_s = self.q_eval.forward(state)
        value_s_dash, advantage_s_dash = self.q_next.forward(next_state)
        
        #Get value and advantage for each agent
        value_s_1, value_s_2 = value_s
        advantage_s_1, advantage_s_2 = advantage_s
        value_s_dash_1, value_s_dash_2 = value_s_dash
        advantage_s_dash_1, advantage_s_dash_2 = advantage_s_dash
        
        # Computes the Q values using
        # Q = V + (A - (1/|A|) * (sum(A))
        q_pred_1 = torch.add(value_s_1, advantage_s_1 - advantage_s_1.mean(dim=1, keepdim=True))[batches, action_1]
        q_pred_2 = torch.add(value_s_2, advantage_s_2 - advantage_s_2.mean(dim=1, keepdim=True))[batches, action_2]
        
        q_next_1 = torch.add(value_s_dash_1, advantage_s_dash_1 - advantage_s_dash_1.mean(dim=1, keepdim=True))
        q_next_2 = torch.add(value_s_dash_2, advantage_s_dash_2 - advantage_s_dash_2.mean(dim=1, keepdim=True))
        
        max_q_next_1 = q_next_1.max(dim=1)
        max_q_next_2 = q_next_2.max(dim=1)

        q_next_1[done] = 0.0
        q_next_2[done] = 0.0
        
        q_target_1 = reward + self.gamma * max_q_next_1[0]
        q_target_2 = reward + self.gamma * max_q_next_2[0]

        # Computes loss and performs backpropagation
        loss = self.q_eval.loss(q_target_1, q_pred_1).to(self.q_eval.device) + self.q_eval.loss(q_target_2, q_pred_2).to(self.q_eval.device)
                
        loss.backward()

        self.q_eval.optimizer.step()
        self.decrement_epsilon()
        self.learn_steps_count += 1

    def save_model(self):
        """
        Saves the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_model(self):
        """
        Loads the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()