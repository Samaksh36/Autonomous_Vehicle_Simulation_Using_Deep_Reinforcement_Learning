import gym
import highway_env

import os
import utils
import numpy as np
import matplotlib.pyplot as plt

import DuelingDQNAgent as duelingdqnAgent


def train_env():
    env = gym.make('intersection-v0')
    env.seed(0)
    env.configure({"controlled_vehicles": 2})
    env.configure({"vehicles_count": 0})
    env.configure({"vehicles_density": 0})
    env.configure({
      "action": {
        "type": "MultiAgentAction",
        "action_config": {
          "type": "DiscreteMetaAction",
        }
      }
    })
    env.configure({
      "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
          "type": "Kinematics",
        }
      }
    })
    env.configure({
      "observation": {
      "type": "GrayscaleObservation",
      "observation_shape": (128, 64),
      "stack_size": 4,
      "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
      "scaling": 1.75,
      },
    })
    
    
    env.reset()
    return env


def test_env():
    env = train_env()
    env.configure({"policy_frequency": 15, "duration": 20 * 15})
    env.reset()
    return env

env = train_env()
n_games = 100
scores = []
epsilon_history = []
steps = []
step_count = 0
best_score = -np.inf

load_checkpoint = True

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint/')

if not checkpoint_dir:
    os.mkdir(checkpoint_dir) 

print(checkpoint_dir)

# creates an agent
agent = duelingdqnAgent.DuelingDQNAgent(learning_rate=0.0001, n_actions=env.action_space[0].n,
                                        input_dims=env.observation_space.shape, gamma=0.99,
                                        epsilon=0.2, batch_size=32, memory_size=1000,
                                        replace_network_count=1000,
                                        checkpoint_dir=checkpoint_dir)

agent.load_model()
if load_checkpoint:
    agent.load_model()


env_ = train_env()

for i in range(n_games):
    obs = env_.reset()
    score = 0
    done = False
    episode_size = 0
    while not done:
        episode_size += 1
        action_1, action_2 = agent.choose_action(obs)
        new_obs, reward, done, info = env_.step((action_1, action_2))
        score += reward
        env_.render()
        if not load_checkpoint:
            agent.store_experience(obs, action_1, action_2, reward, new_obs, int(done))
            agent.learn()
        obs = new_obs
        step_count += 1

    scores.append(score)
    epsilon_history.append(agent.epsilon)
    steps.append(step_count)
    avg_score = np.mean(scores)

    if score > avg_score:
        if not load_checkpoint:
            agent.save_model()

    if score > best_score:
        best_score = score

    print('episode: ', i, ' score: ', score, ' avg. score: ', avg_score,
          ' best_score: ', best_score, ' epsilon: ', agent.epsilon, ' steps ', step_count)






