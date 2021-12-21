import gym
import highway_env
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from Rainbow.model import DQN # rainbow dqn
import Rainbow.memory
from Rainbow.memory import ReplayMemory # replay memory for rainbow dqn
from Rainbow.agent import Agent 


from tqdm import trange
from datetime import datetime
import bz2
import pickle
import joblib

import time

# ALSA error
os.environ['SDL_AUDIODRIVER'] = 'dsp'
#############################
### argument parser #########
#############################

parser = argparse.ArgumentParser(description='Time Retract RL')
parser.add_argument('--tensorboard', default="./log", help="directory for saving tensorboard" )
parser.add_argument('--envs', default="highway-v0", help="directory for saving tensorboard" )
parser.add_argument('--test', action='store_true', help="True is training, False is test")
parser.add_argument('--test_model', type=str, default='checkpoint.pth', help="The name of model parameter file")

parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(1e6), help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(1e4), help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='large-image', help='Network architecture') # large image 로만 setting
parser.add_argument('--hidden-size', type=int, default=128, help='Network hidden size') # network hidden dim
parser.add_argument('--noisy-std', type=float, default=0.5, help='Initial standard deviation of noisy linear layers') # 클 수록 exploration 성능이 증가한다.
parser.add_argument('--atoms', type=int, default=51, help='Discretised size of value distribution') # distribution RL에서 사용하는 bin의 개수
parser.add_argument('--V-min', type=float, default=-10, help='Minimum of value distribution support') #distribution RL 최솟값
parser.add_argument('--V-max', type=float, default=10, help='Maximum of value distribution support') # 최댓값
parser.add_argument('--memory-capacity', type=int, default=int(5e5), help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=2, help='Frequency of sampling from memory') # 너무 작으면 학습속도 느려질 수도??
parser.add_argument('--reset-frequency', type=int, default=2, help='Frequency of reset noise')
parser.add_argument('--priority-exponent', type=float, default=0.5, help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.6, help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=2, help='Number of steps for multi-step return') # 수업시간에 n이 크면 안좋다고 했다..!!!
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(5e2), help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=10, help='Reward clipping (0 to disable)') # Atari와는 환경이 달라서 clipping 은 사용 안함
parser.add_argument('--learning-rate', type=float, default=0.0001,  help='Learning rate') # 조절 필요 1e-4 정도가 적절했다.
parser.add_argument('--adam-eps', type=float, default=1.5e-4, help='Adam epsilon') # 조절하지 않음
parser.add_argument('--batch-size', type=int, default=128, metavar='SIZE', help='Batch size') # 가능한 크게
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping') # gradient clipping 용도
parser.add_argument('--learn-start', type=int, default=int(10e3), metavar='STEPS', help='Number of steps before starting training') # 초기에는 random policy 를 사용한다.
parser.add_argument('--checkpoint-interval', default=200, help='How often to checkpoint the model, defaults to 0 (never checkpoint)') # 모델 저장
parser.add_argument('--observation_type', type=str, default='image_large', help='obseravtion type, defaults to image') # image로 할 지 kinematics로 할 지
parser.add_argument('--gpu_number', type=int, default=-1, help="the number of GPU") # GPU 번호
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--vehicle-density', type=float, default=2.0, help='vehicle density')

parser.add_argument('--video_path', type=str, default='./video/RainbowDQN.mp4', help='video save path' ) # 비디오 저장 경로

args = parser.parse_args()
print(torch.cuda.is_available())
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    if args.gpu_number == -1:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu_number))
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    print("using CPU")
    args.device = torch.device('cpu')

results_dir = os.path.join('ASP_results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if args.observation_type == 'image_large':
    args.architecture = 'large-image'
    Rainbow.memory.chage_global(np.dtype([('timestep', np.int32), ('state', np.uint8, (160, 64)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)]),
                                (0, np.zeros((160, 64), dtype=np.uint8), 0, 0.0, False))


#############################
### argument parser #########
#############################

def lmap(v, x, y):
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

def highway_config(obs):
    if obs == "kinematics":
        config = {
            "lanes_count": 4,
            "vehicles_count": 20,
            "duration": 80,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": [
                    "presence",
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "cos_h",
                    "sin_h"
                ],
                "absolute": False
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "show_trajectories" : False
        }
        return config

    elif obs == "image": 
        screen_width, screen_height = 84, 84
        config = {
            # "offscreen_rendering": True,
            "lanes_count": 4,
            "vehicles_count": 50,
            "vehicles_density" : 2,
            "duration": 200,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "observation": {
                "type": "GrayscaleObservation",
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "stack_size": 4,
                "observation_shape": (screen_width, screen_height)
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "screen_width": screen_width,
            "screen_height": screen_height,
            "scaling": 1.75,
            "policy_frequency": 3,
            "show_trajectories" : False,
            'collision_reward': -100.0,
            'controlled_vehicles': 1,
            'reward_speed_range': [20, 30],
            'lane_change_reward': -1.0,
            'simulation_frequency': 15,
        }
        return config
    
    elif obs == "image_large": 
        screen_width, screen_height = 160, 64
        
        config = {
            # "offscreen_rendering": True,
            "lanes_count": 4,
            "vehicles_count": 30,
            "vehicles_density" : args.vehicle_density,
            "duration": 100,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "centering_position": [0.3, 0.5],
            "observation": {
                "type": "GrayscaleObservation",
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "stack_size": 4,
                "observation_shape": (screen_width, screen_height)
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "screen_width": screen_width,
            "screen_height": screen_height,
            "scaling": 1.75,
            "policy_frequency": 3,
            "show_trajectories" : False,
            'collision_reward': -100.0,
            'controlled_vehicles': 1,
            'reward_speed_range': [20, 30],
            'lane_change_reward': -1.0,
            'simulation_frequency': 15,
        }
        return config

def train():
    date = datetime.now()
    summary = SummaryWriter(os.path.join(args.tensorboard,"rainbowDQN", date.strftime("%Y%b%d_%H_%M_%S"))) # Tensorboard
    env = gym.make(args.envs)
    env.configure(highway_config(args.observation_type)) # configuration
    state = env.reset()

    mem = ReplayMemory(args, args.memory_capacity) # memory capacity 

    dqn = Agent(args, env) # rainbow dqn
    # dqn.load(results_dir, "checkpoint.pth")
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start) # important sampling 

    T, done = 0, True
    episode, r_episode, action_count = -1, 0, np.zeros(5)
    max_episode = args.max_episode_length


    for T in trange(1, args.T_max + 1): # num of time step
        if done:
            if episode > max_episode:
                print("training done")
                break
            if episode % 10 == 0 and episode != 0:
                summary.add_scalar('Average reward', r_episode / 10.0, episode)

                r_episode = 0
                action_count = np.zeros(5)
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255.)
            episode += 1 # episode done
            
        if T% args.reset_frequency == 0:
            dqn.reset_noise() # noisy network reset 해준다. For exploration

        if T >= args.learn_start:
            action = dqn.act(state) # action select
        else:
            action = np.random.randint(0, env.action_space.n) # random policy

        action_count[action] += 1 # countint actions
        next_state, reward, done, _ = env.step(action) # interection with env
        next_state = torch.tensor(next_state, dtype=torch.float32, device=args.device).div_(255.)

        # reward_lmap = lmap(reward, 
        #                     [0, 1],
        #                     [-100.0, 1.1]) # summary에 표시할 reward
        r_episode += reward # for summary record

        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip) # reward clipping -> distribution RL
        
        mem.append(state, action, reward, done) # replay experience

        if T >= args.learn_start: # replay buffer 
            # Anneal importance sampling weight β to 1 (importance factor )
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if T % args.replay_frequency == 0:
                dqn.learn(mem) # Train with n-step distributional double-Q learning
            
            if T % args.target_update == 0:
                dqn.update_target_net()

            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn.save(results_dir, 'checkpoint{}.pth'.format(args.vehicle_density))


        state = next_state



def test():
    from gym.wrappers.monitoring import video_recorder

    env = gym.make(args.envs)
    env.configure(highway_config(args.observation_type))

    state = env.reset()

    dqn = Agent(args,env)
    dqn.load(results_dir, args.test_model)

    for episode in range(50):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=args.device).div_(255.)
        reward_sum = 0
        for t in range(38):
            
            action = dqn.act(state)
            state_next, r, done, _ = env.step(action)
            state_next = torch.tensor(state_next, dtype=torch.float32, device=args.device).div_(255.)
            reward_sum += r
            state = state_next

            env.render()
            
            if done:
                print("episode {}\t reward : {}".format(episode, reward_sum))
                break

if not args.test:
    train()
else:
    test()
    