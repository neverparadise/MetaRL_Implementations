import metaworld
import random
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _format(state, device):
    x = state
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(device=device)
    else:
        x = x.unsqueeze(0)
        x = x.to(device=device)
    return x

class PPO(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, num_actions=4, device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        self.critic = nn.Linear(hidden_dim, 1)
        self.actor_mean = nn.Linear(hidden_dim, num_actions)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, obs):
        obs = _format(obs, self.device)
        hidden = self.fc(obs)
        return hidden
    
    def get_value(self, obs):
        hidden = self.forward(obs)
        value = self.critic(hidden)
        return value

    def get_action_and_probs(self, obs, action=None):
        hidden = self.forward(obs)
        
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

class Storage:
    def __init__(self, T=20, obs_dim=39, num_actions=4, device='cuda'):
        self.T_horizon = T
        self.obs = torch.zeros((T, obs_dim)).to(device)
        action = torch.zeros((T, num_actions)).to(device)




SEED = 3145  # some seed number here
#benchmark = metaworld.Benchmark(seed=SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ml1 = metaworld.ML1('door-close-v2', seed=SEED) # Construct the benchmark, sampling tasks

env = ml1.train_classes['door-close-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
env.seed(3145)
max_length = env.max_path_length
success_rate = 0
num_successful_trajectories = 0

for e in range(2):
    obs = env.reset()  # Reset environment
    #print(obs)
    done = False
    score = 0
    step = 0
    success_curr_time_step = 0.0

    
    while step < max_length and not done:
        env.render()
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        step += 1
        print(info['success'])
        success_curr_time_step += info['success']
        score += reward
        if step > max_length:
            print(score)
            break
    num_successful_trajectories += int(success_curr_time_step)
print(num_successful_trajectories)
