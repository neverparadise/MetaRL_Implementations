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
    def __init__(self, num_actions, device):
        self.device = device
        self.num_actions = num_actions

SEED = 3145  # some seed number here
#benchmark = metaworld.Benchmark(seed=SEED)

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
