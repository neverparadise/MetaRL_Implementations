from collections import namedtuple
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
import torch.nn.functional as F
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
        super().__init__()
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

    def get_action_prob_entropy_value(self, obs, action=None):
        hidden = self.forward(obs)
        
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)

class Storage:
    def __init__(self, T_horizon=20, obs_dim=39, num_actions=4, device='cuda'):
        self.T = T_horizon
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.count = 0
        self.obs = torch.zeros((self.T, obs_dim)).to(device, dtype=torch.float)
        self.actions = torch.zeros((self.T, num_actions)).to(device, dtype=torch.int)
        self.rewards = torch.zeros((self.T, 1)).to(device, dtype=torch.float)
        self.next_obs = torch.zeros((self.T, obs_dim)).to(device, dtype=torch.float)
        self.log_probs = torch.zeros((self.T, 1)).to(device, dtype=torch.float)
        self.values = torch.zeros((self.T, 1)).to(device, dtype=torch.float)

    def reset(self):
        self.count = 0
        self.obs = torch.zeros((self.T, self.obs_dim)).to(device)
        self.actions = torch.zeros((self.T, self.num_actions)).to(device)
        self.rewards = torch.zeros((self.T, 1)).to(device)
        self.next_obs = torch.zeros((self.T, self.obs_dim)).to(device)
        self.log_probs = torch.zeros((self.T, 1)).to(device)
        self.values = torch.zeros((self.T, 1)).to(device)

    def put_data(self, transition):
        self.obs[self.count] = torch.as_tensor(transition['obs'])
        self.actions[self.count] = torch.as_tensor(transition['action'])
        self.rewards[self.count] = torch.as_tensor(transition['reward'])
        self.next_obs[self.count] = torch.as_tensor(transition['next_obs'])
        self.log_probs[self.count] = torch.as_tensor(transition['log_prob'])
        self.values[self.count] = torch.as_tensor(transition['value'])
        self.count = (self.count + 1) % self.T
    
    def choose_mini_batch(self, batch_size, advantages, returns):
        for _ in range(self.T // batch_size):
            indices = torch.random.randint(0, self.T, batch_size)
            yield self.obs[indices], self.actions[indices], self.rewards[indices], \
                    self.next_obs[indices], self.log_probs[indices], self.values[indices], \
                        advantages[indices], returns[indices]

    def get_batch(self):
        return self.obs, self.actions, self.rewards, self.next_obs, self.log_probs, self.values

def train_policy(agent, storage, optimizer, batch_size, T_HORIZON, GAMMA, LAMBDA, EPS_CLIP, ENT_COEFF):
    obs, actions, rewards, next_obs, old_log_pobs, values = storage.get_batch()
    td_target = rewards + GAMMA, * agent.get_value(next_obs)
    delta = td_target - values
    with torch.no_grad():
        delta = delta.cpu().numpy()
        advantages = torch.zeros_like(delta)
        advantages = advantages.to(device, dtype=torch.float)
        adv = 0.0
        for idx in reversed(range(T_HORIZON)):
            adv = GAMMA * LAMBDA * adv + delta[idx]
            advantages[idx] = adv
        returns = advantages + values
    for i in range(K_EPOCHS):
        for s_, a_, r_, ns_, old_log_p_, v_, adv_, ret_ in agent.choose_mini_batch(batch_size, advantages, returns):
            _, cur_log_p, entorpy, cur_v = agent.get_action_prob_entropy_value(s_)
            ratio = torch.exp(cur_log_p - old_log_p_).detach()
            surr1 = ratio * adv_
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * adv_
            actor_loss = (-torch.min(surr1, surr2) - entropy * ENT_COEFF).mean()
            critic_loss = F.smooth_l1_loss(cur_v, td_target)
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Hyperparameters
SEED = 3145  # some seed number here
T_HORIZON = 128
K_EPOCHS = 5
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
BATCH_SIZE = 32
ENT_COEFF = 0.1

# benchmark = metaworld.Benchmark(seed=SEED)
ml1 = metaworld.ML1('door-close-v2', seed=SEED) # Construct the benchmark, sampling tasks
env = ml1.train_classes['door-close-v2']()  # Create an environment with task `pick_place`
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
env.seed(SEED)

task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
max_length = env.max_path_length
success_rate = 0
num_successful_trajectories = 0

agent = PPO(obs_dim=39, hidden_dim=128, num_actions=4, device=device)
agent = agent.to(device)
storage = Storage(T_horizon=20, obs_dim=39, num_actions=4, device=device)
print(env.observation_space)
print(env.action_space)


for e in range(10000):
    obs = env.reset()  # Reset environment
    #print(obs)
    done = False
    score = 0
    step = 0
    success_curr_time_step = 0.0

        
    while step < max_length and not done:
        env.render()
        with torch.no_grad():
            action, log_prob, entropy, value  = agent.get_action_prob_entropy_value(obs)  # Sample an action

        next_obs, reward, done, info = env.step(action.squeeze().cpu().numpy())  # Step the environoment with the sampled random action
        step += 1
        next_obs = obs
        transition = {"obs":obs, "action":action, "reward":reward, \
                        "next_obs":next_obs, "log_prob":log_prob, "value":value}
        storage.put_data(transition)
        score += reward
        if step & 
        
        #print(info['success'])
        #success_curr_time_step += info['success']

        if step == max_length:
            print(score)
            break
    num_successful_trajectories += int(success_curr_time_step)
print(num_successful_trajectories)
