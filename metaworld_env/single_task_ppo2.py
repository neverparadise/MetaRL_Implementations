from collections import deque

# Hyperparameters
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
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PPO(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_actions, device):
        super(PPO, self).__init__()
        self.num_actions = num_actions
        self.device = device
        self.fc_layers = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        )
        self.fc_pi = nn.Linear(hidden_dim, self.num_actions)
        self.fc_v = nn.Linear(hidden_dim, 1)
    
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
            x = x.to(device=self.device)
        else:
            x = x.unsqueeze(0)
            x = x.to(device=self.device)
        return x

    def forward(self, obs, softmax_dim=1):
        x = self._format(obs)
        x = self.fc_layers(x) # (Batch, Linear_size)
        prob = self.fc_pi(x)
        log_prob = F.softmax(prob, dim=softmax_dim)
        value = self.fc_v(x)
        return log_prob, value

class Buffer:
    def __init__(self, T_horizon=20, obs_dim=39, num_actions=4, device='cuda'):
        self.T = T_horizon
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.count = 0
        self.obs = torch.zeros((self.T, obs_dim)).to(device, dtype=torch.float)
        self.actions = torch.zeros((self.T, num_actions)).to(device, dtype=torch.int)
        self.rewards = torch.zeros((self.T, 1)).to(device, dtype=torch.float)
        self.next_obs = torch.zeros((self.T, obs_dim)).to(device, dtype=torch.float)
        self.log_probs = torch.zeros((self.T, num_actions)).to(device, dtype=torch.float)

    def reset(self):
        self.count = 0
        self.obs = torch.zeros((self.T, self.obs_dim)).to(device)
        self.actions = torch.zeros((self.T, self.num_actions)).to(device)
        self.rewards = torch.zeros((self.T, 1)).to(device)
        self.next_obs = torch.zeros((self.T, self.obs_dim)).to(device)
        self.log_probs = torch.zeros((self.T, 1)).to(device)

    def put_data(self, transition):
        self.obs[self.count] = torch.as_tensor(transition['obs'])
        self.actions[self.count] = torch.as_tensor(transition['action'])
        self.rewards[self.count] = torch.as_tensor(transition['reward'])
        self.next_obs[self.count] = torch.as_tensor(transition['next_obs'])
        self.log_probs[self.count] = torch.as_tensor(transition['log_prob'])
        self.count = (self.count + 1) % self.T

    def get_batch(self):
        return self.obs, self.actions, self.rewards, self.next_obs, self.log_probs
    
def train_net(model, buffer, optimizer, K_epoch, lmbda, gamma, eps_clip, entopy_coef):
    obs, a, r, next_obs, done_mask, prob_a = buffer.get_batch()
    for i in range(K_epoch):
        next_log_prob, next_value = model(next_obs)
        pi, value = model(obs) # [batch, 4]

        td_target = r + gamma * next_value * done_mask
        delta = td_target - value
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

        pi_a = pi.gather(1,a) 
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

        m = Categorical(pi)
        entropy = m.entropy().mean()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(value , td_target.detach()) - entopy_coef * entropy

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    
    return loss.mean().item()

# Hyperparameters
SEED = 3145  # some seed number here
T_HORIZON = 128
K_EPOCHS = 5
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
ENT_COEFF = 0.1
BATCH_SIZE = 32
LR = 0.0001

print(metaworld.ML1.ENV_NAMES)
# benchmark = metaworld.Benchmark(seed=SEED)
ml1 = metaworld.ML1('push-v2', seed=SEED) # Construct the benchmark, sampling tasks
tasks = MetaWorldTaskSampler(ml1, 'train')

task = random.choice(ml1.train_tasks)
env = ml1.train_classes['door-close-v2']()  # Create an environment with task `pick_place`
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
env.seed(SEED)

env.set_task(task)  # Set task
max_length = env.max_path_length
success_rate = 0
print(env.observation_space)
print(env.action_space)

agent = PPO(obs_dim=39, hidden_dim=128, num_actions=4, device=device)
agent = agent.to(device)
storage = Storage(T_horizon=20, obs_dim=39, num_actions=4, device=device)
optimizer = optim.Adam(agent.parameters(), lr=LR)
print(os.curdir)
WEIGHT_PATH = "/home/kukjin/Projects/MetaRL/MetaRL_Implementations/metaworld_env/weights/PPO.pt"
LOG_DIR = os.curdir + "/experiemnts/ppo"
writer = SummaryWriter(LOG_DIR)
