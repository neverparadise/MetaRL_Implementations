from collections import namedtuple
from curses import A_ALTCHARSET
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
import collections

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SEED = 3145  # some seed number here
T_HORIZON = 128
K_EPOCHS = 5
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
ENT_COEFF = 0.1
BATCH_SIZE = 32
LR_PI = 0.0001
LR_Q = 0.0005
BUFFER_SIZE = 200000
TAU = 0.01
TARGET_ENTROPY = -1.0
INIT_ALPHA = 0.01
LR_ALPHA = 0.001

class ReplayBuffer_T:
    def __init__(self, buffer_size=100000, obs_dim=39, num_actions=4, device=device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.count = 0
        self.obs = torch.zeros((self.buffer_size, obs_dim)).to(device, dtype=torch.float)
        self.actions = torch.zeros((self.buffer_size, num_actions)).to(device, dtype=torch.int)
        self.rewards = torch.zeros((self.buffer_size, 1)).to(device, dtype=torch.float)
        self.next_obs = torch.zeros((self.buffer_size, obs_dim)).to(device, dtype=torch.float)
        self.dones = torch.zeros((self.buffer_size, 1)).to(device, dtype=torch.int)

    def put(self, transition):
        self.obs[self.count] = torch.as_tensor(transition[0])
        self.actions[self.count] = torch.as_tensor(transition[1])
        self.rewards[self.count] = torch.as_tensor(transition[2])
        self.next_obs[self.count] = torch.as_tensor(transition[3])
        self.dones[self.count] =  torch.as_tensor(transition[4])
        self.count = (self.count + 1) % self.buffer_size
    
    def sample(self, batch_size):
        indices = torch.randint(0, self.buffer_size, (batch_size,))
        return self.obs[indices], self.actions[indices], self.rewards[indices], \
                    self.next_obs[indices], self.dones[indices]
                    
    def size(self):
        return (self.obs.shape[0])

    
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states = torch.empty(n, 39).to(device, dtype=torch.float)
        actions = torch.empty(n, 4).to(device, dtype=torch.int)
        rewards = torch.empty(n, 1).to(device, dtype=torch.float)
        next_states = torch.empty(n, 39).to(device, dtype=torch.float)
        dones = torch.empty(n, 1).to(device, dtype=torch.float)

        for i, transition in enumerate(mini_batch):
            s, a, r, s_prime, done = transition
            states[i] = torch.as_tensor(s)
            actions[i] = torch.as_tensor(a)
            rewards[i] = torch.as_tensor(r)
            next_states[i] = torch.as_tensor(s_prime)
            dones[i] = torch.as_tensor(done)

            # states[i] = torch.tensor(s).to(device, dtype=torch.float).clone().detach().requires_grad_(True)
            # actions[i] = torch.tensor(a).to(device, dtype=torch.int).clone().detach().requires_grad_(True)
            # rewards[i] = torch.tensor(r).to(device, dtype=torch.float).clone().detach().requires_grad_(True)
            # next_states[i] = torch.tensor(s_prime).to(device, dtype=torch.float).clone().detach().requires_grad_(True)
            # dones[i] = torch.tensor(0.0).to(device, dtype=torch.float).requires_grad_(True) if done else torch.tensor(1.0).to(device, dtype=torch.float)
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)
    
    
def _format(state, device):
    x = state
    if not isinstance(x, torch.Tensor): # one transition
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(device=device)
    else: # batch transitions
        x = x.to(device=device, dtype=torch.float32)
    return x
        
        
class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(39, 128)
        self.fc_mu = nn.Linear(128,4)
        self.fc_std  = nn.Linear(128,4)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(INIT_ALPHA))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

    def forward(self, x):
        x = _format(x, device)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + TARGET_ENTROPY).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(39, 64)
        self.fc_a = nn.Linear(4,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,4)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        x = _format(x, device)
        a = _format(a, device)
        
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - TAU) + param.data * TAU)


def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + GAMMA * done * (min_q + entropy)

    return target


# benchmark = metaworld.Benchmark(seed=SEED)
ml1 = metaworld.ML1('door-close-v2', seed=SEED) # Construct the benchmark, sampling tasks
env = ml1.train_classes['door-close-v2']()  # Create an environment with task `pick_place`
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
env.seed(SEED)

task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task0
max_length = env.max_path_length
success_rate = 0
print(env.observation_space)
print(env.action_space)

q1, q2 = QNet(LR_Q).to(device), QNet(LR_Q).to(device)
q1_target, q2_target = QNet(LR_Q).to(device), QNet(LR_Q).to(device)
pi = PolicyNet(LR_PI).to(device)
print(os.curdir)
PI_WEIGHT_PATH = "/home/kukjin/Projects/MetaRL/MetaRL_Implementations/metaworld_env/weights/SAC_PI.pt"
Q_WEIGHT_PATH = "/home/kukjin/Projects/MetaRL/MetaRL_Implementations/metaworld_env/weights/SAC_Q.pt"
LOG_DIR = os.curdir + "/experiemnts/sac"


writer = SummaryWriter(LOG_DIR)
memory = ReplayBuffer_T(50000)
score = 0.0

for e in range(1000000):
    obs = env.reset()  # Reset environment
    #print(obs)
    done = False
    print_interval = 5
    step = 0
    num_successful_trajectories = 0
    success_curr_time_step = 0.0
    total_rewards = 0
    # done은 없지만 성공해서 done 되고 추가적인 스텝을 진행할 수도 있다. 
    while step < max_length and not done:
        # env.render()
        with torch.no_grad():
            a, log_prob  = pi(obs)  # Sample an action
            a = a.squeeze(0)
            next_obs, reward, done, info = env.step(a.cpu().numpy())  # Step the environoment with the sampled random action
        step += 1
        score += reward
        total_rewards += reward
        memory.put((obs, a, reward/10.0, next_obs, done))
        next_obs = obs
        
        if memory.size()>1000:
            mini_batch = memory.sample(BATCH_SIZE) #)
            td_target = calc_target(pi, q1_target, q2_target, mini_batch)
            q1.train_net(td_target, mini_batch)
            q2.train_net(td_target, mini_batch)
            entropy = pi.train_net(q1, q2, mini_batch)
            q1.soft_update(q1_target)
            q2.soft_update(q2_target)
        


        #print(info['success'])
        success_curr_time_step += info['success']

        if step == max_length:
            writer.add_scalar("total_rewards", total_rewards, e)
            print(f"episode: {e} is finished. total_rewards: {total_rewards}")
            break
        
    if e%print_interval==0 and e !=0:
        print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(e, score/print_interval, pi.log_alpha.exp()))
        writer.add_scalar("avg score", score/print_interval, int(e/print_interval))
        score = 0.0
        
    torch.save(pi.state_dict(), PI_WEIGHT_PATH)
    torch.save(q1.state_dict(), Q_WEIGHT_PATH)
    
    num_successful_trajectories += int(success_curr_time_step)
    writer.add_scalar("num_success", num_successful_trajectories, e)

print(num_successful_trajectories)
