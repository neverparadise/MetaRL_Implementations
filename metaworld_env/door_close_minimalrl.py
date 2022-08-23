from collections import namedtuple
import gym
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

#Hyperparameters
learning_rate  = 6e-5
gamma           = 0.99
lmbda           = 0.95
eps_clip        = 0.2
K_epoch         = 5
rollout_len    = 3
buffer_size    = 30
minibatch_size = 64
seed = 3145

class PPO(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(obs_dim,128)
        self.fc_mu = nn.Linear(128,num_actions)
        self.fc_std  = nn.Linear(128,num_actions)
        self.fc_v = nn.Linear(128,num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append(prob_a)
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
            
            s = torch.tensor(s_batch, dtype=torch.float)
            a = torch.tensor(a_batch)
            r = torch.tensor(r_batch, dtype=torch.float)
            ns = torch.tensor(s_prime_batch, dtype=torch.float)
            d = torch.tensor(done_batch, dtype=torch.float)
            prob_a = torch.tensor(prob_a_batch, dtype=torch.float)
            mini_batch = s, a, r, ns, d, prob_a
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    entropy = dist.entropy().mean()

                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target) - entropy * 0.02

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1



def main():
    ml1 = metaworld.ML1('door-close-v2', seed=seed) # Construct the benchmark, sampling tasks
    env = ml1.train_classes['door-close-v2']()  # Create an environment with task `pick_place`
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    print(env.observation_space)
    print(env.action_space)
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task
    max_length = env.max_path_length
    success_rate = 0

    model = PPO(obs_dim, num_actions)
    score = 0.0
    print_interval = 100
    rollout = []

    #WEIGHT_PATH = "/home/kukjin/Projects/MetaRL/MetaRL_Implementations/metaworld_env/weights/PPO_Pendulum.pt"
    WEIGHT_PATH = "/home/slowlab/Desktop/MetaRL/MetaRL_Implementations/metaworld_env/weights/PPO_door_close.pt"
    writer = SummaryWriter()

    total_episodes = 1000000
    score = 0
    for e in range(total_episodes):
        s = env.reset()
        done = False
        total_rewards = 0
        step = 0
        num_successful_trajectories = 0
        success_curr_time_step = 0.0

        while not done and step < max_length:
            for t in range(rollout_len):
                # env.render()
                with torch.no_grad():
                    mu, std = model.pi(torch.from_numpy(s).float())
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)
                    a = a.cpu().numpy()
                    log_prob = log_prob.cpu().numpy()
                s_prime, r, done, info = env.step(a)
                step += 1

                rollout.append((s, a, r/10.0, s_prime, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                total_rewards += r
                success_curr_time_step += info['success']
                if done or step == max_length:
                    writer.add_scalar("total_rewards", total_rewards, e)
                    score += total_rewards
                    break

            model.train_net()
        num_successful_trajectories += int(success_curr_time_step)
        success_rate += num_successful_trajectories

        writer.add_scalar("num_success", num_successful_trajectories, e)
        writer.add_scalar("success_rate", success_rate/total_episodes, e)

        if e % print_interval == 0 and e > 0:
            score /= print_interval
            writer.add_scalar("avg_score", score, e)
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(e, score, model.optimization_step))
            print(info)
            print(info['success'])
            print(success_rate/total_episodes)
            score = 0.0

        torch.save(model.state_dict(), WEIGHT_PATH)
main()