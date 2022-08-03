import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np

def _format(state, device):
    x = state
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(2, 0, 1)
        #x = x.view(-1)
        x = x.unsqueeze(0)
        x = x.to(device=device)
    else:
        x = x.to(device=device)
    return x

class PPO(nn.Module):
    def __init__(self, num_actions, device):
        super(PPO, self).__init__()
        
        self.device = device
        self.num_actions = num_actions
        self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten()
        )
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_size = conv2d_size_out(64, 8, 4)
        conv_size = conv2d_size_out(conv_size, 4, 2)
        conv_size = conv2d_size_out(conv_size, 3, 1)
        linear_input_size = conv_size * conv_size * 64 # 4 x 4 x 64 = 1024
        self.fc = nn.Linear(linear_input_size, 512)
        self.fc_pi = nn.Linear(512, self.num_actions)
        self.fc_v = nn.Linear(512, 1)

    def forward(self, obs, softmax_dim=1):
        # make_batch 코드에서 obs를 잘 만들어야 한다. 
        # pixels (Batch, C, H, W)
        # angles (Batch, angle)
        pixels = obs
        conv_feature = self.conv_layers(pixels) # (Batch, Linear_size)
        feature = F.relu(self.fc(conv_feature))
        prob = self.fc_pi(feature)
        prob = F.softmax(prob, dim=softmax_dim)
        value = self.fc_v(feature)
        return prob, value

class TrajectoriesBuffer:
    def __init__(self, K=5):
        self.trajectories = deque(maxlen=K)

    def put_trajectory(self, item):
        self.trajectories.append(item)

    def get_trajectories(self):
        return self.trajectories

    def reset(self):
        self.trajectories.clear()
    

class Trajectory:
    def __init__(self, max_len=200):
        self.data = []

    def put_data(self, item):
        self.data.append(item)

    def get_data(self):
        return self.data

    def reset(self):
        self.data = []
    
    def __len__(self):
        return len(self.data)

def make_6action(env, action_index):
    #action = np.array([0.2, -0.5])        # action[0] left(-) right(+) # action[1] forward(+) backward(-)
    action = env.action_space.noop()

    if action_index == 0: # no action
        action = np.array([0.0, 0.0])
    elif action_index == 1: # turn left
        action = np.array([-0.25, 0.0])
    elif action_index == 2: # turn right
        action = np.array([0.25, 0.0])
    elif action_index == 3: # forward
        action = np.array([0.0, 0.25])
    elif action_index == 4: # backward
        action = np.array([0.0, -0.25])

class Buffer:
    def __init__(self, T_horizon):
        self.T_horizon = T_horizon
        self.data = deque(maxlen=T_horizon)

    def put_data(self, transition):
        # obs : pixels, angle
        # trans (obs, a, r, next_obs, prob[a].item(), done)
        self.data.append(transition)
        
    def make_batch(self):
        pixels_lst, angles_lst = [], []
        a_lst, r_lst, prob_a_lst, done_lst = [], [], [], []
        n_pixels_lst, n_angles_lst = [], []

        for transition in self.data:
            obs, a, r, next_obs, prob_a, done = transition
            pixels_lst.append(obs)
            a_lst.append([a])
            r_lst.append([r])
            n_pixels_lst.append(next_obs)
            prob_a_lst.append([prob_a])
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        pixels = torch.cat(pixels_lst).to(device)
        a = torch.tensor(a_lst, dtype=torch.int64).to(device)
        r = torch.tensor(r_lst, dtype=torch.float).to(device)
        n_pixels = torch.cat(n_pixels_lst).to(device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        prob_a = torch.tensor(prob_a_lst).to(device)
        self.data = deque(maxlen=self.T_horizon)
        obs = pixels
        next_obs = n_pixels
        return obs, a, r, next_obs, done_mask, prob_a