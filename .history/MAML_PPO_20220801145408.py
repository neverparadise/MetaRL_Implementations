import gym
import metagym.metamaze
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os

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

def update_policy(trajectory, optimizer, gamma):
    accum_R = 0  # 누적 reward를 계산하는 변수    
    loss = 0   
    trajectory_data = trajectory.get_data()
    traj_len = len(trajectory) - 1 # policy network을 update할 때, 앞에 gamma^t 부분을 계산하기 위한 변수입니다. 아래 gamma_powers 변수 참고.

    for idx, [log_p, r] in enumerate(trajectory_data[::-1]):
        accum_R = r + (gamma * accum_R)
        gamma_pow_t = gamma**(traj_len-idx) # gamma^t
        one_step_loss = -log_p * gamma_pow_t * accum_R  # sutton REINFORCE pseudo code. Chapter.13.3
        loss += one_step_loss
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()       
    trajectory.reset()
    return loss.item() 


def adaptation(K, env, task, traj_buffer, policy, adapted_policy, temp_policy, inner_optimizer, gamma):
    sample_trajectories(K, env, task, traj_buffer, policy)
    trajectories = traj_buffer.get_trajectories()
    
    # weight store
    temp_policy.load_state_dict(policy.state_dict())
    accum_R = 0
    loss = 0
    for traj in trajectories:
        traj_data = traj.get_data()
        traj_len = len(traj_data) - 1 
        for idx, [log_p, r] in enumerate(traj_data[::-1]):
            accum_R = r + (gamma * accum_R)
            gamma_pow_t = gamma**(traj_len-idx) # gamma^t
            one_step_loss = -log_p * gamma_pow_t * accum_R  # sutton REINFORCE pseudo code. Chapter.13.3
            loss += one_step_loss
    inner_optimizer.zero_grad()  
    loss.backward() 
    inner_optimizer.step()   
    adapted_policy.load_state_dict(policy.state_dict())
    policy.load_state_dict(temp_policy.state_dict())
    traj_buffer.reset()
    sample_trajectories(K, env, task, traj_buffer, adapted_policy)


def meta_update(traj_buffer, adapted_policy, outer_optimizer, gamma):
    print("meta update phase...")
    trajectories = traj_buffer.get_trajectories()
    accum_R = 0
    loss = 0
    for traj in trajectories:
        traj_data = traj.get_data()
        traj_len = len(traj_data) - 1 
        for idx, [log_p, r] in enumerate(traj_data[::-1]):
            accum_R = r + (gamma * accum_R)
            gamma_pow_t = gamma**(traj_len-idx) # gamma^t
            one_step_loss = -log_p * gamma_pow_t * accum_R  # sutton REINFORCE pseudo code. Chapter.13.3
            loss += one_step_loss
    outer_optimizer.zero_grad()
    loss.backward() # ! question: 여기서 meta policy로 그래디언트가 전달되는가?
    outer_optimizer.step()
    return loss.item()

def sample_trajectories(K, env, task, traj_buffer, policy):
    env.set_task(task)
    traj = Trajectory(200)
    score = 0
    for i in range(K):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action, log_prob = policy.sample_action(obs)
            obs_prime, reward, done, info = env.step(action.item())
            traj.put_data((log_prob, reward))
            obs = obs_prime
            score += reward
            if done:
                traj_buffer.put_trajectory(traj)
                break
    print(f"K average scroe: {score / K}")
    

def meta_train(writer):
    maze_env = gym.make("meta-maze-3D-v0", enable_render=True, view_grid=2) # Running a 2D Maze

    # Require 1: distribution over tasks
    num_tasks = 1000
    task_set = [maze_env.sample_task(
        cell_scale=7,  
        allow_loops=False,  
        crowd_ratio=0.40,   
        cell_size=2.0, 
        wall_height=3.2,
        agent_height=1.6, 
        view_grid=2,
        step_reward=-0.01, 
        goal_reward=1.0
        ) for i in range(num_tasks)]
    maze_env.set_task(task_set[0])
    # Require 2: Hyperparameters
    alpha = 6.25 * 1e-5
    beta = 1e-4
    K=5
    gamma = 0.99
    batch_size = 8
    # 1. randomly initialize theta
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_shape = maze_env.observation_space.shape
    print(obs_shape)
    H = obs_shape[0]
    W = obs_shape[1]
    num_actions = maze_env.action_space.n
    policy = Policy(W * H, num_actions, device)
    policy = policy.to(device, dtype=torch.float)
    adapted_policy = Policy(W * H, num_actions, device)
    adapted_policy = adapted_policy.to(device, dtype=torch.float)
    adapted_policy.load_state_dict(policy.state_dict())
    # Temporal policy for weights store
    temp_policy = Policy(W * H, num_actions, device)
    temp_policy = temp_policy.to(device, dtype=torch.float)

    inner_optimizer = optim.Adam(policy.parameters(), lr=alpha)
    outer_optimizer = optim.Adam(policy.parameters(), lr=beta)
    traj_buffer = TrajectoriesBuffer(K=K*batch_size)

    # 2. Iteration
    for epoch in range(10000):
        # 3. Sample batch of tasks T_i ~ p(T)
        tasks = random.choices(task_set, k=batch_size)
        # 4 ~ 9. Inner adaptation
        for i, task in enumerate(tasks):
            print(f"adaptation phase. epoch:{epoch} task: {i}")
            adaptation(K, maze_env, task, traj_buffer, policy, adapted_policy, temp_policy, inner_optimizer, gamma)

        # 10. Meta update
        loss = meta_update(traj_buffer, adapted_policy, outer_optimizer, gamma)
        traj_buffer.reset()
        writer.add_scalar("loss", loss, epoch)

        print(f"epoch: {epoch}, loss: {loss}")
        torch.save(policy.state_dict(), PATH)


def meta_test():
    pass

if __name__ == "__main__":
    PATH = os.curdir + '/MetaRL_Implementations/weights/meta_policy.pt'
    summary_path = os.curdir + "/MetaRL_Implementations/experiments/"
    if not os.path.isdir(summary_path):
        os.mkdir(summary_path)
    writer = SummaryWriter(summary_path)
    meta_train(writer)
