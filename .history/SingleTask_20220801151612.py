import gym
import metagym.metamaze
import time
import random
import numpy as np
from model.PPO import *



def main():
    maze_env = gym.make("meta-maze-3D-v0", enable_render=True, view_grid=2) # Running a 2D Maze
    maze_env.seed(3145)
    # Require 1: distribution over tasks
    num_tasks = 2
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_actions = 5
    policy = PPO(num_actions, device)
    policy = policy.to(device, dtype=torch.float)

    K = 5
    for episode in range(1000):
        tasks = random.choices(task_set, k=1)
        for i, task in enumerate(tasks):
            maze_env.set_task(task)
