import gym
import metagym.metamaze
import time
import random
import numpy as np
from model.PPO import PPO, make_5action
import torch

def main():
    maze_env = gym.make("meta-maze-3D-v0", enable_render=True) # Running a 2D Maze
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
            obs = maze_env.reset()
            done = False
            step = 0
            total_rewards = 0
            while not done:
                maze_env.render()
                action = maze_env.action_space.sample() 
                print(action)
                #prob, action_index = policy.sample_action(obs)
                #action = make_5action(maze_env, action_index)
                next_obs, reward, done, info = maze_env.step(action)
                total_rewards += reward
                if done:
                    print(f"episode: {episode}, task: {i}, step: {step}, total_rewards: {total_rewards}")
                    break

main()
