import gym
import metagym.metamaze
import time
import random
import numpy as np
maze_env = gym.make("meta-maze-3D-v0", enable_render=True) # Running a 3D Maze
print(maze_env.observation_space)
print(maze_env.action_space)

#np.random.seed(1)
#random.seed(1)

#maze_env = gym.make("meta-maze-2D-v0", enable_render=True) # Running a 2D Maze
task1 = maze_env.sample_task(
    cell_scale=7,  # Number of cells = cell_scale * cell_scale
    allow_loops=False,  # Whether loops are allowed
    crowd_ratio=0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
    cell_size=2.0, # specifying the size of each cell, only valid for 3D mazes
    wall_height=3.2, # specifying the height of the wall, only valid for 3D mazes
    agent_height=1.6, # specifying the height of the agent, only valid for 3D mazes
    view_grid=3, # specifiying the observation region for the agent, only valid for 2D mazes
    step_reward=-0.01, # specifying punishment of each step
    goal_reward=1.0 # specifying reward of reaching the goal
    )

task2 = maze_env.sample_task(
    cell_scale=7,  # Number of cells = cell_scale * cell_scale
    allow_loops=False,  # Whether loops are allowed
    crowd_ratio=0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
    cell_size=2.0, # specifying the size of each cell, only valid for 3D mazes
    wall_height=3.2, # specifying the height of the wall, only valid for 3D mazes
    agent_height=1.6, # specifying the height of the agent, only valid for 3D mazes
    view_grid=3, # specifiying the observation region for the agent, only valid for 2D mazes
    step_reward=-0.01, # specifying punishment of each step
    goal_reward=1.0 # specifying reward of reaching the goal
    )

task3 = maze_env.sample_task(
    cell_scale=7,  # Number of cells = cell_scale * cell_scale
    allow_loops=False,  # Whether loops are allowed
    crowd_ratio=0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
    cell_size=2.0, # specifying the size of each cell, only valid for 3D mazes
    wall_height=3.2, # specifying the height of the wall, only valid for 3D mazes
    agent_height=1.6, # specifying the height of the agent, only valid for 3D mazes
    view_grid=3, # specifiying the observation region for the agent, only valid for 2D mazes
    step_reward=-0.01, # specifying punishment of each step
    goal_reward=1.0 # specifying reward of reaching the goal
    )

task_list = [task1, task2, task3]

for e in range(100):
    task = random.choice(task_list)
    maze_env.set_task(task1)
    obs = maze_env.reset()
    print(obs)
    print(obs.shape)

    done = False
    step = 0
    while not done:
        #  The action space is discrete actions specifying UP/DOWN/LEFT/RIGHT
        action = maze_env.action_space.sample() 
        #action = np.array([0.2, -0.5])        # action[0] left(-) right(+) # action[1] forward(+) backward(-)
        
        step += 1
        #  The observation being 3 * 3 numpy array, the observation of its current neighbours
        #  Reward is set to be 20 when arriving at the goal, -0.1 for each step taken
        #  Done = True when reaching the goal or maximum steps (200 as default)
        observation, reward, done, info = maze_env.step(action)
        time.sleep(0.02)
        maze_env.render()
        if done:
            print(f"episode: {e}, step: {step}")
            break

