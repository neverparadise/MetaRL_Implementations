import metaworld
import random

SEED = 3145  # some seed number here
#benchmark = metaworld.Benchmark(seed=SEED)
print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('door-open-v2', seed=SEED) # Construct the benchmark, sampling tasks

env = ml1.train_classes['door-open-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

for e in range(100):
    obs = env.reset()  # Reset environment
    print(obs)
    done = False
    score = 0
    timestep = 0
    while timestep <= 501:
        env.render()
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

        timestep += 1
        score += reward
        if timestep > 501:
            print(score)
            break