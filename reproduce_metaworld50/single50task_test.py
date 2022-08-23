from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


env_cls_dict = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
for k, v in env_cls_dict.items():
    print(k, v)

env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['push-v2-goal-observable']
eval_env= env_cls(seed=0)
eval_env.seed(0)
num_evals = 1000
num_successful_eval_trajectories = 0

for _ in range(num_evals):
    avg_reward = 0 
    obs = eval_env.reset()
    done = False
    success_curr_time_step = False
    stp = 0
    while not done and stp < eval_env.max_path_length:
        #eval_env.render()
        obs, reward, done, info = eval_env.step(eval_env.action_space.sample())
        stp += 1
        avg_reward += reward
        success_curr_time_step |= bool(info['success'])
    num_successful_eval_trajectories += int(success_curr_time_step)
    print(f"episode: {_}, avg_reward: {avg_reward/500}")

print(num_successful_eval_trajectories/num_evals)