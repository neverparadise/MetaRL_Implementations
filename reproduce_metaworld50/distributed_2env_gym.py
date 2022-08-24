
import gym
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.tune.logger import pretty_print

ray.init(num_cpus=12, num_gpus=2)

env_names = ['CartPole-v0', 'MountainCar-v0', "Taxi-v3", 'Humanoid-v2']
 #           'LunarLander-v2','FrozenLake-v0', 'HandManipulateBlock-v0']

def env_creator(env_config):
    env_name = env_config["env"]
    SEED = env_config["seed"]
    env = gym.make(env_name)
    env.seed(SEED)
    return env

for env_name in env_names:
    register_env(env_name, env_creator)

@ray.remote(num_cpus=3, num_gpus=0.5)
def distributed_trainer(env_name):
    config = PPOConfig()
    config.training(
            gamma=0.99,
            lr=0.0005,
            train_batch_size=128,
            model={
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "tanh",
                    },
            use_gae=True,
            lambda_=0.95,
            vf_loss_coeff=0.2, 
            entropy_coeff=0.001,
            num_sgd_iter=5,
            sgd_minibatch_size=32,
            shuffle_sequences=True,
            )\
        .resources(
            num_gpus=0.5,
            num_cpus_per_worker=1,
                    )\
        .framework(
            framework='torch'
        )\
        .environment(
            env=env_name,
            render_env=True,
            env_config = {"env": env_name, "seed": 1}
        )\
        .rollouts(
            num_rollout_workers=3,
            num_envs_per_worker=3,
            create_env_on_local_worker=False,
            rollout_fragment_length=16,
            horizon=500,
            soft_horizon=False,
            no_done_at_end=False,
        )\
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=100,
            evaluation_duration_unit='auto',
            evaluation_num_workers=3,
            evaluation_parallel_to_training=True
            #evaluation_config=,
            #custom_evaluation_function=,
        )
    print(env_name)
    trainer = PPOTrainer(env=env_name, config=config)
    for epoch in range(10):
        result = trainer.train()
        #print(pretty_print(result))
        print(f"env: {env_name}, epoch: {epoch}")
        if epoch % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
    
    return env_name

result_envs = [distributed_trainer.remote(env_name) for env_name in env_names]
ray.get(result_envs)
# while len(result_envs):
#     done_envs, result_envs = ray.wait(result_envs)
#     print(result_envs)