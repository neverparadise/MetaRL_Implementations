import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig

import random
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

env_names = ['door-close-v2-goal-observable', 'door-open-v2-goal-observable']

def env_creator(env_config):
    env_name = env_config["env"]
    SEED = env_config["seed"]
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = env_cls(seed=SEED)
    env.seed(SEED)
    random.seed(SEED)
    return env

for env_name in env_names:
    register_env(env_name, env_creator)

def make_trainer(env_name):
    config = PPOConfig()
    config.training(
            gamma=0.99,
            lr=0.0005,
            train_batch_size=1000,
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
            num_gpus=0,
            num_cpus_per_worker=1,
                    )\
        .framework(
            framework='tf'
        )\
        .environment(
            env=env_name,
            render_env=True,
            env_config = {"env": env_name, "seed": 1}
        )\
        .rollouts(
            num_rollout_workers=1,
            num_envs_per_worker=1,
            create_env_on_local_worker=False,
            rollout_fragment_length=250,
            horizon=500,
            soft_horizon=False,
            no_done_at_end=False,
        )\
        .evaluation(
            evaluation_interval=10,
            #evaluation_duration=100,
            evaluation_duration_unit='auto',
            evaluation_num_workers=1,
            evaluation_parallel_to_training=True
            #evaluation_config=,
            #custom_evaluation_function=,
        )
    print(env_name)
    trainer = PPOTrainer(env=env_name, config=config)
    return trainer

ray.init()
trainer = make_trainer(env_names[0])
trainers = [make_trainer(env_name) for env_name in env_names]

@ray.remote
def distributed_train2(algo):
    result = algo.train()
    return result

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   #result = algo.train()
   result_refs = [distributed_train2.remote(trainer) for trainer in trainers]
   result = ray.get(result_refs)
   print(pretty_print(result))
   trainer.evaluate()
   if i % 10 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)