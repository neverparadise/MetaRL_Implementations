import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from custom_metric_callback import MyCallbacks
import tensorboard
import random
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

import logging
import logging
 
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
 

#env_name = 'door-close-v2'
 #['door-close-v2-goal-observable', 'door-open-v2-goal-observable',
env_names = ['button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable',
             'drawer-close-v2-goal-observable', 'drawer-open-v2-goal-observable',
             'push-back-v2-goal-observable', 'push-v2-goal-observable', ]

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
            train_batch_size=2048,
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
            num_gpus=4,
            num_cpus_per_worker=4,
                    )\
        .framework(
            framework='torch'
        )\
        .environment(
            env=env_name,
            render_env=False,
            env_config = {"env": env_name, "seed": 0}
        )\
        .rollouts(
            num_rollout_workers=8,
            num_envs_per_worker=4,
            create_env_on_local_worker=False,
            rollout_fragment_length=64,
            horizon=500,
            ignore_worker_failures=True,
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
            soft_horizon=False,
            no_done_at_end=False,
        )\
        .callbacks(MyCallbacks)
        # .evaluation(
        #     evaluation_interval=10,
        #     #evaluation_duration=100,
        #     evaluation_duration_unit='auto',
        #     evaluation_num_workers=2,
        #     evaluation_parallel_to_training=True
        #     #evaluation_config=,
        #     #custom_evaluation_function=,
        # )\

    print(env_name)
    trainer = PPOTrainer(env=env_name, config=config)
    return trainer

ray.init()
for env_name in env_names:
    trainer = make_trainer(env_name)
    for i in range(4000):
        # Perform one iteration of training the policy with PPO
        try: 
            result = trainer.train()
        except:
            continue
        result.pop('info')
        result.pop('sampler_results')
        print(pretty_print(result))
        custom_metrics = result["custom_metrics"]
        print(custom_metrics)
        #trainer.evaluate()
        if i % 20 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

