
from collections import namedtuple
import metaworld
import random
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.tune.logger import pretty_print
from custom_metric_callback import MyCallbacks

import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

# env_cls_dict = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
# for k, v in env_cls_dict.items():
#     print(k, v)

# env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['push-v2-goal-observable']
# eval_env= env_cls(seed=0)
# eval_env.seed(0)


env_names = ['door-close-v2-goal-observable', 'door-open-v2-goal-observable',
             'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable',
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

num_gpus = 8
num_envs = len(env_names)
gpu_fractions = num_gpus / num_envs

@ray.remote(num_cpus=cpu_fractions, num_gpus=gpu_fractions)
def distributed_trainer(env_name):
    config = PPOConfig()
    config.training(
            gamma=0.99,
            lr=0.0005,
            train_batch_size=500,
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
            num_gpus=1,
            num_cpus_per_worker=1,
                    )\
        .framework(
            framework='torch'
        )\
        .environment(
            env=env_name,
            env_config = {"env": env_name, "seed": 1}
        )\
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=1,
            create_env_on_local_worker=False,
            rollout_fragment_length=250,
            horizon=500,
            soft_horizon=False,
            no_done_at_end=False,
            ignore_worker_failures=True,
            recreate_failed_workers=True,
            restart_failed_sub_environments=True,
        )\
        .callbacks(MyCallbacks)
        # .evaluation(
        #     evaluation_interval=10,
        #     evaluation_duration=100,
        #     evaluation_duration_unit='auto',
        #     evaluation_num_workers=3,
        #     evaluation_parallel_to_training=True
        #     #evaluation_config=,
        #     #custom_evaluation_function=,
        # )
    trainer = PPOTrainer(env=env_name, config=config)
    print(f"env_name: {env_name}")
    print(f"fractions: {cpu_fractions}, {gpu_fractions}")
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    # model = trainer.get_policy().model
    for epoch in range(4000):
        result = trainer.train()
        custom_metrics = result["custom_metrics"]
        print(f"env_name: {env_name}, epoch: {epoch}, \n custom_metrics: {custom_metrics}")
        # print(pretty_print(result))
        if epoch % 20 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
    
    return 0

distributed_trainier_refs = [distributed_trainer.remote(env_name) for env_name in env_names]
results = ray.get(distributed_trainier_refs)