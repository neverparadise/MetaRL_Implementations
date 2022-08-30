
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
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", help="environment name", type=str)
args = parser.parse_args()

env_names = ['door-close-v2-goal-observable', 'door-open-v2-goal-observable',
             'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable',
             'drawer-close-v2-goal-observable', 'drawer-open-v2-goal-observable',
             'push-back-v2-goal-observable', 'push-v2-goal-observable', ]

env_name = args.env_name

#assert env_name in env_names,  'No environment in list'
    
def env_creator(env_config):
    env_name = env_config["env"]
    SEED = env_config["seed"]
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = env_cls(seed=SEED)
    env.seed(SEED)
    random.seed(SEED)
    return env

register_env(env_name, env_creator)

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
        num_gpus=1,
        num_cpus_per_worker=3,
                )\
    .framework(
        framework='torch'
    )\
    .environment(
        env=env_name,
        env_config = {"env": env_name, "seed": 1},
        remote_worker_envs=True
    )\
    .rollouts(
        num_rollout_workers=3,
        num_envs_per_worker=4,
        create_env_on_local_worker=False,
        rollout_fragment_length=416,
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
    
trainer = PPOTrainer(env=env_name, config=config)
print(env_name)
for epoch in range(10):
    result = trainer.train()
    print(result)
    if epoch % 2 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
