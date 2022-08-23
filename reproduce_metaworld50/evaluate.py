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

env_list = []

def env_creator(env_config):
    env_name = env_config["env"]
    SEED = env_config["seed"]
    ml1 = metaworld.ML1(env_name, seed=SEED)
    env = ml1.train_classes[env_name]()
    env.seed(SEED)
    random.seed(SEED)
    task = random.choice(ml1.train_tasks)
    env.set_task(task) 
    env_list.append(env)
    return env

env_name = "door-close-v2"
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
        num_cpus_per_worker=4,
                )\
    .framework(
        framework='torch'
    )\
    .environment(
        env=env_name,
        env_config = {"env": env_name, "seed": 1},
        render_env=True,
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
        evaluation_num_workers=1
    )

checkpoint = "/home/slowlab/ray_results/PPOTrainer_door-close-v2_2022-08-22_02-21-02grr4_45u/checkpoint_000301.tune_metadata"
trainer = PPOTrainer(env=env_name, config=config)
trainer.restore(checkpoint)
for epoch in range(10):
    result = trainer.evaluate()
    print(pretty_print(result))