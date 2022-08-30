
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

# env_cls_dict = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
# for k, v in env_cls_dict.items():
#     print(k, v)

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
            num_gpus=1,
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
            evaluation_duration=100,
            evaluation_duration_unit='auto',
            evaluation_num_workers=1,
            evaluation_parallel_to_training=True
            #evaluation_config=,
            #custom_evaluation_function=,
        )
    print(env_name)
    trainer = PPOTrainer(env=env_name, config=config)
    return trainer

@ray.remote(num_gpus=1)
def distributed_trainer(env_name):
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
            num_cpus_per_worker=2,
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
            evaluation_duration=100,
            evaluation_duration_unit='auto',
            evaluation_num_workers=1,
            evaluation_parallel_to_training=True
            #evaluation_config=,
            #custom_evaluation_function=,
        )
    print(env_name)
    trainer = PPOTrainer(env=env_name, config=config)
    return trainer

@ray.remote
def distributed_train(trainer_ref):
    trainer = ray.get(trainer_ref)
    result = trainer.train()
    return result

@ray.remote(num_cpus=4, num_gpus=1)
def distributed_train2(trainer):
    result = trainer.train()
    return result

results = []

trainers = [make_trainer(env_name) for env_name in env_names]
#trainers_ref = [ray.put(trainer) for trainer in trainers]

for epoch in range(100):
    result_refs = [distributed_train2.remote(trainer) for trainer in trainers]
    results = ray.get_results(result_refs)
    print(pretty_print(results))
    if epoch % 10 == 0:
        checkpoints = [trainer.save() for trainer in trainers]
    #    checkpoints = [trainer.save() for trainer in distributed_trainier_refs]

    #while len(results) > 0:
    #    results, data_references = ray.wait(data_references, num_returns=2, timeout=7.0)
    #    print(pretty_print(results))
    #print(pretty_print(ray.get(results)))
