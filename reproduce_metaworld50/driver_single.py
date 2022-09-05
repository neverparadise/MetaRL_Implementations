
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

# device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device3 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device4 = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device5 = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# device6 = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device7 = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# env_names = metaworld.ML1.ENV_NAMES
# print(env_names)
# print(len(env_names))
env_list = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2',
             'button-press-topdown-v2', 'button-press-topdown-wall-v2',
             'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
             'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2',
             'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2',
             'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2',
             'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2',
             'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2',
             'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2',
             'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2',
             'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2',
             'plate-slide-side-v2', 'plate-slide-back-v2',
             'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2',
             'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2',
             'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2',
             'window-close-v2']

env_names = ['door-close-v2', 'door-open-v2',
             'button-press-topdown-v2', 'button-press-topdown-wall-v2',
             'drawer-close-v2', 'drawer-open-v2',
             'push-back-v2', 'push-v2', ]

env_names = ['door-close-v2']

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

for env_name in env_names:
    register_env(env_name, env_creator)

for env_name in env_names:
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
            num_cpus_per_worker=8,
                    )\
        .framework(
            framework='torch'
        )\
        .environment(
            env=env_name,
            env_config = {"env": env_name, "seed": 1},
            render_env=True
        )\
        .rollouts(
            num_rollout_workers=1,
            num_envs_per_worker=1,
            create_env_on_local_worker=False,
            rollout_fragment_length=1000,
            horizon=500,
            soft_horizon=False,
            no_done_at_end=False,
        )\
        .evaluation(
            evaluation_interval=1,
            evaluation_num_workers=1,
        )

    # trainer = config.build(env=env_name)
    trainer = PPOTrainer(env=env_name, config=config)
    # model = trainer.get_policy().model
    for epoch in range(1000):
        result = trainer.train()
        print(pretty_print(result))
        if epoch % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    # model_config = {
    #                 "fcnet_hiddens": [128, 128],
    #                 "fcnet_activation": "tanh",
    #                 }
    # env_config = {"env": env_name, "seed": 1}
    # config = {
    #             "num_workers": 1, 
    #             "model": model_config,
    #             "env_config": env_config,
    #             "framework": "torch",
    #             "num_envs_per_worker": 1,
    #             "horizon": 500,
    #             "train_batch_size": 5000,
    #             "rollout_fragment_length": 500,
    #             "no_done_at_end": True,
    #             "gamma": 0.99,
    #             "lr": 0.0005,
    #             "train_batch_size": 200,
    #             "seed": 1,
    #             }
