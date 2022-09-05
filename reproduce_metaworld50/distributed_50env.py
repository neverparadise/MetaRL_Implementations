
from collections import namedtuple
import metaworld
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
from custom_logger import custom_log_creator
import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

hidden_env_names = ['assembly-v2-goal-hidden', 'basketball-v2-goal-hidden', 'bin-picking-v2-goal-hidden', 'box-close-v2-goal-hidden',
                    'button-press-topdown-v2-goal-hidden', 'button-press-topdown-wall-v2-goal-hidden',
                    'button-press-v2-goal-hidden', 'button-press-wall-v2-goal-hidden', 'coffee-button-v2-goal-hidden',
                    'coffee-pull-v2-goal-hidden', 'coffee-push-v2-goal-hidden', 'dial-turn-v2-goal-hidden',
                    'disassemble-v2-goal-hidden', 'door-close-v2-goal-hidden', 'door-lock-v2-goal-hidden', 'door-open-v2-goal-hidden',
                    'door-unlock-v2-goal-hidden', 'hand-insert-v2-goal-hidden', 'drawer-close-v2-goal-hidden',
                    'drawer-open-v2-goal-hidden', 'faucet-open-v2-goal-hidden', 'faucet-close-v2-goal-hidden', 'hammer-v2-goal-hidden',
                    'handle-press-side-v2-goal-hidden', 'handle-press-v2-goal-hidden', 'handle-pull-side-v2-goal-hidden',
                    'handle-pull-v2-goal-hidden', 'lever-pull-v2-goal-hidden', 'peg-insert-side-v2-goal-hidden',
                    'pick-place-wall-v2-goal-hidden', 'pick-out-of-hole-v2-goal-hidden', 'reach-v2-goal-hidden',
                    'push-back-v2-goal-hidden', 'push-v2-goal-hidden', 'pick-place-v2-goal-hidden', 'plate-slide-v2-goal-hidden',
                    'plate-slide-side-v2-goal-hidden', 'plate-slide-back-v2-goal-hidden',
                    'plate-slide-back-side-v2-goal-hidden', 'peg-unplug-side-v2-goal-hidden', 'soccer-v2-goal-hidden',
                    'stick-push-v2-goal-hidden', 'stick-pull-v2-goal-hidden', 'push-wall-v2-goal-hidden', 'reach-wall-v2-goal-hidden',
                    'shelf-place-v2-goal-hidden', 'sweep-into-v2-goal-hidden', 'sweep-v2-goal-hidden', 'window-open-v2-goal-hidden',
                    'window-close-v2-goal-hidden']

observable_env_names = ['assembly-v2-goal-observable', 'basketball-v2-goal-observable', 'bin-picking-v2-goal-observable', 'box-close-v2-goal-observable',
                        'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable',
                        'button-press-v2-goal-observable', 'button-press-wall-v2-goal-observable', 'coffee-button-v2-goal-observable',
                        'coffee-pull-v2-goal-observable', 'coffee-push-v2-goal-observable', 'dial-turn-v2-goal-observable',
                        'disassemble-v2-goal-observable', 'door-close-v2-goal-observable', 'door-lock-v2-goal-observable', 'door-open-v2-goal-observable',
                        'door-unlock-v2-goal-observable', 'hand-insert-v2-goal-observable', 'drawer-close-v2-goal-observable',
                        'drawer-open-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable', 'hammer-v2-goal-observable',
                        'handle-press-side-v2-goal-observable', 'handle-press-v2-goal-observable', 'handle-pull-side-v2-goal-observable',
                        'handle-pull-v2-goal-observable', 'lever-pull-v2-goal-observable', 'peg-insert-side-v2-goal-observable',
                        'pick-place-wall-v2-goal-observable', 'pick-out-of-hole-v2-goal-observable', 'reach-v2-goal-observable',
                        'push-back-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable', 'plate-slide-v2-goal-observable',
                        'plate-slide-side-v2-goal-observable', 'plate-slide-back-v2-goal-observable',
                        'plate-slide-back-side-v2-goal-observable', 'peg-unplug-side-v2-goal-observable', 'soccer-v2-goal-observable',
                        'stick-push-v2-goal-observable', 'stick-pull-v2-goal-observable', 'push-wall-v2-goal-observable', 'reach-wall-v2-goal-observable',
                        'shelf-place-v2-goal-observable', 'sweep-into-v2-goal-observable', 'sweep-v2-goal-observable', 'window-open-v2-goal-observable',
                        'window-close-v2-goal-observable']

def env_creator_hidden(env_config):
    env_name = env_config["env"]
    SEED = env_config["seed"]
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name]
    env = env_cls(seed=SEED)
    env.seed(SEED)
    random.seed(SEED)
    return env

def env_creator_observable(env_config):
    env_name = env_config["env"]
    SEED = env_config["seed"]
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = env_cls(seed=SEED)
    env.seed(SEED)
    random.seed(SEED)
    return env

for env_name in hidden_env_names:
    register_env(env_name, env_creator_hidden)

for env_name in observable_env_names:
    register_env(env_name, env_creator_observable)


num_gpus = 8
num_envs = len(observable_env_names)
gpu_fractions = num_gpus / num_envs

@ray.remote(num_cpus=2, num_gpus=gpu_fractions)
def distributed_trainer(env_name):
    config = PPOConfig()
    config.training(
        gamma=0.99,
        lr=0.0005,
        train_batch_size=2000,
        model={
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "tanh",
        },
        use_gae=True,
        lambda_=0.95,
        vf_loss_coeff=0.2,
        entropy_coeff=0.001,
        num_sgd_iter=5,
        sgd_minibatch_size=64,
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
            env_config={"env": env_name, "seed": 1}
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
    #         .build(
    #     logger_creator=custom_log_creator(os.path.expanduser(
    #         "~/another_ray_results/subdir"), 'custom_dir')
    # )\

    trainer = PPOTrainer(env=env_name, config=config)
    print(f"env_name: {env_name}")
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(
        os.environ["CUDA_VISIBLE_DEVICES"]))

    # model = trainer.get_policy().model
    for epoch in range(10000):
        result = trainer.train()
        result.pop('info')
        result.pop('sampler_results')
        if epoch % 200 == 0:
            custom_metrics = result["custom_metrics"]
            print(
                f"env_name: {env_name}, epoch: {epoch}, \n custom_metrics: {custom_metrics}")
            print(pretty_print(result))
            checkpoint = trainer.save()

        # if epoch % 200 == 0:
            #print("checkpoint saved at", checkpoint)

    return 0

distributed_trainier_refs = [distributed_trainer.remote(env_name) for env_name in hidden_env_names]
results = ray.get(distributed_trainier_refs)

distributed_trainier_refs = [distributed_trainer.remote(env_name) for env_name in observable_env_names]
results = ray.get(distributed_trainier_refs)