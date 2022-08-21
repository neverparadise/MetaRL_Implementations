import ray
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

device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device3 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device4 = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
device5 = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
device6 = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
device7 = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")



env_names = metaworld.ML1.ENV_NAMES
print(env_names)
print(len(env_names))
env_list = []
def env_creater(env_name, SEED):
    random.seed(SEED)
    env.seed(SEED)
    ml1 = metaworld.ML1(env_name, seed=SEED)
    env = ml1.train_classes[env_name]()
    task = random.choice(ml1.train_tasks)
    env_list.append(env)
    return env
    

