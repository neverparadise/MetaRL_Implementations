import gym
import metagym.metamaze
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
import sys, time, random, re, requests
import concurrent.futures
from multiprocessing import Process, Queue, Pool, cpu_count, current_process, Manager
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

channel = logging.StreamHandler()
channel.setLevel(logging.DEBUG)
channel.setFormatter(formatter)
logger.addHandler(channel)


def main(producer_task, consumer_task):
    data_queue = Queue()
    number_of_cpus = cpu_count()
    manager = Manager()
    fibo_dict = manager.dict()

    producer = Process(target=producer_task, args=(data_queue,fibo_dict))
    producer.start()
    producer.join()

    consumer_lst = []
    for i in range(number_of_cpus):
        consumer = Process(target=consumer_task, args=(data_queue,fibo_dict))
        consumer.start()
        consumer_lst.append(consumer)

    [consumer.join() for consumer in consumer_lst]
    logger.info(fibo_dict)

def producer_task(queue, fibo_dict):
    for i in range(15):
        value = random.randint(1, 20)
        fibo_dict[value] = None
        logger.info(f"Producer {current_process().name} putting value {value} into queue... ")
        queue.put(value)

def consumer_task(queue, fibo_dict):
    while not queue.empty():
        value = queue.get(True, 0.05)
        a, b = 0, 1
        for item in range(value):
            a, b, = b, a + b
            fibo_dict[value] = a
        logger.info(f"consumer {current_process().name} getting vlaue {value} from queue...")


if __name__ == "__main__":
    main(producer_task, consumer_task)