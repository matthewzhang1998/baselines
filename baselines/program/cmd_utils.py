#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:36:09 2018

@author: matthewszhang

Command Line Helpers for Program Environment (refer to algorithmic@baselines)
"""
import os
import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_program_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for custom environment
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_program.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='program-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser