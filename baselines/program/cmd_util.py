#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:36:09 2018

@author: matthewszhang

Command Line Helpers for Program Environment (refer to algorithmic@baselines)
"""
import os
import gym
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds

def make_program_env(env_id, seed, hier=True, curiosity=True, visualize=True, model='LSTM'):
    """
    Create a wrapped, monitored gym.Env for custom environment
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.set_curiosity(curiosity, model)
    env.set_hier(hier)
    env.set_visualize(visualize)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def program_arg_parser():
    """
    Create an argparse.ArgumentParser for run_program.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='NumSwap-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('--recurrent', help='recurrency', type=str, default=False)
    parser.add_argument('--pol', help='type of policy', type=str, default='hier2')
    parser.add_argument('--cur', help='curiosity model', type=bool, default=False)
    parser.add_argument('--vis', help='visualizations', type=bool, default=False)
    parser.add_argument('--model', help='encoding model', type=str, default='mlp')
    return parser