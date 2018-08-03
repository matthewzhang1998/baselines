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

def feudal_arg_parser():
    parser = arg_parser()
    # simulation parameters
    parser.add_argument('--env', help='environment ID', type=str, default='NumSwap-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--tsteps', help='total simulation size', type=int, default=int(1e7))
    parser.add_argument('--nsteps', help='batch size', type=int, default=int(1e4))
    
    # feudal parameters
    parser.add_argument('--encoef', help='entropy coefficient', type=float, default=0.0)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--cliphigh', help='cliprange max', type=float, default=0.2)
    parser.add_argument('--clipinc', help='cliprange hierarchical divisor', type=float, default=1)
    parser.add_argument('--vcoef', help='value coefficient', type=float, default=0.5)
    parser.add_argument('--enfnx', help='entropy decay {osc, exp, cst}', type=str, default='cst')
    parser.add_argument('--mgn', help='max grad norm', type=float, default=0.5)
    parser.add_argument('--gmax', help='gamma max', type=float, default=1e-2)
    parser.add_argument('--ginc', help='gamma hierarchical divisor', type=float, default=0.25)
    parser.add_argument('--lam', help='lambda', type=float, default=0.95)
    parser.add_argument('--nhier', help='number of hierarchies', type=int, default=2)
    parser.add_argument('--nmb', help='number of minibatches', type=int, default=4)
    parser.add_argument('--noe', help='number of opt. epochs per batch', type=int, default=10)
    parser.add_argument('--ngmin', help='min number of goals', type=int, default=8)
    parser.add_argument('--nginc', help='hierarchical multiplier', type=int, default=4)
    parser.add_argument('--bmin', help='beta min', type=float, default=0)
    parser.add_argument('--bmax', help='beta max', type=float, default=1)
    parser.add_argument('--nhist', help='nhist lookahead per hier', type=int, default=4)
    parser.add_argument('--stoch', help='stochasticity', type=float, default=0.2)
    parser.add_argument('--maxlen', help='max sim length', type=int, default=100)
    parser.add_argument('--cos', help='use cosine metric', type=int, default=0)
    parser.add_argument('--fm', help='fix manager', type=int, default=0)
    parser.add_argument('--fa', help='fix actor', type=int, default=0)
    #parser.add_argument('--lambda-cur', help='curiosity weighting', type=float, default=1e-3)
    
    # policy parameters
    parser.add_argument('--recurrent', help='recurrency', type=int, default=0)
    parser.add_argument('--pol', help='type of policy', type=str, default='null')
    parser.add_argument('--cur', help='curiosity model', type=str, default='null')
    parser.add_argument('--vis', help='visualizations', type=int, default=0)
    parser.add_argument('--model', help='encoding model', type=str, default='mlp')
    parser.add_argument('--log', help='writedir', type=str, default='test')
    parser.add_argument('--val', help='value network', type=int, default=0)
    parser.add_argument('--quiet', help='suppress tf', type=int, default=0)
    parser.add_argument('--ts', help='train_supervised', type=int, default=0)
    parser.add_argument('--ti', help='test iteration', type=int, default=1)
    
    parser.add_argument('--intermediate', help='intermediate goals', type=int, default=0)
    parser.add_argument('--nhidden', help='number of hidden units', type=int, default=64)
    return parser

    