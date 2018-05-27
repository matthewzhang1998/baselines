#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:40:49 2018

@author: matthewszhang
"""
from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.program.cmd_util import make_program_env
from baselines.program import program_wrapper


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='NumSort-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--hier', type=bool, default=False)
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_program_env(args.env, args.hier)
    #env = program_wrapper(env)
    if args.hier: model = deepq.models.hier_mlp([64])
    else: model = deepq.models.mlp([64])

    deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )

    env.close()


if __name__ == '__main__':
    main()
