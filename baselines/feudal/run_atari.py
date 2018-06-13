#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:16:40 2018

@author: matthewszhang
"""

#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.feudal.feudal import learn
from baselines.feudal.policies import CnnPolicy
import multiprocessing
import tensorflow as tf


def train(env_id, num_timesteps, seed, policy):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    policy = CnnPolicy
    learn(recurrent=False, policy=policy, env=env, nsteps=2048, nminibatches=32,
               noptepochs=10, log_interval=1,
               encoef=0,
               lr=1e-4,
               total_timesteps=num_timesteps)

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy)

if __name__ == '__main__':
    main()
