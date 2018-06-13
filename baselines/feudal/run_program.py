#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:42:33 2018

@author: matthewszhang
"""
from baselines.program.cmd_util import program_arg_parser
from baselines.feudal import policies
from baselines import bench, logger

def train(env_id, num_timesteps, seed, recurrent, cur, vis, model, pol):
    from baselines.common import set_global_seeds
    from baselines.feudal.feudal import learn
    import gym
    import gym_program
    import tensorflow as tf
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.feudal.policies import CnnPolicy, MlpPolicy, NullPolicy
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    
    hier = False
    def make_env():
        set_global_seeds(seed)
        env = gym.make(env_id)
        env.set_curiosity(cur, model)
        env.set_hier(hier)
        env.set_visualize(vis)
        env = bench.Monitor(env, logger.get_dir())
        env.seed(seed)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)

    policy = {'mlp':MlpPolicy, 'null':NullPolicy, 'cnn':CnnPolicy}[pol]
    
    learn(recurrent=recurrent, policy=policy, env=env, nsteps=2048, nminibatches=32,
               noptepochs=10, log_interval=1,
               encoef=0,
               lr=1e-4,
               total_timesteps=num_timesteps)


def main():
    args = program_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, recurrent=args.recurrent,
          cur=args.cur, vis=args.vis, model=args.model, pol=args.pol)

if __name__ == '__main__':
    main()