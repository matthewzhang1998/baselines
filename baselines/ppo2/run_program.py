#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:19:22 2018

@author: matthewszhang
"""
#!/usr/bin/env python3  
from baselines.program.cmd_util import program_arg_parser
from baselines import bench, logger

def train(env_id, num_timesteps, seed, hier, cur, vis, model):
    from baselines.common import set_global_seeds
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import HierPolicy, MlpPolicy
    import gym
    import gym_program
    import tensorflow as tf
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    
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

    if hier: policy = HierPolicy
    else: policy = MlpPolicy
    
    ppo2.learn(policy=policy, env=env, hier=hier, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=1e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    args = program_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, hier=args.hier,
          cur=args.cur, vis=args.vis, model=args.model)

if __name__ == '__main__':
    main()
