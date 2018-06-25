#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:42:33 2018

@author: matthewszhang
"""
import os
import tensorflow as tf
from baselines.program.cmd_util import feudal_arg_parser
from baselines.feudal import policies
from baselines import bench, logger

def train(env_id, 
          seed, 
          tsteps,
          nsteps,
          stoch,
          cur,
          vis,
          model,
          encoef,
          lr,
          cliphigh,
          clipinc,
          vcoef,
          mgn,
          gmax,
          ginc,
          lam,
          nhier,
          nmb,
          noe,
          ngmin,
          nginc,
          bmin,
          bmax,
          nhist,
          recurrent,
          pol,
          val):
    from baselines.common import set_global_seeds
    from baselines.feudal.feudal import learn
    import gym
    import gym_program
    import tensorflow as tf
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.feudal.policies import CnnPolicy, MlpPolicy, NullPolicy
    from baselines.exploration.vime import vime
    from baselines.exploration import null
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    
    curiosity = {'vime':vime.BayesianInferenceNetwork,
                'null':null.NullNetwork}[cur]
    def make_env():
        set_global_seeds(seed)
        env = gym.make(env_id)
        env.set_path(logger.get_dir())
        env.set_curiosity(curiosity, model)
        env.set_hier(False)
        env.set_visualize(vis)
        env.set_stoch(stoch)
        env = bench.Monitor(env, logger.get_dir())
        env.seed(seed)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)

    policy = {'mlp':MlpPolicy, 'null':NullPolicy, 'cnn':CnnPolicy}[pol]
    learn(env=env,
          tsteps=tsteps,
          nsteps=nsteps,
          encoef=encoef,
          lr=lr,
          cliphigh=cliphigh,
          clipinc=clipinc,
          vcoef=vcoef,
          mgn=mgn,
          gmax=gmax,
          ginc=ginc,
          lam=lam,
          nhier=nhier,
          nmb=nmb,
          noe=noe,
          ngmin=ngmin,
          nginc=nginc,
          bmin=bmin,
          bmax=bmax,
          nhist=nhist,
          recurrent=recurrent,
          policy=policy,
          #curiosity=curiosity,
          val=val)

def main():
    args = feudal_arg_parser().parse_args()
    if args.quiet:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    dir=logger.configure()
    tolog = vars(args)
    logger.logvar(tolog, dir=dir)
    train(env_id=args.env, 
          seed=args.seed, 
          tsteps=args.tsteps,
          nsteps=args.nsteps,
          encoef=args.encoef,
          lr=args.lr,
          cliphigh=args.cliphigh,
          clipinc=args.clipinc,
          vcoef=args.vcoef,
          mgn=args.mgn,
          gmax=args.gmax,
          ginc=args.ginc,
          lam=args.lam,
          nhier=args.nhier,
          nmb=args.nmb,
          noe=args.noe,
          ngmin=args.ngmin,
          nginc=args.nginc,
          bmin=args.bmin,
          bmax=args.bmax,
          nhist=args.nhist,
          recurrent=args.recurrent,
          pol=args.pol,
          cur=args.cur,
          vis=args.vis,
          model=args.model,
          val=args.val,
          stoch=args.stoch)
            
if __name__ == '__main__':
    main()