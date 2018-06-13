#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:53:53 2018

@author: matthewszhang
"""

#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.program.cmd_util import program_arg_parser, make_program_env
from baselines import logger
from baselines.ppo1.mlp_policy import HierPolicy
from baselines.trpo_mpi import trpo_mpi

def train(env_id, num_timesteps, seed, hier):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return HierPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    env = make_program_env(env_id, workerseed, hier=hier)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, hier=hier)
    env.close()

def main():
    args = program_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, hier=args.hier)


if __name__ == '__main__':
    main()
