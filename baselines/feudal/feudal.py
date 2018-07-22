    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:51:54 2018

@author: matthewszhang
"""
import time
import os
import os.path as osp
import numpy as np
from collections import deque
from baselines.feudal.models import FeudalModel, RecurrentFeudalModel, I2AModel
from baselines.feudal.runners import FeudalRunner, I2ARunner, TestRunner
from baselines.program.decode import decode_index, decode
from baselines.program.mlogger import Logger
from baselines.feudal.utils import sf01, fl01

PATH="tmp/build/graph"
    
def num(l):
    N_NUMS = len(l)
    val = 0
    for i in range(N_NUMS):
        val += l[i] * (N_NUMS ** (N_NUMS - 1 - i))
    return val

def sort_by_state(scalars, encoded_states, env):
    scalar_dict = {}
    for i in range(len(encoded_states)):
        for j in range(encoded_states[i].shape[0]):
            state = num(decode(encoded_states[i][j])['state'])
            if state in scalar_dict:
                scalar_dict[state][1] = (scalar_dict[state][0] * scalar_dict[state][1] + scalars[i][j][1]) \
                                        /(scalar_dict[state][0] + 1)
                scalar_dict[state][0] += 1
            else:
                scalar_dict[state] = [1,scalars[i][j][1]]
    return scalar_dict

def sort_by_time(scalars, neplength):
    scalar_dict = {}
    for i in range(len(scalars)):
        for j in range(neplength):
            if j in scalar_dict:
                try:
                    scalar_dict[j][1] = (scalar_dict[j][0] * scalar_dict[j][1] + scalars[i][j][1]) \
                                        /(scalar_dict[j][0] + 1)
                    scalar_dict[j][0] += 1
                except:
                    continue
            else:
                try:
                    scalar_dict[j] = [1, scalars[i][j][1]]
                except:
                    continue
    return scalar_dict

def invalids_by_goal(goals):
    invalids = []
    for i in range(goals.shape[0]):
        if np.all(goals[i] == 0):
            invalids.append(i)
    return invalids

def decode_trajectories(states):
    trajectory = []
    for i in range(states.shape[0]):
        trajectory.append(decode(states[i]))
    return trajectory
            
def pad(arr, minsize): # arr must be at least minsize
    nbatch = arr.shape[0]
    d = minsize - nbatch
    if d > 0:
        rep = [1]*(nbatch-1)+[d+1]
        arr = np.repeat(arr, rep, axis=0)
    return arr

def sbi(arr, dones):
    nbatch=dones.shape[0]
    abd=[]
    si=0
    for t in range(nbatch):
        if dones[t] == 1:
            abd.append(arr[si:t+1])
            si=t+1
        elif t==nbatch-1:
            abd.append(arr[si:])
    return abd

def pack(arr):
    try:
        arr = np.vstack(arr)
        if arr.shape[0]==1:
            return np.flatten(arr)
        else: return arr
    except:
        return np.hstack(arr)

def constfn(val):
    def f(_):
        return val
    return f

def mcret(actions, rews, dones, vals, lam=0.95, gam=0.99):
    mb_returns = np.zeros_like(rews)
    mb_advs = np.zeros_like(rews)
    lastgaelam = 0
    nsteps = rews.shape[0]
    rews = np.concatenate([rews, [np.zeros_like(rews[0])]], axis=0) # add a zero column at end
    nextvalues=vals[-1:,]
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextnonterminal = 0
            nextvalues = 0 # assume last is terminal -> won't be too significant unless tstep is large
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = vals[t+1]
        delta = rews[t+1] + gam * nextvalues * nextnonterminal - vals[t]
        mb_advs[t] = lastgaelam = delta + gam * lam * nextnonterminal * lastgaelam
        
    mb_returns = mb_advs + vals
    return mb_returns, mb_advs

def recurrent_mcret(actions, rews, dones, vals, lam=0.95, gam=0.99):
    mb_returns = np.zeros_like(rews)
    mb_advs = np.zeros_like(rews)
    lastgaelam = 0
    nsteps = rews.shape[1]
    nextvalues=vals[:,-1:,]
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextnonterminal = 0
            nextvalues = 0 # assume last is terminal -> won't be too significant unless tstep is large
        else:
            nextnonterminal = 1.0
            nextvalues = vals[:,t+1,:]
        delta = rews[:,t,:] + gam * nextvalues * nextnonterminal - vals[:,t,:]
        mb_advs[:,t,:] = lastgaelam = delta + gam * lam * nextnonterminal * lastgaelam
        
    mb_returns = mb_advs + vals
    return mb_returns, mb_advs

def safe_vstack(arr, dim1):
    assert arr
    shape = arr[0].shape
    return np.reshape(np.vstack(arr), (dim1,) + shape)
    
def learn(*, policy, env, tsteps, nsteps, encoef, lr, cliphigh, clipinc, vcoef,
          mgn, gmax, ginc, lam, nhier, nmb, noe, ngmin, nginc, bmin, bmax, nhist,
          recurrent, cos, val, fixed_manager, fixed_agent, goal_state, nhidden=64, max_len=100,
          save_interval=0, log_interval=1, test_interval=1, test_env=None,
          logger=None, load_path=None):
    
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliphigh, float):
        arr = np.asarray([cliphigh*(clipinc**i) for i in range(nhier)], dtype=np.float32) 
        cliprange = constfn(arr)
    else: 
        def cr(t):
            arr = [cliphigh(t)*(clipinc(t)**i) for i in range(nhier)]
            return np.asarray(arr, dtype=np.float32)
        cliprange = cr    

    neplength = max_len
    ob_space = env.observation_space
    ac_space = env.action_space
    assert nsteps%max_len == 0
    if recurrent:
        nbatch = nsteps//max_len
    else:
        nbatch = nsteps
    nupdates = tsteps//nsteps
    nbatch_train = nbatch // nmb
    
    def ng(k):
        return ngmin * (nginc **(nhier - k))
    def gamma(k):
        return 1 - (gmax * (ginc ** (nhier - 1 - k)))
    def nh(k):
        return nhist ** (k)
    def beta(k):
        if nhier==1:
            return bmin            
        else:
            return bmin + (bmax - bmin) * k / (nhier - 1)
    
    if recurrent:
        make_model = lambda : RecurrentFeudalModel(policy, env, ob_space, ac_space,
              neplength=neplength, max_grad=mgn,
              ngoal=ng, recurrent=recurrent, g=gamma, nhist=nh, b=beta, nhier=nhier,
              val=val, cos=cos, fixed_agent=fixed_agent, fixed_network=fixed_manager, goal_state=goal_state,
              encoef=encoef, vcoef=vcoef, nh=nhidden)
    else:
        make_model = lambda : FeudalModel(policy, env, ob_space, ac_space, max_grad=mgn,
              ngoal=ng, recurrent=recurrent, g=gamma, nhist=nh, b=beta, nhier=nhier,
              val=val, cos=cos, fixed_agent=fixed_agent, fixed_network=fixed_manager, goal_state=goal_state,
              encoef=encoef, vcoef=vcoef, nh=nhidden)
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    
    runner = FeudalRunner(env=env, model=model, nsteps=max_len, 
                          recurrent=recurrent, fixed_manager=fixed_manager,
                          fixed_agent=fixed_agent)
    test_runner = FeudalRunner(env=test_env, model=model, nsteps=max_len,
                          recurrent=recurrent, fixed_manager=fixed_manager,
                          fixed_agent=fixed_agent)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
    
    state_rew_logger = Logger(dir=logger.dir, output_format=['CSV'], csv_tag='sinrs.csv')
    time_rew_logger = Logger(dir=logger.dir, output_format=['CSV'], csv_tag='tinrs.csv')
    test_run_logger = Logger(dir=logger.dir, output_format=['TXT'], txt_tag='test_traj.txt')
    
    if not val:
        vre = np.zeros((nhier), dtype=np.float32)
        val_temp = 0.9
    
    for update in range(1, nupdates+1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, rewards, actions, dones, mbpi, init_goals, goals, states, epinfos = runner.run()
        trun = time.time()
        print(trun - tstart)
        epinfobuf.extend(epinfos)
        mblossvals = []
        
        if not recurrent:
            rewards, vfs, nlps, inrs = model.av(obs, actions, rewards, dones, goals, states, init_goals)
            tstats = time.time()
            print(tstats - trun)
            #perform tally for each unique goal
            if fixed_manager and nhier > 1:
                inrs_per_goal = sort_by_state(inrs, init_goals, env)
            
                for index,i in enumerate(inrs_per_goal.items()):
                    state_rew_logger.logkv("{}".format(i[0]), i[1][1])
                state_rew_logger.dumpkvs()
                
                inrs_per_timestep = sort_by_time(inrs, neplength)
                for index,i in enumerate(inrs_per_timestep.items()):
                    time_rew_logger.logkv("{}".format(index), i[1][1])
                time_rew_logger.dumpkvs()
            rewards, vfs, nlps, inrs = map(np.asarray,(rewards, vfs, nlps, inrs))
            states = states[:,np.newaxis,:]
            states = np.tile(states, (1, neplength, *np.ones_like(states.shape[2:])))
            obs, actions, dones, mbpi, init_goals, goals, states, rewards, \
                vfs, nlps, inrs = (fl01(arr) for arr in (obs, actions, dones,
                                   mbpi, init_goals, goals, states, rewards,
                                   vfs, nlps, inrs))
            number_of_correct = np.sum(np.where(inrs[:,-1] > 0.99, True, False))
            mean_inr = np.mean(inrs, axis=0)
            if not val:
                vre = vre * val_temp + np.mean(rewards, axis=0) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nsteps), [nsteps, nhier])
            rewards, advs = mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            actions = actions.flatten() #safety
            inds = np.arange(nbatch)
            invalid_inds = invalids_by_goal(init_goals)
            if nhier == 1:
                goals = np.zeros((nbatch, 0, model.maxdim))
            for _ in range(noe):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    mbinds_deleted = [i for i in mbinds if i not in invalid_inds]
                    slices = (arr[mbinds_deleted] for arr in (obs, actions, rewards, advs, goals, nlps, vfs, states, init_goals))    
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
            
            ttrain = time.time()
            print(ttrain - tstats)

        else: # recurrent version
            rewards, vfs, nlps, inrs = model.av(obs, actions, rewards, dones, goals, states, init_goals)
            pre_vars = (obs,actions,rewards,dones,goals,nlps,vfs,states,inrs,init_goals) 
            map_vars = (safe_vstack(arr, nbatch) for arr in pre_vars)
            (obs,actions,rewards,dones,goals,nlps,vfs,states,inrs,init_goals) = map_vars
            
            if not val:
                vre = vre * val_temp + np.apply_over_axes(np.mean, rewards, [0,1]) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nbatch*neplength), [nbatch, neplength, nhier])
            rewards, advs = recurrent_mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            
            feed_vars = (obs, actions, rewards, advs, goals, nlps, vfs, states, init_goals)
            mean_inr = np.mean(inrs, axis=(0,1))
            inds = np.arange(nbatch)
            for _ in range(noe):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[0:nbatch_train]
                    slices = (arr[mbinds] for arr in feed_vars)
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        if update == 1 or update % test_interval == 0:
            obs, rewards, actions, dones, mbpi, init_goals, goals, states, kepinfos = test_runner.run()
            rewards, vfs, nlps, inrs = model.av(obs, actions, rewards, dones, goals, states, init_goals)
            trajectories = decode_trajectories(obs[0])
            test_run_logger.logkv('update_number', update)
            for i in range(len(trajectories)):
                test_run_logger.logkv('state_{}'.format(i), '{}'.format(trajectories[i]))
                test_run_logger.logkv('goal_{}'.format(i), '{}'.format(decode(init_goals[0][i])))
                test_run_logger.logkv('inr_{}'.format(i), '{}'.format(inrs[0][i]))
                test_run_logger.logkv('act_{}'.format(i), '{}'.format(actions[0][i]))
                test_run_logger.logkv('rew_{}'.format(i), '{}'.format(rewards[0][i]))
            test_run_logger.dumpkvs()

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            if logger is not None:
                logger.logkv("serial_timesteps", update*nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                logger.logkv("exact_matches", number_of_correct)
                for i in range(1, nhier):
                    logger.logkv('intrinsic_reward_{}'.format(i), mean_inr[i] * neplength/(neplength - 1))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

    

    