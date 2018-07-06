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
from baselines.feudal.runners import FeudalRunner, I2ARunner

PATH="tmp/build/graph"
    
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
    nextvalues=vals[-1:,]
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextnonterminal = 0
            nextvalues = 0 # assume last is terminal -> won't be too significant unless tstep is large
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = vals[t+1]
        delta = rews[t] + gam * nextvalues * nextnonterminal - vals[t]
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
          recurrent, cos, val, max_len=100, save_interval=0, log_interval=1,
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

    nenvs = env.num_envs
    neplength = max_len
    ob_space = env.observation_space
    ac_space = env.action_space
    assert (nenvs * nsteps)%max_len == 0
    if recurrent:
        nbatch = (nenvs * nsteps)//max_len
    else:
        nbatch = (nenvs * nsteps)
    nupdates = tsteps//(nenvs * nsteps)
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
        make_model = lambda : RecurrentFeudalModel(policy, ob_space, ac_space, neplength=neplength, max_grad=mgn,
              ngoal=ng, recurrent=recurrent, g=gamma, nhist=nh, b=beta, nhier=nhier,
              val=val, cos=cos)
    else:
        make_model = lambda : FeudalModel(policy, ob_space, ac_space, max_grad=mgn,
              ngoal=ng, recurrent=recurrent, g=gamma, nhist=nh, b=beta, nhier=nhier,
              val=val, cos=cos)
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    
    runner = FeudalRunner(env=env, model=model, nsteps=nsteps, recurrent=recurrent)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
    
    if not val:
        vre = np.zeros((nhier), dtype=np.float32)
        val_temp = 0.9
    
    for update in range(1, nupdates+1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, rewards, actions, dones, mbpi, goals, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        obs, actions, rewards, dones, goals, states = (sbi(arr, dones) for arr in
                                        (obs, actions, rewards, dones, goals, states))
        if not recurrent:
            rewards, vecs, vfs, nlps, inrs = model.av(obs, actions, rewards, dones, goals, states)
            obs,actions,rewards,dones,vecs,goals,nlps,vfs,states, inrs = \
                map(pack,(obs,actions,rewards,dones,vecs,goals,nlps,vfs,states,inrs))
            print(inrs.shape)
            mean_inr = np.mean(inrs, axis=0)
            if not val:
                vre = vre * val_temp + np.mean(rewards, axis=0) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nsteps), [nsteps, nhier])
            rewards, advs = mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            actions = actions.flatten() #safety
            inds = np.arange(nbatch)
            if nhier == 1:
                goals = np.zeros((nbatch, 0, model.maxdim))
            for _ in range(noe):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, actions, rewards, advs, vecs, goals, nlps, vfs, states))    
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else: # recurrent version
            rewards, vecs, vfs, nlps, inrs = model.av(obs, actions, rewards, dones, goals, states)
            pre_vars = (obs,actions,rewards,dones,goals,nlps,vfs,states,inrs) 
            map_vars = (safe_vstack(arr, nbatch) for arr in pre_vars)
            (obs,actions,rewards,dones,goals,nlps,vfs,states,inrs) = map_vars
            
            if not val:
                vre = vre * val_temp + np.apply_over_axes(np.mean, rewards, [0,1]) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nbatch*neplength), [nbatch, neplength, nhier])
            rewards, advs = recurrent_mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            
            feed_vars = (obs, actions, rewards, advs, goals, nlps, vfs, states)
            print(inrs.shape)
            mean_inr = np.mean(inrs, axis=(0,1))
            inds = np.arange(nbatch)
            for _ in range(noe):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in feed_vars)
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            if logger is not None:
                logger.logkv("serial_timesteps", update*nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                for i in range(1, nhier):
                    logger.logkv('intrinsic_reward_{}'.format(i), mean_inr[i]*neplength)
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

    

    