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
from baselines import logger
from collections import deque
from baselines.feudal.models import I2AModel
from baselines.feudal.runners import I2ARunner

PATH="tmp/build/graph"

def package_environment(states, actions, rewards):
    train_states = []
    train_actions = []
    train_rewards = []
    train_nstates = []
    
    for (state, action, reward) in states, actions, rewards:
        train_states.append(state[:-1])
        train_nstates.append(state[1:])
        train_actions.append(action[:-1])
        train_rewards.append(reward[:-1])
        
    (np.asarray(arr).reshape((-1, arr.shape[-1])))

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

def learn(*, policy, env, tsteps, nsteps, encoef, lr, cliphigh, clipinc, vcoef,
          mgn, gmax, ginc, lam, nhier, nmb, noe, ngmin, nginc, bmin, bmax, nhist,
          recurrent, val, max_len=100, save_interval=0, log_interval=1, load_path=None):
    
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
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = (nenvs * nsteps)
    nbatch_train = nbatch // nmb
    
    make_model = lambda : I2AModel(policy, ob_space, ac_space, max_grad=mgn,
              encoef=encoef, vcoef=vcoef, klcoef=klcoef, aggregator='concat',
              traj_len = tl, nh=nh)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    
    runner = I2ARunner(env=env, model=model, nsteps=nsteps)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
    nupdates = tsteps//nbatch
    
    if not val:
        vre = np.zeros((nhier), dtype=np.float32)
        val_temp = 0.9
    
    for update in range(1, nupdates+1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, rewards, actions, dones, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        obs, actions, rewards, dones = (sbi(arr, dones) for arr in
                                        (obs, actions, rewards, dones))
        env_train_set = package_environment(obs, actions, rewards)
        if not recurrent:
            nlps, vfs = model.info(obs, actions)
            obs, actions, rewards, dones, nlps, vfs = \
                map(pack,(obs,actions,rewards,dones,nlps,vfs))
            if not val:
                vre = vre * val_temp + np.mean(rewards, axis=0) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nsteps), [nsteps, nhier])
            rewards, advs = mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            actions = actions.flatten() #safety
            inds = np.arange(nbatch)
            for _ in range(noe):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, actions, nlps, advs, rewards, vfs))    
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else: # recurrent version
            pass
        
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
