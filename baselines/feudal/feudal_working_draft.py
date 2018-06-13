#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 00:53:58 2018

@author: matthewszhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:51:54 2018

@author: matthewszhang
"""
import time
import random
import os
import os.path as osp
import tensorflow as tf
import numpy as np
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import fc

PATH="tmp/build/graph"

class FeudalNetwork(object):
    def __init__(self, mgoal, state, pdtype=None, nhist=4, nin=32, ngoal=16,
                 nembed=8, manager=False, nh=64, activ=tf.nn.relu, name=1, nbatch=1e3):
        self.mgoal=mgoal
        self.state=state
        self.manager=manager
        self.name=name
        self.initial_state=None
        nout = ngoal if manager else nh
        self.pdtype = pdtype
        
        with tf.variable_scope("level" + str(self.name)):
            em_h2 = activ(fc(state, 'em_fc2', nh=nout, init_scale=np.sqrt(2)))
            embed_weights = tf.get_variable("embed", [nin, nembed])
            g_out = tf.matmul(self.mgoal, embed_weights)
            pi_h1 = activ(fc(em_h2, 'pi_fc1', nh=nembed * nout, init_scale=np.sqrt(2)))
            
            pi_flat = tf.reshape(pi_h1, [-1, nout, nembed])
            pout = tf.einsum("ijk,ik->ij", pi_flat, g_out)
            vout = tf.nn.tanh(fc(em_h2, 'vf', 1))[:,0]
        
        if self.manager:
            aout = tf.nn.l2_normalize(pout, axis=-1)
            neglogpout = None
            
        else:
            assert self.pdtype is not None
            self.pd, self.pi = self.pdtype.pdfromlatent(pout, init_scale=0.01)
            aout = self.pd.sample()
            neglogpout = self.pd.neglogp(aout)
            
        self.aout = aout
        self.nlp = neglogpout
        
        def bcs(state, spad, gpad, nhist):
            rew = tf.fill([nbatch], 0.0)
            for t in range(nhist):
                svec = state - spad[nhist-t-1:-(t+1),:]
                gvec = gpad[nhist-t-1:-(t+1),:]
                nsv = tf.nn.l2_normalize(svec, axis=-1)
                ngv = tf.nn.l2_normalize(gvec, axis=-1)
                rew += tf.reduce_sum(tf.multiply(nsv, ngv), axis=-1)
            return rew
        
        def fcs(state, fstate, gvec, nhist):
            svec = fstate - state
            nsv = tf.nn.l2_normalize(svec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(tf.stop_gradient(nsv), ngv), axis=-1)
            return sim
        
        self.vf = vout
        if self.manager:    
            pad = tf.constant([[nhist,0], [0,0]])
            spad = tf.pad(em_h2, pad, "CONSTANT")
            gpad = tf.pad(aout, pad, "CONSTANT")
            self.inr = 1/nhist * tf.stop_gradient(bcs(em_h2, spad, gpad, nhist))
            
            lstate = em_h2[-1,:]
            rep = tf.reshape(tf.tile(lstate, tf.constant([nhist])),(nhist,nout))
            spadf = tf.concat([em_h2, rep], axis=0)
            self.traj_sim = fcs(em_h2, spadf[nhist:,], aout, nhist)
           
class RecurrentFeudalNetwork(object):
    def __init__(self, pdtype, ngoal=16, nfeat=32, nac=8, manager=False, 
                 hier=2, nh=64, activ=tf.nn.relu, init_state = None):
        self.init_state = init_state
        pass
    
class FeudalModel(object):
    def __init__(self, ob_space, ac_space, nhier=3, max_grad=0.5,
                 ngoal=lambda x:max(8, int(32/(2**x))),
                 b=lambda x:0.25*x, recurrent=False, 
                 g=lambda x:1-0.25**(x+1), nhist=lambda x:4**x,
                 lr=1e-4, vcoef=0.5, encoef=0, nbatch_train=1e3, nh=64,
                 activ=tf.nn.relu):
        self.sess = tf.get_default_session()
        self.net = RecurrentFeudalNetwork if recurrent else FeudalNetwork
        self.maxnhist=nhist(nhier)
        self.networks=[]
        self.nhier=nhier
        self.initial_state=None
        beta, gam=[],[]
        self.lavg=0
        self.init_goal = np.ones(shape=(ngoal(-1)))
        pdtype = make_pdtype(ac_space)
        nfeat = ob_space.shape
        
        self.obs=tf.placeholder(dtype=tf.float32, shape=(None,)+nfeat)
        self.goal=tf.placeholder(dtype=tf.float32, shape=(None, ngoal(-1)))
        self.rew=tf.placeholder(dtype=tf.float32, shape=(None, nhier))
        self.actions=tf.placeholder(dtype=tf.int32, shape=(None,))
        self.lr=tf.placeholder(dtype=tf.float32)
        nbatch=tf.shape(self.obs)[0]
        inr = [tf.zeros(shape=(nbatch))]
        pol_loss=tf.zeros(1)
        val_loss = tf.zeros(shape=(nbatch))
        goal=[self.goal]
        tsim=[]
        val=[]
        
        flatten = tf.layers.flatten
        with tf.variable_scope("common"):
            em_h0 = activ(fc(flatten(self.obs), 'em_fc0', nh=nh, init_scale=np.sqrt(2)))
            em_h1 = activ(fc(em_h0, 'em_fc1', nh=nh, init_scale=np.sqrt(2)))
        
        for t in range(nhier-1):
            beta.append(b(t))
            gam.append(g(nhier - t))
            self.networks.append(self.net(mgoal=goal[t], state=em_h1, nhist=nhist(nhier-t), 
                           nin=ngoal(t-1), ngoal=ngoal(t), nbatch=nbatch,
                           name=nhier-t, pdtype=None, manager=True))
            goal.append(tf.stop_gradient(self.networks[t].aout))
            tsim.append(1-self.networks[t].traj_sim)
            val.append(self.networks[t].vf)
            val_loss += 1/2 * tf.square(val[t] - self.rew[:,t])
            pol_loss += tf.multiply(tsim[t], self.rew[:,t] - tf.stop_gradient(val[t]))
            inr.append(self.networks[t].inr)
            
        beta.append(b(nhier))
        gam.append(g(1))
        self.networks.append(self.net(mgoal=goal[nhier-1], state=em_h1, nin=ngoal(nhier-2), name=0,
                       ngoal=ngoal(nhier-1), pdtype=pdtype, manager=False,
                       nhist=nhist(1), nbatch=nbatch))
        val.append(self.networks[nhier-1].vf)
        val_loss += 1/2 * tf.square(val[nhier-1] - self.rew[:,t])
        nlp = self.networks[nhier-1].pd.neglogp(self.actions)
        pol_loss += tf.multiply(nlp, self.rew[:,nhier-1] - tf.stop_gradient(val[nhier-1]))
        self.pi=self.networks[nhier-1].pi
        self.adv = tf.transpose(tf.stack(inr))
        self.val_loss = tf.reduce_mean(val_loss)
        self.entropy = tf.reduce_mean(self.networks[nhier-1].pd.entropy())
        self.beta = np.asarray(beta)
        self.gam = np.asarray(gam)
        self.act = self.networks[nhier-1].aout
        self.pol_loss = tf.reduce_mean(pol_loss)
        self.inrmean=tf.reduce_mean(self.adv)
        self.loss = self.pol_loss + self.val_loss * vcoef - self.entropy * encoef
        self.loss_names = ["entropy", "policy loss", "value loss"]
        optimizer = tf.train.AdamOptimizer(lr)
        
        if max_grad is not None:
            gradients = optimizer.compute_gradients(self.loss)
            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -1, 1)
            clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
            self._trainer = optimizer.apply_gradients(clipped_gradients)
        else:
            self._trainer = optimizer.minimize(self.loss)
        
        tf.global_variables_initializer().run(session=self.sess)
        
    def train(self, ar, lr, obs, acts, rews, dones, eplens=None, init_goal=None):
        nbatch=obs.shape[0]
        rews=np.reshape(np.repeat(rews, self.nhier, -1),(nbatch, self.nhier))
        if init_goal==None: goal = np.tile(self.init_goal,(nbatch,1))
        else: goal = init_goal
        lr = self.lr if lr == None else lr
        inr=self.rewards(obs, goal=goal)
        mbrews=rews*(1-self.beta) + inr*(self.beta)
        mbadvs=np.zeros((nbatch,self.nhier))
        mbnonterm=1.0-dones
        mbvals=np.zeros(self.nhier)
        for t in reversed(range(nbatch)):
            mbvals=mbvals*mbnonterm[t]*self.gam+mbrews[t,:]
            mbadvs[t,:]=mbvals
            index=int(acts[t])
            ar[index]+=mbadvs[t,self.nhier-1]
        feed={self.rew:mbadvs, self.actions:acts, self.obs:obs,
              self.goal:goal, self.lr:lr}
        return self.sess.run([self.entropy,
                              self.pol_loss,
                              self.val_loss,
                              self._trainer], feed)[:-1]
    
    def step(self, obs, init_goal=None):
        if init_goal==None: goal = self.init_goal
        else: goal = init_goal
        goal=np.tile(goal, (obs.shape[0],1))
        feed={self.obs:obs, self.goal:goal}
        return self.sess.run([self.act, self.pi], feed)
            
    def rewards(self, obs, goal):
        feed={self.obs:obs, self.goal:goal}
        rew = self.sess.run([self.adv], feed)[0]
        return rew
    
    def extend(self):
        pass # should create another level of hierarchy dynamically as the look-ahead threshold is surpassed
        # idk how to do this yet
        
class FeudalRunner(AbstractEnvRunner):
    # to do -> work on making a recurrent version
    def __init__(self, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        
    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi = [],[],[],[],[]
        epinfos = []
        mb_states = self.states
        for _ in range(self.nsteps):
            actions, pi = self.model.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_pi.append(pi)
            mb_actions.append(actions)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_pi = np.asarray(mb_pi, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        return (*map(sf01, (mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi)), 
                mb_states, epinfos)

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

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
    
def constfn(val):
    def f(_):
        return val
    return f

def learn(*, env, nsteps, total_timesteps, encoef, lr,
            vcoef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=1,
            save_interval=0, load_path=None, nhier=3, recurrent=False):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    
    make_model = lambda : FeudalModel(ob_space, ac_space, nhier=nhier, encoef=encoef,
                                      vcoef=vcoef, recurrent=recurrent,
                                      nbatch_train=nbatch_train)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)
    
    runner = FeudalRunner(env=env, model=model, nsteps=nsteps)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        avrew=np.zeros(ac_space.n)
        obs, rewards, actions, dones, mbpi, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None:
            obs, actions, rewards, dones = (sbi(arr, dones) for arr in
                                            (obs, actions, rewards, dones))
            ntraj = list(range(len(obs)))
            ntraj *= noptepochs
            random.shuffle(ntraj)
            for i in ntraj:
                slices = (arr[i] for arr in (obs, actions, rewards, dones))
                mblossvals.append(model.train(avrew, lrnow, *slices))

        else: # recurrent version
            pass
        
        lossvals = np.mean(mblossvals, axis=0)
        pi = np.mean(mbpi, axis=0)
        print(pi)
        print(avrew/nbatch)
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

    

    