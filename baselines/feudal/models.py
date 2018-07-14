#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:18:48 2018

@author: matthewszhang
"""
import tensorflow as tf
import numpy as np
from baselines.feudal.distributions import make_pdtype
from baselines.feudal.networks import FeudalNetwork, RecurrentFeudalNetwork, \
                                     FixedManagerNetwork
from baselines.feudal.i2a_helpers import Encoder, MBPolicy, MFPolicy, EnvironmentModel

from gym import spaces

PATH="tmp/build/graph"
    
class FeudalModel(object):
    '''
    General class for organizing feudal networks
    Can train all networks in one sess.run call
    '''
    def __init__(self, policy, env, ob_space, ac_space, nhier=2, max_grad=0.5,
                 ngoal=lambda x:max(8, int(64/(2**x))), recurrent=False, 
                 g=lambda x:1-0.25**(x+1), nhist=lambda x:4**x, val=True,
                 lr=1e-4, vcoef=0.5, encoef=0, nh=64, b=lambda x:0.3 * x,
                 activ=tf.nn.relu, cos=False, fixed_network=0, 
                 goal_state=None):
        '''
        INPUTS:
           policy - encoding function for input states
           ob_space - gym.spaces object for obs (inputs)
           ac_space - gym.spaces object for actions (outputs)
           max_grad - gradient clipping threshold (float)
           nhier - number of hierarchical actions (int)
           ngoal - size of goal output at each level (int function)
           recurrent - use recurrent model (Boole)
           g - gamma, discounting parameter (float function)
           nhist - lookahead horizon at each level (int function)
           lr - learning rate (float)
           vcoef - value loss weighting (float)
           encoef - entropy weighting (float)
           nh - hidden layer size (int)
           activ - activation function (TF activation function)
        '''
        
        self.sess = tf.get_default_session() # get session
        if fixed_network:
            self.manager_net = FixedManagerNetwork
            self.maxdim = policy.out_shape
            if self.maxdim == None:
                self.maxdim = goal_state.shape[-1]
            self.initdim = goal_state.shape[-1]
        else:
            self.maxdim = ngoal(1)
            self.initdim = self.maxdim
            self.manager_net = FeudalNetwork # get single network object
        self.net = FeudalNetwork
        self.recurrent=recurrent
        self.val=val
        self.networks=[] # network array
        self.nhier=nhier # set hierarchy
        beta, gam, tsim, val, nlp, nstate=[],[],[],[],[],[] # hierarchically dependent parameters
        self.init_goal = np.zeros(shape=(self.maxdim)) # 
        self.initial_state = np.zeros(shape=(nhier, nh*2))
        nfeat = ob_space.shape
        
        self.STATES=tf.placeholder(dtype=tf.float32, shape=(None, nhier, nh*2))
        self.OBS=tf.placeholder(dtype=tf.float32, shape=(None,)+nfeat)
        self.INITGOALS=tf.placeholder(dtype=tf.float32, shape=(None, self.initdim))
        self.R=tf.placeholder(dtype=tf.float32, shape=(None, nhier))
        self.ADV=tf.placeholder(dtype=tf.float32, shape=(None, nhier))
        self.OLDACTIONS=tf.placeholder(dtype=tf.int32, shape=(None,))
        self.OLDVALUES=tf.placeholder(dtype=tf.float32, shape=(None, nhier))
        self.OLDNLPS=tf.placeholder(dtype=tf.float32, shape=(None,nhier))
        self.OLDGOALS=tf.placeholder(dtype=tf.float32, shape=(None, nhier-1, self.maxdim))
        self.CLIPRANGE=tf.placeholder(dtype=tf.float32, shape=(nhier))
        self.LR=tf.placeholder(dtype=tf.float32)
        
        nbatch=tf.shape(self.OBS)[0]
        inr = [tf.zeros(shape=(nbatch))]
        ploss=tf.zeros(1)
        vloss = tf.zeros(1)
        entropy = tf.zeros(1)
        goal=[self.INITGOALS]
        
        with tf.variable_scope("common"):
            em_h1=policy(self.OBS)
            
        for t in range(nhier-1):
            beta.append(b(t))
            gam.append(g(t))
            pdtype=make_pdtype(spaces.Box(low=-1, high=1, shape=(ngoal(nhier-t-1),)))
            
            if fixed_network:
                self.networks.append(self.manager_net(goal_state=goal[-1],
                                              state=em_h1, 
                                              recurrent=recurrent, 
                                              nhist=nhist(nhier-t-1),
                                              nh=nh,
                                              policy=policy,
                                              nbatch=nbatch))
            else:
                self.networks.append(self.manager_net(mgoal=goal[t][:,:ngoal(nhier-t)],
                                              state=em_h1,
                                              pstate=self.STATES[:,t,:],
                                              nhist=nhist(nhier-t-1),
                                              pdtype=pdtype,
                                              nin=ngoal(nhier-t),
                                              ngoal=ngoal(nhier-t-1),
                                              nbatch=nbatch,
                                              name=nhier-t-1,
                                              manager=True,
                                              val=val))
            if fixed_network:
                goal.append(self.networks[t].aout)
            else:
                goal.append(tf.pad(self.networks[t].aout,
                                tf.constant([[0,0],[0,self.maxdim-ngoal(nhier-t-1)]]),
                                mode='CONSTANT'))
            nlp.append(self.networks[t].pd.neglogp(self.OLDGOALS[:,t,:ngoal(nhier-t-1)]))
            
            tsim.append(1-self.networks[t].traj_sim)
            inr.append(self.networks[t].inr)
            nstate.append(self.networks[t].nstate)
            if cos:
                adv = self.ADV[:,t] * tsim[t]
            else:
                adv = self.ADV[:,t] #* tsim[t]
            #tmax = tf.reduce_max(tf.exp(self.OLDNLPS[:,t] - nlp[t]))
            ratio = tf.exp(self.OLDNLPS[:,t] - nlp[t])
            pl1 = -adv * ratio
            pl2 = -adv * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE[t], 1.0 + self.CLIPRANGE[t])
            ploss += tf.reduce_mean(tf.maximum(pl1, pl2))
            if val:
                val.append(self.networks[t].vf)
                vclip = self.OLDVALUES[:,t] + tf.clip_by_value(val[t]-self.OLDVALUES[:,t],
                                      -self.CLIPRANGE[t], self.CLIPRANGE[t]) 
                vl1 = tf.square(val[t] - self.R[:,t])
                vl2 = tf.square(vclip - self.R[:,t])
                vloss += .5 * tf.reduce_mean(tf.maximum(vl1, vl2))
            entropy += tf.reduce_mean(self.networks[t].pd.entropy())
            
        beta.append(b(nhier-1))
        gam.append(g(nhier-1))
        pdtype = make_pdtype(ac_space)
        self.nout = self.maxdim if fixed_network else ngoal(1)
        self.networks.append(self.net(mgoal=goal[nhier-1],
                                      state=em_h1,
                                      pstate=self.STATES[:,nhier-1,:],
                                      nin=self.nout,
                                      name=0,
                                      nh=nh,
                                      ngoal=ngoal(0),
                                      pdtype=pdtype,
                                      manager=False,
                                      nhist=nhist(0),
                                      nbatch=nbatch,
                                      val=val))
        nlp.append(self.networks[nhier-1].pd.neglogp(self.OLDACTIONS))
        nstate.append(self.networks[nhier-1].nstate)
        adv = self.ADV[:,nhier-1]
        ratio = tf.exp(self.OLDNLPS[:,nhier-1] - nlp[nhier-1])
        pl1 = -adv * ratio
        pl2 = -adv * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE[nhier-1], 1.0 + self.CLIPRANGE[nhier-1])
        ploss += tf.reduce_mean(tf.maximum(pl1, pl2))
        
        if val:
            val.append(self.networks[nhier-1].vf)
            vclip = self.OLDVALUES[:,nhier-1]+tf.clip_by_value(val[nhier-1]-
                                  self.OLDVALUES[:,nhier-1], -self.CLIPRANGE[nhier-1], self.CLIPRANGE[nhier-1]) 
            vf_losses1 = tf.square(val[nhier-1] - self.R[:,nhier-1])
            vf_losses2 = tf.square(vclip - self.R[:,nhier-1])
            vloss += .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.approxkl = .5 * tf.reduce_mean(tf.square(nlp[nhier-1] - self.OLDNLPS[:,nhier-1]))
        self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.CLIPRANGE[nhier-1])))
        
        entropy += tf.reduce_mean(self.networks[nhier-1].pd.entropy())
        
        self.pi=self.networks[nhier-1].pi
        if nhier > 1:  
            self.goals = tf.transpose(tf.stack(goal[1:]),[1,0,2])
        else:
            self.goals=tf.zeros(shape=[nbatch, nhier-1, self.maxdim])
        if self.recurrent:
            self.nstate=tf.stack(nstate)
        else:
            self.nstate=tf.zeros(shape=(nhier, nh*2))
        self.inr = tf.transpose(tf.stack(inr))
        self.nlp = tf.transpose(tf.stack(nlp))
        if val:
            self.vf = tf.transpose(tf.stack(val))
        else:
            self.vf = tf.zeros((nbatch, nhier))
        self.vloss = vloss[0]
        self.beta = np.asarray(beta)
        self.gam = np.asarray(gam)
        self.act = self.networks[nhier-1].aout
        self.ploss = ploss[0]
        self.entropy = entropy[0]
        self.inrmean = tf.reduce_mean(self.inr)
        self.loss = self.ploss + self.vloss * vcoef - self.entropy * encoef
        self.loss_names = ["entropy", "policy loss", "value loss", "approxkl", "clipfrac"]
        optimizer = tf.train.AdamOptimizer(lr)
        if max_grad > 0:
            params = tf.trainable_variables()
            grads = tf.gradients(self.loss, params)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
            grads = list(zip(grads, params))
            self._trainer = optimizer.apply_gradients(grads)
        else:
            self._trainer = optimizer.minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)
        
    def train(self, lr,
              cliprange, obs,
              acts, rews,
              advs,
              goals, nlps,
              vfs, states,
              init_goal=None):
        nbatch=obs.shape[0]
        if init_goal is None: init_goal = np.tile(self.init_goal,(nbatch,1))
        else: init_goal = init_goal
        if isinstance(cliprange, float): cliprange = np.array([cliprange/(2 ** i) for i in range(self.nhier)])
        else: assert cliprange.shape[0] == self.nhier
        
        try:
            goals = np.reshape(goals, [nbatch, self.nhier-1, self.maxdim])
        except:
            pass
        lr = self.lr if lr == None else lr
        feed={self.STATES:states,
              self.R:rews,
              self.ADV:advs,
              self.OLDACTIONS:acts,
              self.OBS:obs,
              self.INITGOALS:init_goal,
              self.LR:lr, 
              self.CLIPRANGE:cliprange,
              self.OLDGOALS:goals,
              self.OLDNLPS:nlps,
              self.OLDVALUES:vfs}
        return self.sess.run([self.entropy,
                              self.ploss,
                              self.vloss,
                              self.approxkl,
                              self.clipfrac,
                              self._trainer], feed)[:-1]
    
    def av(self, obs, acts, rews, dones, goals, states, init_goal=None):
        vadvs, vvfs, vnlps, vinr =[],[],[],[]
        if init_goal is None:
            init_goal = [None] * len(obs)
            
        for (ob, act, rew, done, goal, state, init_goal) in \
                zip(obs,acts,rews,dones,goals,states,init_goal):
            nbatch=ob.shape[0]
            trews=np.reshape(np.repeat(rew, self.nhier, -1),(nbatch, self.nhier))
            if init_goal is None:
                feed_goal = np.tile(self.init_goal,(nbatch,1))
            else:
                feed_goal = init_goal
            inr=self.rewards(ob, state, init_goal=feed_goal)
            mbrews=trews*(1-self.beta) + inr*(self.beta)
            mbvfs, mbnlps =self.ifv(ob, act, goal, state, init_goal=feed_goal)
            vadvs.append(mbrews)
            vvfs.append(mbvfs)
            vnlps.append(mbnlps)
            vinr.append(inr)
        return vadvs, vvfs, vnlps, vinr
    
    def step(self, obs, state, init_goal=None):
        if init_goal is None: goal = self.init_goal
        else: goal = init_goal
        goal=np.tile(goal, (obs.shape[0],1))
        feed={self.STATES:[state], self.OBS:obs, self.INITGOALS:goal}
        return self.sess.run([self.act, self.goals, self.pi, self.nstate], feed)
            
    def rewards(self, obs, state, init_goal=None):
        feed={self.STATES:state, self.OBS:obs, self.INITGOALS:init_goal}
        rew = self.sess.run([self.inr], feed)[0]
        return rew
    
    def ifv(self, obs, acts, goal, state, init_goal):
        if self.nhier > 1:
            goal = np.reshape(goal, [obs.shape[0], self.nhier-1, self.maxdim])
        else:
            goal = np.zeros((obs.shape[0], self.nhier - 1, self.maxdim))
        feed={self.STATES:state, self.OBS:obs, self.OLDACTIONS:acts, self.OLDGOALS:goal, self.INITGOALS:init_goal}
        return self.sess.run([self.vf, self.nlp], feed)
    
    def extend(self):
        pass # should create another level of hierarchy dynamically as the look-ahead threshold is surpassed
        # idk how to do this yet
        
class RecurrentFeudalModel(object):
    '''
    General class for organizing feudal networks
    Can train all networks in one sess.run call
    '''
    def __init__(self, policy, env, ob_space, ac_space, neplength=100, nhier=2, max_grad=0.5,
                 ngoal=lambda x:max(8, int(64/(2**x))), recurrent=False, 
                 g=lambda x:1-0.25**(x+1), nhist=lambda x:4**x, val=True,
                 lr=1e-4, vcoef=0.5, encoef=0, nh=64, b=lambda x:0.3 * x,
                 activ=tf.nn.relu, cos=False, fixed_network=False, goal_state=None):
        '''
        INPUTS:
           policy - encoding function for input states
           ob_space - gym.spaces object for obs (inputs)
           ac_space - gym.spaces object for actions (outputs)
           max_grad - gradient clipping threshold (float)
           nhier - number of hierarchical actions (int)
           ngoal - size of goal output at each level (int function)
           recurrent - use recurrent model (Boole)
           g - gamma, discounting parameter (float function)
           nhist - lookahead horizon at each level (int function)
           lr - learning rate (float)
           vcoef - value loss weighting (float)
           encoef - entropy weighting (float)
           nh - hidden layer size (int)
           activ - activation function (TF activation function)
        '''
        
        self.sess = tf.get_default_session() # get session
        self.neplength=neplength
        if fixed_network:
            self.manager_net = FixedManagerNetwork
            self.maxdim = policy.out_shape
            if self.maxdim == None:
                self.maxdim=goal_state.shape[-1]
            self.initdim = goal_state.shape[-1]
        else:
            self.manager_net = RecurrentFeudalNetwork
            self.maxdim = ngoal(1)
            self.initdim = self.maxdim
        self.net = RecurrentFeudalNetwork
        self.recurrent=recurrent
        self.val=val
        self.networks=[] # network array
        self.nhier=nhier # set hierarchy
        beta, gam, tsim, val, nlp, nstate=[],[],[],[],[],[] # hierarchically dependent parameters
        self.init_goal = np.zeros(shape=(self.maxdim)) # 
        self.initial_state = np.zeros(shape=(nhier, nh*2))
        nfeat = ob_space.shape
        
        self.STATES=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier, nh*2))
        self.OBS=tf.placeholder(dtype=tf.float32, shape=(None, None,)+nfeat)
        self.INITGOALS=tf.placeholder(dtype=tf.float32, shape=(None, None, self.initdim))
        self.R=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier))
        self.ADV=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier))
        self.OLDACTIONS=tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.OLDVALUES=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier))
        self.OLDNLPS=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier))
        self.OLDGOALS=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier-1, self.maxdim))
        self.CLIPRANGE=tf.placeholder(dtype=tf.float32, shape=(nhier))
        self.LR=tf.placeholder(dtype=tf.float32)
        
        nbatch=tf.shape(self.OBS)[0]
        inr = [tf.zeros(shape=(nbatch, neplength))]
        ploss=tf.zeros(1)
        vloss = tf.zeros(1)
        entropy = tf.zeros(1)
        goal=[self.INITGOALS]
        
        with tf.variable_scope("common"):
            em_h1=policy(self.OBS)
            
        for t in range(nhier-1):
            beta.append(b(t))
            gam.append(g(t))
            pdtype=make_pdtype(spaces.Box(low=-1, high=1, shape=(ngoal(nhier-t-1),)), r=True)
            if fixed_network: 
                self.networks.append(self.manager_net(goal_state=goal[-1],
                                              state=em_h1, 
                                              recurrent=recurrent, 
                                              nhist=nhist(nhier-t-1),
                                              nh=nh,
                                              policy=policy,
                                              nbatch=nbatch))
            else:
                self.networks.append(self.manager_net(mgoal=goal[t][:,:,:ngoal(nhier-t)],
                                          state=em_h1,
                                          pstate=self.STATES[:,:,t,:],
                                          nhist=nhist(nhier-t-1),
                                          pdtype=pdtype,
                                          nh=nh,
                                          neplength=neplength,
                                          nin=ngoal(nhier-t),
                                          ngoal=ngoal(nhier-t-1),
                                          nbatch=nbatch,
                                          name=nhier-t-1,
                                          manager=True,
                                          val=val))
            if fixed_network:
                goal.append(self.networks[t].aout)
            else:
                goal.append(tf.pad(self.networks[t].aout,
                                tf.constant([[0,0],[0,0],[0,self.maxdim-ngoal(nhier-t-1)]]),
                                mode='CONSTANT'))
            nlp.append(self.networks[t].pd.neglogp(self.OLDGOALS[:,:,t,:ngoal(nhier-t-1)]))
            
            tsim.append(1-self.networks[t].traj_sim)
            inr.append(self.networks[t].inr)
            nstate.append(self.networks[t].nstate)
            if cos:
                adv = self.ADV[:,:,t] * tsim[t]
            else:
                adv = self.ADV[:,:,t] #* tsim[t]
            #tmax = tf.reduce_max(tf.exp(self.OLDNLPS[:,t] - nlp[t]))
            ratio = tf.exp(self.OLDNLPS[:,:,t] - nlp[t])
            pl1 = -adv * ratio
            pl2 = -adv * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE[t], 1.0 + self.CLIPRANGE[t])
            ploss += tf.reduce_mean(tf.maximum(pl1, pl2))
            if val:
                val.append(self.networks[t].vf)
                vclip = self.OLDVALUES[:,:,t] + tf.clip_by_value(val[t]-self.OLDVALUES[:,:,t],
                                      -self.CLIPRANGE[t], self.CLIPRANGE[t]) 
                vl1 = tf.square(val[t] - self.R[:,:,t])
                vl2 = tf.square(vclip - self.R[:,:,t])
                vloss += .5 * tf.reduce_mean(tf.maximum(vl1, vl2))
            entropy += tf.reduce_mean(self.networks[t].pd.entropy())
            
        beta.append(b(nhier-1))
        gam.append(g(nhier-1))
        pdtype = make_pdtype(ac_space, r=True)
        self.nout = self.maxdim if fixed_network else ngoal(1)
        self.networks.append(self.net(mgoal=goal[nhier-1],
                                      state=em_h1,
                                      pstate=self.STATES[:,:,nhier-1,:],
                                      nin=self.nout,
                                      neplength=neplength,
                                      name=0,
                                      nh=nh,
                                      ngoal=ngoal(0),
                                      pdtype=pdtype,
                                      manager=False,
                                      nhist=nhist(0),
                                      nbatch=nbatch,
                                      val=val))
        nlp.append(self.networks[nhier-1].pd.neglogp(self.OLDACTIONS))
        nstate.append(self.networks[nhier-1].nstate)
        adv = self.ADV[:,:,nhier-1]
        ratio = tf.exp(self.OLDNLPS[:,:,nhier-1] - nlp)
        
        pl1 = -adv * ratio
        pl2 = -adv * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE[nhier-1], 1.0 + self.CLIPRANGE[nhier-1])
        ploss += tf.reduce_mean(tf.maximum(pl1, pl2))
        
        if val:
            val.append(self.networks[nhier-1].vf)
            vclip = self.OLDVALUES[:,:,nhier-1]+tf.clip_by_value(val[nhier-1]-
                                  self.OLDVALUES[:,:,nhier-1], -self.CLIPRANGE[nhier-1], self.CLIPRANGE[nhier-1]) 
            vf_losses1 = tf.square(val[nhier-1] - self.R[:,:,nhier-1])
            vf_losses2 = tf.square(vclip - self.R[:,nhier-1])
            vloss += .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.approxkl = .5 * tf.reduce_mean(tf.square(nlp[nhier-1] - self.OLDNLPS[:,:,nhier-1]))
        self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.CLIPRANGE[nhier-1])))
        
        entropy += tf.reduce_mean(self.networks[nhier-1].pd.entropy())
        
        self.pi=self.networks[nhier-1].pi
        if nhier > 1:
            self.goals = tf.transpose(tf.stack(goal[1:]),[2,1,0,3])
        else:
            self.goals=tf.zeros(shape=[nbatch, neplength, nhier-1, self.maxdim])
        if self.recurrent:
            self.nstate=tf.transpose(tf.stack(nstate),[1,0,2])
        else:
            self.nstate=tf.zeros(shape=(nbatch, nhier, nh*2))
        self.inr = tf.transpose(tf.stack(inr),[1,2,0])
        self.nlp = tf.transpose(tf.stack(nlp),[1,2,0])
        if val:
            self.vf = tf.transpose(tf.stack(val),[1,2,0])
        else:
            self.vf = tf.zeros((nbatch, neplength, nhier))
        self.vloss = vloss[0]
        self.beta = np.asarray(beta)
        self.gam = np.asarray(gam)
        self.act = self.networks[nhier-1].aout
        self.ploss = ploss[0]
        self.entropy = entropy[0]
        self.inrmean = tf.reduce_mean(self.inr)
        self.loss = self.ploss + self.vloss * vcoef - self.entropy * encoef
        self.loss_names = ["entropy", "policy loss", "value loss", "approxkl", "clipfrac"]
        optimizer = tf.train.AdamOptimizer(lr)
        if max_grad > 0:
            params = tf.trainable_variables()
            grads = tf.gradients(self.loss, params)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
            grads = list(zip(grads, params))
            self._trainer = optimizer.apply_gradients(grads)
        else:
            self._trainer = optimizer.minimize(self.loss)
        tf.global_variables_initializer().run(session=self.sess)
        
    def train(self, lr,
              cliprange, obs,
              acts, rews,
              advs,
              goals, nlps,
              vfs, states,
              init_goal=None):
        nbatch=obs.shape[0]
        
        if init_goal is None: init_goal = np.tile(self.init_goal,(nbatch,self.neplength,1))
        else: init_goal = init_goal
        
        if isinstance(cliprange, float): cliprange = np.array([cliprange/(2 ** i) for i in range(self.nhier)])
        else: assert cliprange.shape[0] == self.nhier
        
        lr = self.lr if lr == None else lr
        feed={self.STATES:states,
              self.R:rews,
              self.ADV:advs,
              self.OLDACTIONS:acts,
              self.OBS:obs,
              self.INITGOALS:init_goal,
              self.LR:lr, 
              self.CLIPRANGE:cliprange,
              self.OLDGOALS:goals,
              self.OLDNLPS:nlps,
              self.OLDVALUES:vfs}
        return self.sess.run([self.entropy,
                              self.ploss,
                              self.vloss,
                              self.approxkl,
                              self.clipfrac,
                              self._trainer], feed)[:-1]
    
    def av(self, obs, acts, rews, dones, goals, states, init_goal=None):
        vadvs, vvfs, vnlps, vinr =[],[],[],[]
        if init_goal is None:
            init_goal = [None] * len(obs)
            
        for trans_tuple in zip(obs,acts,rews,dones,goals,states,init_goal):
            (ob, act, rew, done, goal, state, init_goal) = trans_tuple
                
            nbatch=1
            trews=np.reshape(np.repeat(rew, self.nhier, -1),(1, self.neplength, self.nhier))
            
            if init_goal is None:
                feed_goal = np.tile(self.init_goal,(nbatch,self.neplength,1,1))
            else:
                feed_goal = np.tile(init_goal,(nbatch,1,1))
            
            inr=self.rewards(ob, state, init_goal=feed_goal)
            mbrews=trews*(1-self.beta) + inr*(self.beta)
            mbvfs, mbnlps =self.ifv(ob, act, goal, state, init_goal=feed_goal)
            vadvs.append(mbrews[0,:])
            vvfs.append(mbvfs[0,:])
            vnlps.append(mbnlps[0,:])
            vinr.append(inr[0,:])
        return vadvs, vvfs, vnlps, vinr
    
    def step(self, obs, state, init_goal=None):
        if init_goal is None: goal = self.init_goal
        else: goal = init_goal
        goal=np.tile(goal, (obs.shape[0],1,1))
        obs = obs[:,np.newaxis,:]
        state = state[np.newaxis,np.newaxis,:,:]
        feed={self.STATES:state, self.OBS:obs, self.INITGOALS:goal}
        return self.sess.run([self.act, self.goals, self.pi, self.nstate], feed)
            
    def rewards(self, obs, state, init_goal):
        state = state[np.newaxis,:,:]
        obs = obs[np.newaxis,:,:]
        feed={self.STATES:state, self.OBS:obs, self.INITGOALS:init_goal}
        rew = self.sess.run([self.inr], feed)[0]
        return rew
    
    def ifv(self, obs, acts, goal, state, init_goal):
        # returns value function and logprob statistics
        
        state = state[np.newaxis,:,:]
        obs = obs[np.newaxis,:,:]
        acts = acts[np.newaxis,:]
        goal = goal[np.newaxis,:,:]
        feed={self.STATES:state, self.OBS:obs, self.OLDACTIONS:acts, self.OLDGOALS:goal, self.INITGOALS:init_goal}
        return self.sess.run([self.vf, self.nlp], feed)
    
    def extend(self):
        pass # should create another level of hierarchy dynamically as the look-ahead threshold is surpassed
        # idk how to do this yet
        
class I2AModel(object):
    def __init__(self, policy, ob_space, ac_space, max_grad, encoef=0.01, 
                 vcoef=0.5, klcoef=0.1, aggregator='concat', traj_len=8, nh=64):
        ob_shape = ob_space.shape
        nfeat = np.prod(ob_shape)
        nactions = ac_space.n
        self.OBS=tf.placeholder(dtype=tf.float32, shape=(None,)+ob_shape)
        self.NEXT_OBS=tf.placeholder(dtype=tf.float32, shape=(None,)+ob_shape) 
        self.R=tf.placeholder(dtype=tf.float32, shape=(None,))
        self.ADV=tf.placeholder(dtype=tf.float32, shape=(None,))
        self.OLDACTIONS=tf.placeholder(dtype=tf.int32, shape=(None,))
        self.OLDVALUES=tf.placeholder(dtype=tf.float32, shape=(None,))
        self.OLDNLPS=tf.placeholder(dtype=tf.float32, shape=(None,))
        self.CLIPRANGE=tf.placeholder(dtype=tf.float32)
        self.LR=tf.placeholder(dtype=tf.float32)#placeholder
        
        flat_obs = tf.flatten(self.OBS)
        flat_next = tf.flatten(self.NEXT_OBS)
        
        pdtype = make_pdtype(ac_space)
        Aggregator = {'concat':tf.layers.flatten}[aggregator]
        
        self.sess = tf.get_default_session()
        
        encoder_params = [nh, tf.contrib.rnn.BasicLSTMCell, None, 'Encoder']
        model_params = [traj_len, 'Env_Model', self.LR, nh,
                        nfeat, vcoef, tf.nn.tanh, max_grad]
        mb_policy_params = ['Model_Based', nh, tf.nn.tanh]
        mf_policy_params = ['Model_Free', nh, tf.nn.tanh]
        
        trajectories = []
        # Imagination Core
        # trajectory encoder
        self.model_free_policy = MFPolicy(flat_obs, pdtype, *mf_policy_params)
        model_free_pd = self.model_free_policy.pd(flat_obs)[0]
        model_free_output = model_free_pd.logits
        
        # model free policy for imagined rollouts
        self.environment_model = EnvironmentModel(flat_obs,
                                 nactions,
                                 self.OLDACTIONS,
                                 flat_next,
                                 self.R,
                                 self.model_free_policy,
                                 *model_params)
        # trajectory rollout model
        trajectories = tf.reshape(tf.stack(self.environment_model.trajectories),
                                  [1,0,2,3])
    
        
        self.encoder = Encoder(trajectories, *encoder_params)
        encoded_traj = self.encoder.encoded_traj
        plan = Aggregator(encoded_traj)
        # encoding
        self.model_based_policy = MBPolicy(flat_obs,
                                           model_free_output,
                                           plan,
                                           pdtype,
                                           *mb_policy_params)
        # model based policy
        self.act = self.model_based_policy.act
        nlp = self.model_based_policy.pd.neglogp(self.OLDACTIONS)
        val = self.model_based_policy.vf
        
        # training terms
        adv = self.ADV
        ratio = tf.exp(self.OLDNLPS - nlp)
        pl1 = -adv * ratio
        pl2 = -adv * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE,
                                      1.0 + self.CLIPRANGE)
        ploss = tf.reduce_mean(tf.maximum(pl1, pl2))
        
        vclip = self.OLDVALUES+tf.clip_by_value(val-self.OLDVALUES,
                                                -self.CLIPRANGE,
                                                self.CLIPRANGE) 
        vf_losses1 = tf.square(val - self.R)
        vf_losses2 = tf.square(vclip - self.R)
        vloss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.approxkl = .5 * tf.reduce_mean(tf.square(nlp - self.OLDNLPS))
        self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.CLIPRANGE)))
        
        cross_entropy = model_free_pd.kl(self.model_based_policy.pd)
        label_entropy = self.model_based_policy.pd.entropy()
        logit_entropy = model_free_pd.entropy()
        entropy = klcoef * cross_entropy - encoef * (label_entropy + logit_entropy)
        
        self.rl_loss = ploss + vloss + entropy
        optimizer = tf.train.AdamOptimizer(self.LR)
        params = tf.trainable_variables()
        grads = tf.gradients(self.rl_loss, params)
        if max_grad is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
        grads = list(zip(grads, params))
        self.rl_trainer = optimizer.apply_gradients(grads)
        
        self.environment_trainer = self.environment_model.trainer
        self.curiosity = self.environment_model.curiosity
        
        self.ploss = ploss
        self.vloss = vloss
        self.entropy = entropy
        self.nlp = nlp
        self.values = val
        
        self.loss_names = ['approxkl', 'clipfrac', 'ploss', 'vloss', 'entropy']
        tf.global_variables_initializer().run(session=self.sess)
        
    def train_rl(self, lr, cliprange, obs, actions, nlps, advs, rewards, values):
        feed = {self.OBS:obs, self.OLDACTIONS:actions,
                self.OLDVALUES:values, self.R:rewards, self.ADV:advs,
                self.OLDNLPS:nlps, self.LR:lr, self.CLIPRANGE:cliprange}
        
        return self.sess.run([self.approxkl, 
                              self.clipfrac, 
                              self.ploss, 
                              self.vloss,
                              self.entropy, 
                              self.rl_trainer], feed)[:-1]
    
    def train_environment(self, obs, actions, next_obs, rewards):
        feed = {self.OBS:obs, self.OLDACTIONS:actions, self.NEXT_OBS:next_obs,
                self.R: rewards}
        self.sess.run([self.environment_trainer], feed)
        return
    
    def step(self, obs):
        feed = {self.OBS:obs}
        return self.sess.run([self.act], feed)
    
    def info(self, obs, actions):
        feed = {self.OBS:obs, self.OLDACTIONS:actions}
        return self.sess.run([self.nlp, self.values], feed)