    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:51:54 2018

@author: matthewszhang
"""
import time
import os
import os.path as osp
import tensorflow as tf
import numpy as np
from baselines import logger
from collections import deque
from baselines.common.runners import AbstractEnvRunner
from baselines.feudal.distributions import make_pdtype
from baselines.feudal.utils import fc
from gym import spaces

PATH="tmp/build/graph"

class FeudalNetwork(object):
    '''
    Feudal Agent without recurrency
    '''
    def __init__(self, mgoal, state, pstate, mfvec, pdtype=None, nhist=4, nin=32, ngoal=16,
                 nembed=8, manager=False, nh=64, activ=tf.nn.tanh, name=1, nbatch=1e3, val=True):
        '''
        INPUTS:
            mgoal - goal tensor of supervisor
            state - observation tensor post-embedding
            pstate - recurrent state tensor, ignored in this call
            mfvec - 
        '''
        self.mgoal = mgoal[:,:nin]
        self.state = state
        #state = tf.concat([self.mgoal, self.state], axis=-1)
        nph = nh # policy hidden layer size
        self.manager = manager
        self.name = name
        self.initial_state = None
        nout = ngoal if manager else nh
        self.pdtype = pdtype
        
        with tf.variable_scope("level" + str(self.name)):
            em_h2 = activ(fc(state, 'em_fc2', nh=nout, init_scale=np.sqrt(2)))
            embed_goal = activ(fc(self.mgoal, 'embed', nh=nph, init_scale=np.sqrt(2)))
            pi_h1 = activ(fc(state, 'pi_fc1', nh=nh, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=nph, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(state, 'vf_fc1', nh=nh, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=nh, init_scale=np.sqrt(2)))
            
            pout = embed_goal + pi_h2
            vout = tf.nn.tanh(fc(vf_h2, 'vf', 1))[:,0]
            #pout = pi_h2
        
            self.pd, self.pi = self.pdtype.pdfromlatent(pout, init_scale=0.01)
            aout = self.pd.sample()
            neglogpout = self.pd.neglogp(aout)
            
        self.aout = aout
        self.nlp = neglogpout
        self.nstate = None
        
        A = tf.constant(np.exp(1) - np.exp(-1), dtype=tf.float32)
        B = tf.constant(np.exp(-1)/(np.exp(-1) - np.exp(1)), dtype=tf.float32)
        
        def bcs(state, spad, gpad, nhist):
            rew = tf.fill([nbatch], 0.0)
            for t in range(nhist):
                svec = state - spad[nhist-t-1:-(t+1),:]
                gvec = gpad[nhist-t-1:-(t+1),:]
                nsv = tf.nn.l2_normalize(svec, axis=-1)
                ngv = tf.nn.l2_normalize(gvec, axis=-1)
                exp = tf.exp(tf.reduce_sum(tf.multiply(nsv, ngv), axis=-1))
                rew += exp/A + B
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv, ngv), axis=-1)
            expsim = tf.exp(sim)/tf.constant(np.exp(1)-np.exp(-1),
                         dtype=tf.float32)/A + B
            return expsim
        
        self.vf = vout
        if self.manager:    
            self.mfvec=mfvec[:,:ngoal]
            pad = tf.constant([[nhist,0], [0,0]])
            spad = tf.pad(em_h2, pad, "CONSTANT")
            gpad = tf.pad(aout, pad, "CONSTANT")
            self.inr = 1/nhist * tf.stop_gradient(bcs(em_h2, spad, gpad, nhist))
            
            lstate = em_h2[-1,:]
            rep = tf.reshape(tf.tile(lstate, tf.constant([nhist])),(nhist,nout))
            spadf = tf.concat([em_h2, rep], axis=0)
            self.fvec = spadf[nhist:,] - em_h2
            self.traj_sim = fcs(self.mfvec, aout, nhist)
           
class RecurrentFeudalNetwork(object):
    def __init__(self, mgoal, state, pstate, pdtype=None, nhist=4, nin=32, ngoal=16,
                 nembed=8, manager=False, nh=64, activ=tf.nn.tanh, name=1, nbatch=1,
                 neplength=1e2, cell=tf.contrib.rnn.LSTMCell, val=False):
        self.mgoal=mgoal[:,:,:nin]
        self.state=state
        self.pstate=pstate
        #state = tf.concat([self.mgoal, self.state], axis=-1)
        nph=nh
        self.manager=manager
        self.name=name
        self.initial_state=None
        nout = ngoal if manager else nh
        self.pdtype = pdtype
        
        with tf.variable_scope("level" + str(self.name)):
            em_h2 = tf.nn.l2_normalize(activ(fc(state, 'em_fc2', nh=nout, init_scale=np.sqrt(2), r=True)))
            embed_goal = activ(fc(self.mgoal, 'embed', nh=nph, init_scale=np.sqrt(2), r=True))
            
            cell = cell(nh, state_is_tuple=False)
            a_h1, nstate = tf.nn.dynamic_rnn(cell, inputs=state, 
                                             initial_state=pstate[:,0,:])
            c_h1 = activ(a_h1)
            pi_h2 = activ(fc(c_h1, 'pi_fc2', nh=nph, init_scale=np.sqrt(2), r=True))
            vf_h2 = activ(fc(c_h1, 'vf_fc2', nh=nh, init_scale=np.sqrt(2), r=True))                
            vout = tf.nn.tanh(fc(vf_h2, 'vf', 1, r=True))[:,:,0]

            pout = tf.multiply(embed_goal, pi_h2)
            #pout = pi_h2
        
            self.pd, self.pi = self.pdtype.pdfromlatent(pout, init_scale=0.01)
            aout = self.pd.sample()
            neglogpout = self.pd.neglogp(aout)
            
        self.nstate = nstate
        self.aout = aout
        self.nlp = neglogpout
        
        def bcs(state, spad, gpad, nhist):
            rew = tf.zeros(shape=(nbatch,neplength), dtype=tf.float32)
            for t in range(nhist):
                svec = state - spad[:,nhist-t-1:-(t+1),:]
                gvec = gpad[:,nhist-t-1:-(t+1),:]
                nsv = tf.nn.l2_normalize(svec, axis=-1)
                ngv = tf.nn.l2_normalize(gvec, axis=-1)
                cos = tf.exp(tf.reduce_sum(tf.multiply(nsv, ngv), axis=-1))/ \
                        tf.constant(np.exp(1), dtype=tf.float32)
                rew += cos
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv, ngv), axis=-1)
            sim = tf.exp(sim)/tf.constant(np.exp(1), dtype=tf.float32)
            return sim
        
        self.vf = vout
        if self.manager:
            pad = tf.constant([[0,0],[nhist,0], [0,0]])
            spad = tf.pad(em_h2, pad, "CONSTANT")
            gpad = tf.pad(aout, pad, "CONSTANT")
            
            self.inr = 1/nhist * tf.stop_gradient(bcs(em_h2, spad, gpad, nhist))
            
            lstate = em_h2[:,-1,:]
            rep = tf.reshape(tf.tile(lstate, [nbatch, nhist]),
                             (nbatch,nhist,nout))
            spadf = tf.concat([em_h2, rep], axis=1)
            self.fvec = spadf[:,nhist:,] - em_h2
            self.traj_sim = fcs(self.fvec, aout, nhist)
    
class FeudalModel(object):
    '''
    General class for organizing feudal networks
    Can train all networks in one sess.run call
    '''
    def __init__(self, policy, ob_space, ac_space, nhier=2, max_grad=0.5,
                 ngoal=lambda x:max(8, int(64/(2**x))), recurrent=False, 
                 g=lambda x:1-0.25**(x+1), nhist=lambda x:4**x, val=True,
                 lr=1e-4, vcoef=0.5, encoef=0, nh=64, b=lambda x:0.3 * x,
                 activ=tf.nn.relu):
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
        self.net = FeudalNetwork # get single network object
        self.recurrent=recurrent
        self.val=val
        self.networks=[] # network array
        self.nhier=nhier # set hierarchy
        beta, gam, tsim, val, fvecs, nlp, nstate=[],[],[],[],[],[],[] # hierarchically dependent parameters
        self.maxdim = ngoal(1) # max goal dimensions for uniform input/output
        self.init_goal = np.zeros(shape=(self.maxdim)) # 
        self.initial_state = np.zeros(shape=(nhier, nh*2))
        nfeat = ob_space.shape
        
        self.STATES=tf.placeholder(dtype=tf.float32, shape=(None, nhier, nh*2))
        self.OBS=tf.placeholder(dtype=tf.float32, shape=(None,)+nfeat)
        self.INITGOALS=tf.placeholder(dtype=tf.float32, shape=(None, self.maxdim))
        self.VECS=tf.placeholder(dtype=tf.float32, shape=(None, nhier-1, self.maxdim))
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
            
            self.networks.append(self.net(mgoal=goal[t][:,:ngoal(nhier-t)],
                                          state=em_h1,
                                          mfvec=self.VECS[:,t,:],
                                          pstate=self.STATES[:,t,:],
                                          nhist=nhist(nhier-t),
                                          pdtype=pdtype,
                                          nin=ngoal(nhier-t),
                                          ngoal=ngoal(nhier-t-1),
                                          nbatch=nbatch,
                                          name=nhier-t-1,
                                          manager=True,
                                          val=val))
            goal.append(tf.pad(self.networks[t].aout,
                                tf.constant([[0,0],[0,self.maxdim-ngoal(nhier-t-1)]]),
                                mode='CONSTANT'))
            nlp.append(self.networks[t].pd.neglogp(self.OLDGOALS[:,t,:ngoal(nhier-t-1)]))
            fvecs.append(tf.pad(self.networks[t].fvec, 
                                tf.constant([[0,0],[0,self.maxdim-ngoal(nhier-t-1)]]),
                                mode='CONSTANT'))
            
            tsim.append(1-self.networks[t].traj_sim)
            inr.append(self.networks[t].inr)
            nstate.append(self.networks[t].nstate)
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
        self.networks.append(self.net(mgoal=goal[nhier-1],
                                      state=em_h1,
                                      pstate=self.STATES[:,nhier-1,:],
                                      mfvec=None, 
                                      nin=ngoal(1),
                                      name=0,
                                      ngoal=ngoal(0),
                                      pdtype=pdtype,
                                      manager=False,
                                      nhist=nhist(1),
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
            self.fvecs=tf.transpose(tf.stack(fvecs),[1,0,2])   
            self.goals = tf.transpose(tf.stack(goal[1:]),[1,0,2])
        else:
            self.fvecs=tf.zeros(shape=[nbatch, nhier-1, self.maxdim])
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
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        if max_grad is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
        grads = list(zip(grads, params))
        self._trainer = optimizer.apply_gradients(grads)
        tf.global_variables_initializer().run(session=self.sess)
        
    def train(self, lr,
              cliprange, obs,
              acts, rews,
              advs, vecs,
              goals, nlps,
              vfs, states,
              eplens=None, init_goal=None):
        nbatch=obs.shape[0]
        if init_goal is None: init_goal = np.tile(self.init_goal,(nbatch,1))
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
              self.VECS:vecs, 
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
        vadvs, vvecs, vvfs, vnlps, vinr =[],[],[],[],[]
        for (ob, act, rew, done, goal,state) in zip(obs,acts,rews,dones,goals,states):
            nbatch=ob.shape[0]
            trews=np.reshape(np.repeat(rew, self.nhier, -1),(nbatch, self.nhier))
            if init_goal is None:
                feed_goal = np.tile(self.init_goal,(nbatch,1))
            else:
                feed_goal = init_goal
            inr=self.rewards(ob, state, init_goal=feed_goal)
            mbrews=trews*(1-self.beta) + inr*(self.beta)
            mbfvecs, mbvfs, mbnlps =self.ifv(ob, act, goal, state, init_goal=feed_goal)
            vadvs.append(mbrews)
            vvecs.append(mbfvecs)
            vvfs.append(mbvfs)
            vnlps.append(mbnlps)
            vinr.append(inr)
        return vadvs, vvecs, vvfs, vnlps, vinr
    
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
        feed={self.STATES:state, self.OBS:obs, self.OLDACTIONS:acts, self.OLDGOALS:goal, self.INITGOALS:init_goal}
        return self.sess.run([self.fvecs, self.vf, self.nlp], feed)
    
    def extend(self):
        pass # should create another level of hierarchy dynamically as the look-ahead threshold is surpassed
        # idk how to do this yet
        
class RecurrentFeudalModel(object):
    '''
    General class for organizing feudal networks
    Can train all networks in one sess.run call
    '''
    def __init__(self, policy, ob_space, ac_space, neplength=100, nhier=2, max_grad=0.5,
                 ngoal=lambda x:max(8, int(64/(2**x))), recurrent=False, 
                 g=lambda x:1-0.25**(x+1), nhist=lambda x:4**x, val=True,
                 lr=1e-4, vcoef=0.5, encoef=0, nh=64, b=lambda x:0.3 * x,
                 activ=tf.nn.relu):
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
        self.net = RecurrentFeudalNetwork 
        self.recurrent=recurrent
        self.val=val
        self.networks=[] # network array
        self.nhier=nhier # set hierarchy
        beta, gam, tsim, val, nlp, nstate=[],[],[],[],[],[] # hierarchically dependent parameters
        self.maxdim = ngoal(1) # max goal dimensions for uniform input/output
        self.init_goal = np.zeros(shape=(self.maxdim)) # 
        self.initial_state = np.zeros(shape=(nhier, nh*2))
        nfeat = ob_space.shape
        
        self.STATES=tf.placeholder(dtype=tf.float32, shape=(None, None, nhier, nh*2))
        self.OBS=tf.placeholder(dtype=tf.float32, shape=(None, None,)+nfeat)
        self.INITGOALS=tf.placeholder(dtype=tf.float32, shape=(None, None, self.maxdim))
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
            self.networks.append(self.net(mgoal=goal[t][:,:,:ngoal(nhier-t)],
                                          state=em_h1,
                                          pstate=self.STATES[:,:,t,:],
                                          nhist=nhist(nhier-t),
                                          pdtype=pdtype,
                                          neplength=neplength,
                                          nin=ngoal(nhier-t),
                                          ngoal=ngoal(nhier-t-1),
                                          nbatch=nbatch,
                                          name=nhier-t-1,
                                          manager=True,
                                          val=val))
            goal.append(tf.pad(self.networks[t].aout,
                                tf.constant([[0,0],[0,0],[0,self.maxdim-ngoal(nhier-t-1)]]),
                                mode='CONSTANT'))
            nlp.append(self.networks[t].pd.neglogp(self.OLDGOALS[:,:,t,:ngoal(nhier-t-1)]))
            
            tsim.append(1-self.networks[t].traj_sim)
            inr.append(self.networks[t].inr)
            nstate.append(self.networks[t].nstate)
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
        self.networks.append(self.net(mgoal=goal[nhier-1],
                                      state=em_h1,
                                      pstate=self.STATES[:,:,nhier-1,:],
                                      nin=ngoal(1),
                                      neplength=neplength,
                                      name=0,
                                      ngoal=ngoal(0),
                                      pdtype=pdtype,
                                      manager=False,
                                      nhist=nhist(1),
                                      nbatch=nbatch,
                                      val=val))
        nlp.append(self.networks[nhier-1].pd.neglogp(self.OLDACTIONS))
        nstate.append(self.networks[nhier-1].nstate)
        adv = self.ADV[:,:,nhier-1]
        ratio = tf.exp(self.OLDNLPS[:,:,nhier-1] - nlp[nhier-1])
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
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        if max_grad is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
        grads = list(zip(grads, params))
        self._trainer = optimizer.apply_gradients(grads)
        tf.global_variables_initializer().run(session=self.sess)
        
    def train(self, lr,
              cliprange, obs,
              acts, rews,
              advs,
              goals, nlps,
              vfs, states,
              eplens=None, init_goal=None):
        nbatch=obs.shape[0]
        if init_goal is None: init_goal = np.tile(self.init_goal,(nbatch,self.neplength,1))
        else: init_goal = np.tile(init_goal,(nbatch,1,1))
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
        vadvs, vvecs, vvfs, vnlps, vinr =[],[],[],[],[]
        
        for trans_tuple in zip(obs,acts,rews,dones,goals,states):
            (ob, act, rew, done, goal, state) = trans_tuple
                
            nbatch=1
            trews=np.reshape(np.repeat(rew, self.nhier, -1),(1, self.neplength, self.nhier))
            
            if init_goal is None:
                feed_goal = np.tile(self.init_goal,(nbatch,self.neplength,1))
            else:
                feed_goal = np.tile(init_goal,(nbatch,1,1))
            
            inr=self.rewards(ob, state, init_goal=feed_goal)
            mbrews=trews*(1-self.beta) + inr*(self.beta)
            mbvfs, mbnlps =self.ifv(ob, act, goal, state, init_goal=feed_goal)
            vadvs.append(mbrews[0,:])
            vvfs.append(mbvfs[0,:])
            vnlps.append(mbnlps[0,:])
            vinr.append(inr[0,:])
        return vadvs, vvecs, vvfs, vnlps, vinr
    
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
        state = state[np.newaxis,:,:]
        obs = obs[np.newaxis,:,:]
        acts = acts[np.newaxis,:]
        goal = goal[np.newaxis,:,:]
        feed={self.STATES:state, self.OBS:obs, self.OLDACTIONS:acts, self.OLDGOALS:goal, self.INITGOALS:init_goal}
        return self.sess.run([self.vf, self.nlp], feed)
    
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
        
        # not sure why but one step is required at the beginning
        actions, goal, pi, self.states = self.model.step(self.obs, self.states)
        self.states = self.states[0]
        self.obs[:], rewards, self.dones, _ = self.env.step(actions) # perform 1 step, safety
        
    def run(self):
        mb_obs, mb_rewards, mb_goals, mb_actions, mb_dones, mb_pi, mb_states = [],[],[],[],[],[],[]
        epinfos = []
        for i in range(self.nsteps):
            mb_states.append(self.states)
            actions, goal, pi, self.states = self.model.step(self.obs, self.states)
            self.states = self.states[0]
            mb_obs.append(self.obs.copy())
            mb_goals.append(goal[0])
            mb_pi.append(pi)
            mb_actions.append(actions[0])
            mb_dones.append(self.dones)
            if self.dones[0] == True: # lose parallelism again
                self.states = self.model.initial_state
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_pi = np.asarray(mb_pi, dtype=np.float32)
        mb_goals = np.asarray(mb_goals, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_states = np.asarray(mb_states, dtype=np.float32)
        # lose environment parallelism here -> need to fix
        
        return (*map(sf01, (mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi, mb_goals)), mb_states, epinfos)

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
    nsteps = rews.shape[0]
    nextvalues=vals[:,-1:,]
    for t in reversed(range(nsteps)):
        if t == nsteps - 1:
            nextnonterminal = 0
            nextvalues = 0 # assume last is terminal -> won't be too significant unless tstep is large
        else:
            nextnonterminal = 1.0
            nextvalues = vals[:,t+1,:]
        delta = rews[:,t] + gam * nextvalues * nextnonterminal - vals[:,t]
        mb_advs[:,t] = lastgaelam = delta + gam * lam * nextnonterminal * lastgaelam
        
    mb_returns = mb_advs + vals
    return mb_returns, mb_advs

def safe_vstack(arr, dim1):
    assert arr
    shape = arr[0].shape
    return np.reshape(np.vstack(arr), (dim1,) + shape)
    
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
    neplength = max_len
    ob_space = env.observation_space
    ac_space = env.action_space
    assert (nenvs * nsteps)%max_len == 0
    if recurrent:
        nbatch = (nenvs * nsteps)//max_len
        print(nbatch, nmb)
    else:
        nbatch = (nenvs * nsteps)
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
              val=val)
    else:
        make_model = lambda : FeudalModel(policy, ob_space, ac_space, max_grad=mgn,
              ngoal=ng, recurrent=recurrent, g=gamma, nhist=nh, b=beta, nhier=nhier,
              val=val)
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
    nupdates = tsteps//nbatch
    
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
            mean_inr = np.mean(inrs)
            if not val:
                vre = vre * val_temp + np.mean(rewards, axis=0) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nsteps), [nsteps, nhier])
            rewards, advs = mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            print(np.mean(goals))
            actions = actions.flatten() #safety
            inds = np.arange(nbatch)
            for _ in range(noe):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, actions, rewards, advs, vecs, goals, nlps, vfs, states))    
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else: # recurrent version
            rewards, vecs, vfs, nlps, inrs = model.av(obs, actions, rewards, dones, goals, states)
            print(len(nlps), nlps[0].shape)
            pre_vars = (obs,actions,rewards,dones,goals,nlps,vfs,states,inrs) 
            map_vars = (safe_vstack(arr, nbatch) for arr in pre_vars)
            (obs,actions,rewards,dones,goals,nlps,vfs,states,inrs) = map_vars
            
            if not val:
                vre = vre * val_temp + np.apply_over_axes(np.mean, rewards, [0,1]) * (1-val_temp)
                vfs = np.reshape(np.repeat(vre, nbatch*neplength), [nbatch, neplength, nhier])
            rewards, advs = recurrent_mcret(actions, rewards, dones, vfs, lam=lam, gam=model.gam)
            
            feed_vars = (obs, actions, rewards, advs, goals, nlps, vfs, states)
            mean_inr = np.mean(inrs)
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
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv('intrinsic_reward', mean_inr)
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

    

    