#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:17:12 2018

@author: matthewszhang
"""
import tensorflow as tf
import numpy as np
from baselines.feudal.utils import fc
from baselines.feudal.distributions import ConstantPd

PATH="tmp/build/graph"


class FixedActorNetwork(object):
    '''
    sets up a network which returns dummy actions,
    reduces propagation time and preserves model structure
    '''
    def __init__(self, mgoal, state, pstate, nh, recurrent, *args, **kwargs):
        self.dummy_actions = tf.zeros_like(state, dtype=tf.float32)
        self.pd = ConstantPd(self.dummy_actions)
        self.pi = self.dummy_actions # this is a dummy variable
        
        self.aout = tf.reduce_sum(self.pd.sample(), axis=-1)
        self.nlp = self.pd.neglogp(self.aout)
        if recurrent:
            self.nstate = tf.zeros(shape=(tf.shape(state)[0], nh*2), dtype=tf.float32)
        else:
            self.nstate = None
        
        # unused
        self.fvec = None
        self.inr = None

class FixedManagerNetwork(object):
    def __init__(self, goal_state, state, recurrent, nhist, nh, policy, nbatch, 
                 *args, **kwargs):
        self.state = state
        self.nhist = nhist
        
        # Do not embed state
        with tf.variable_scope("common", reuse=tf.AUTO_REUSE):
            em_h2 = policy(goal_state)
        self.pd = ConstantPd(em_h2 - state)
        self.aout = self.pd.sample()
        self.nlp = self.pd.neglogp(self.aout)
        self.vf = tf.constant([0.0])
        
        def bcs(state, spad, gpad, nhist, axis=0):
            rew = tf.fill([nbatch], 0.0)
            for t in range(nhist):
                if axis==1:                    
                    svec = state - spad[:,nhist-t-1:-(t+1),:]
                    gvec = gpad[:,nhist-t-1:-(t+1),:]
                else:
                    svec = state - spad[nhist-t-1:-(t+1),:]
                    gvec = gpad[nhist-t-1:-(t+1),:]
                nsv = tf.nn.l2_normalize(svec, axis=-1)
                ngv = tf.nn.l2_normalize(gvec, axis=-1)
                cosine_dist = tf.reduce_sum(tf.multiply(nsv,ngv), axis=-1)
                #rew = tf.Print(rew, [cosine_dist, tf.shape(cosine_dist)])
                rew += cosine_dist
            print("bcs shape: {}".format(rew.get_shape()))
            
            return rew

        def sparse_bcs(state, spad, gpad, nhist, axis=0):
            rew = tf.fill([nbatch], 0.0)
            for t in range(nhist):
                if axis==1:                    
                    svec = state - spad[:,nhist-t-1:-(t+1),:]
                    gvec = gpad[:,nhist-t-1:-(t+1),:]
                else:
                    svec = state - spad[nhist-t-1:-(t+1),:]
                    gvec = gpad[nhist-t-1:-(t+1),:]
                delta_gs = tf.to_float(tf.equal(tf.reduce_mean(tf.to_float(tf.equal(svec, gvec)), axis=-1), 1.))
                zero_mask = tf.to_float(tf.equal(tf.reduce_mean(tf.to_float(tf.equal(tf.zeros_like(gvec), gvec)), axis=-1), 1.))
                delta_gs *= (1.-zero_mask)
                #rew = tf.Print(rew, [delta_gs, tf.shape(delta_gs)])
                rew += delta_gs
                #rew += tf.to_float(tf.equal(tf.reduce_mean(tf.to_float(tf.equal(svec, gvec)), axis=-1), 1.))
            print("sparse_bcs shape: {}".format(rew.get_shape()))
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv,ngv), axis=-1)
            return sim
        
        if recurrent:
            pad = tf.constant([[0,0],[nhist,0], [0,0]])
            spad = tf.pad(self.state, pad, "CONSTANT")
            gpad = tf.pad(self.aout, pad, "CONSTANT")
            self.inr = 1/nhist * tf.stop_gradient(bcs(self.state, spad, gpad, nhist, axis=1))        
        else:
            pad = tf.constant([[nhist,0], [0,0]])
            spad = tf.pad(self.state, pad, "CONSTANT")
            gpad = tf.pad(self.aout, pad, "CONSTANT")
            self.inr = 1/nhist * tf.stop_gradient(bcs(self.state, spad, gpad, nhist, axis=0))
            self.sparse_inr = 1/nhist * (sparse_bcs(self.state, spad, gpad, nhist, axis=0))
        self.fvec = tf.zeros_like(self.state)
        self.traj_sim = tf.reduce_sum(tf.zeros_like(self.state), axis=-1)
        if recurrent:
            self.nstate = tf.zeros(shape=(tf.shape(self.state)[0], nh*2), dtype=tf.float32)
        else:
            self.nstate = None # fix this later
            
        self.nout = goal_state.shape[-1]

class FeudalNetwork(object):
    '''
    Feudal Agent without recurrency
    '''
    def __init__(self, mgoal, state, pstate, pdtype=None, nhist=4, nin=32, ngoal=16, recurrent=0,
                 nembed=8, manager=False, nh=64, activ=tf.nn.relu, name=1, nbatch=1e3, val=True):
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
        self.nout = nout
        self.pdtype = pdtype
        
        with tf.variable_scope("level" + str(self.name)):
            em_h2 = activ(fc(state, 'em_fc2', nh=nout, init_scale=np.sqrt(2)))
            embed_goal = fc(self.mgoal, 'embed', nh=nph, init_scale=np.sqrt(2))
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
        print(self.nlp)
        self.nstate = None
        
        def bcs(state, spad, gpad, nhist):
            rew = tf.fill([nbatch], 0.0)
            for t in range(nhist):
                svec = state - spad[nhist-t-1:-(t+1),:]
                gvec = gpad[nhist-t-1:-(t+1),:]
                nsv = tf.nn.l2_normalize(svec, axis=-1)
                ngv = tf.nn.l2_normalize(gvec, axis=-1)
                rew += tf.reduce_sum(tf.multiply(nsv,ngv), axis=-1)
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv,ngv), axis=-1)
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
            self.fvec = spadf[nhist:,] - em_h2
            self.traj_sim = fcs(self.fvec, aout, nhist)
           
class RecurrentFeudalNetwork(object):
    def __init__(self, mgoal, state, pstate, pdtype=None, nhist=4, nin=32, ngoal=16,
                 nembed=8, manager=False, nh=64, activ=tf.nn.relu, name=1, nbatch=1,
                 neplength=1e2, cell=tf.contrib.rnn.LSTMCell, val=False, recurrent=1):
        self.mgoal=mgoal[:,:,:nin]
        self.state=state
        self.pstate=pstate
        #state = tf.concat([self.mgoal, self.state], axis=-1)
        nph=nh
        self.manager=manager
        self.name=name
        nout = ngoal if manager else nh
        self.pdtype = pdtype
        
        with tf.variable_scope("level" + str(self.name)):
            em_h2 = activ(fc(state, 'em_fc2', nh=nout, init_scale=np.sqrt(2), r=True))
            embed_goal = activ(fc(self.mgoal, 'embed', nh=nph, init_scale=np.sqrt(2), r=True))
            
            cell = cell(nh, state_is_tuple=False)
            a_h1, nstate = tf.nn.dynamic_rnn(cell, inputs=state, 
                                             initial_state=pstate[:,0,:])
            c_h1 = activ(a_h1)
            pi_h2 = activ(fc(c_h1, 'pi_fc2', nh=nph, init_scale=np.sqrt(2), r=True))
            vf_h2 = activ(fc(c_h1, 'vf_fc2', nh=nh, init_scale=np.sqrt(2), r=True))                
            vout = tf.nn.tanh(fc(vf_h2, 'vf', 1, r=True))[:,:,0]

            pout = embed_goal + pi_h2
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
                rew += tf.reduce_sum(tf.multiply(nsv,ngv), axis=-1)
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv,ngv), axis=-1)
            return sim
        
        self.vf = vout
        if self.manager:
            pad = tf.constant([[0,0],[nhist,0], [0,0]])
            spad = tf.pad(em_h2, pad, "CONSTANT")
            gpad = tf.pad(aout, pad, "CONSTANT")
            
            self.inr = 1/nhist * tf.stop_gradient(bcs(em_h2, spad, gpad, nhist))
            
            lstate = em_h2[:,-1,:]
            rep = tf.reshape(tf.tile(lstate, [nhist,1]),
                             (nbatch,nhist,nout))
            spadf = tf.concat([em_h2, rep], axis=1)
            self.fvec = spadf[:,nhist:,] - em_h2
            self.traj_sim = fcs(self.fvec, aout, nhist)
