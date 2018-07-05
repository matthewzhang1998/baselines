#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:17:12 2018

@author: matthewszhang
"""
import tensorflow as tf
import numpy as np
from baselines.feudal.utils import fc

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
        
        def bcs(state, spad, gpad, nhist):
            rew = tf.fill([nbatch], 0.0)
            for t in range(nhist):
                svec = state - spad[nhist-t-1:-(t+1),:]
                gvec = gpad[nhist-t-1:-(t+1),:]
                nsv = tf.nn.l2_normalize(svec, axis=-1)
                ngv = tf.nn.l2_normalize(gvec, axis=-1)
                cos = tf.reduce_sum(tf.multiply(nsv, ngv), axis=-1)
                rew += cos
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv, ngv), axis=-1)
            return sim
        
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
            self.traj_sim = fcs(self.fvec, aout, nhist)
           
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
                cos = tf.reduce_sum(tf.multiply(nsv, ngv), axis=-1)
                rew += cos
            return rew
        
        def fcs(fvec, gvec, nhist):
            nfv = tf.nn.l2_normalize(fvec, axis=-1)
            ngv = tf.nn.l2_normalize(gvec, axis=-1)
            sim = tf.reduce_sum(tf.multiply(nfv, ngv), axis=-1)
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