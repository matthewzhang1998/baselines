#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:35:25 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf
from baselines.feudal.utils import fc

class Encoder(object):
    def __init__(self, traj, nh=64, cell=tf.contrib.rnn.BasicLSTMCell,
                 initial_state=None, name='Encoder'):
        with tf.variable_scope(name):
            rnn_layer = cell(nh)
            traj_shape = tf.shape(traj)
            flat_traj = tf.reshape(traj, [-1, traj_shape[2], traj_shape[3]])
            feed_traj = tf.reverse(flat_traj, dims=1)
            embed_traj = tf.nn.dynamic_rnn(rnn_layer, feed_traj, 
                                          initial_state=initial_state,
                                          dtype=tf.float32)
            out_traj = tf.reshape(embed_traj, traj_shape)
        return out_traj
    
class EnvironmentModel(object):
    def __init__(self, obs, nactions, actions, nobs, rewards, policy,
                 trajectory_length=8, name='Env_Model', LR=tf.constant(1e-4),
                 nh=64, nout=64, vcoef=0.5, activ = tf.nn.tanh, max_grad=0.5):
        all_trajectories = []
        all_rewards = []
        # rollout graph
        for action in range(nactions):
            action_list = [action]
            rollout_obs = [obs]
            rollout_rews = []
        
            for t in range(trajectory_length):
                x_in = tf.concat(rollout_obs[t], tf.one_hot(action_list[t],
                                 nactions))
                with tf.variable_scope(name):                 
                    ns_1 = activ(fc(x_in, 'ns_1', nh, init_scale=np.sqrt(2)))
                    ns_2 = tf.nn.sigmoid(fc(ns_1, 'ns_2', nout, init_scale=np.sqrt(2)))
                    vf_1 = activ(fc(x_in, 'vf_1', nh, init_scale=np.sqrt(2)))
                    vf_2 = activ(fc(vf_1, 'vf_1', 1, init_scale=np.sqrt(2)))
                rollout_obs.append(ns_2)
                rollout_rews.append(vf_2)
                action = self.pdtype.pdfromlatent(rollout_obs[t+1]).sample()
                action_list.append(action)
            
            all_trajectories.append(tf.stack(rollout_obs[1:]))
            all_rewards.append(tf.stack(rollout_rews))

        # training graph            
        with tf.variable_scope(name):
            X_IN = tf.concat(obs, tf.one_hot(actions, nactions))
            ns_1 = activ(fc(X_IN, 'ns_1', nh, init_scale=np.sqrt(2)))
            ns_2 = tf.nn.sigmoid(fc(ns_1, 'ns_2', nout, init_scale=np.sqrt(2)))
            vf_1 = activ(fc(X_IN, 'vf_1', nh, init_scale=np.sqrt(2)))
            vf_2 = activ(fc(vf_1, 'vf_1', 1, init_scale=np.sqrt(2)))
        
        prediction_loss = tf.mean(tf.sum(tf.square(ns_2 - nobs), axis=-1))
        value_loss = tf.mean(tf.sum(tf.square(vf_2 - rewards), axis=-1))
        env_loss = prediction_loss + vcoef * value_loss
        optimizer = tf.train.AdamOptimizer(LR)
        params = tf.trainable_variables()
        grads = tf.gradients(env_loss, params)
        if max_grad is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
        grads = list(zip(grads, params))
        self.trainer = optimizer.apply_gradients(grads)
        
    
class MBPolicy(object):
    def __init__(self, obs, pi, trajectories, pdtype, name='Model_Based',
                 nh=64, activ=tf.nn.tanh):
        self.pdtype = pdtype
        mb_input = tf.concat([obs, pi, trajectories], axis=-1)
        with tf.variable_scope(name):
            pi_1 = activ(fc(mb_input, scope='pi_1', nh=nh, init_scale=np.sqrt(2)))
            pi_2 = activ(fc(pi_1, scope='pi_2', nh=nh, init_scale=np.sqrt(2)))
            vf_1 = activ(fc(mb_input, scope='vf_1', nh=nh,init_scale=np.sqrt(2)))
            vf_2 = activ(fc(vf_1, scope='vf_2', nh=1, init_scale=np.sqrt(2)))
        self.pd, self.pi = self.pdtype.pdfromlatent(pi_2)
        self.act = self.pd.sample()
        self.vf = vf_2
            
class MFPolicy(object):
    def __init__(self, pdtype, name='Model_Free',
                 nh=64, activ=tf.nn.tanh):
        self.pdtype = pdtype
        self.nh = nh
        self.activ = activ
        self.name=name
        
    def pd(self, obs):
        mb_input = obs
        name = self.name
        activ = self.activ
        nh = self.nh
        with tf.variable_scope(name):
            pi_1 = activ(fc(mb_input, scope='pi_1', nh=nh, init_scale=np.sqrt(2)))
            pi_2 = activ(fc(pi_1, scope='pi_2', nh=nh, init_scale=np.sqrt(2)))
            pd, _ = self.pdtype.pdfromlatent(pi_2)
        return pd