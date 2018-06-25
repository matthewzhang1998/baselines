#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:37:53 2018

@author: matthewszhang
"""
import os
import numpy as np
import tensorflow as tf

EPOCHS=10
MINIBATCHES=4
LAMBDA=1e0
SAMPLES=10

def log_norm_loss(logit, label, sigma=tf.constant((0.5), dtype=tf.float32)):
    #t1 = tf.subtract(tf.log(sigma), tf.log(tf.constant(np.sqrt(2*np.pi),
    #                                  dtype=tf.float32)), name='loss_1')
    t2 = tf.divide(tf.square(logit - label),(2*tf.square(sigma)), name='loss_2')
    return tf.reduce_mean(tf.reduce_sum(t2, axis=-1, name='loss_sum'), name='avg_loss')

def kl_div_norm(mu_0, sigma_0, mu_1=tf.constant(0.0, dtype=tf.float32), 
                sigma_1=tf.constant(np.log(1+np.exp(-3)), dtype=tf.float32)):
    t1 = tf.log(sigma_1/sigma_0)
    t2 = (tf.square(sigma_0) + tf.square(mu_0 - mu_1))/(2*tf.square(sigma_1))
    return t1 + t2 - 1/2

def mu_hess(rho):
    return tf.square(1/(tf.log(1 + tf.exp(rho))))
    
def rho_hess(rho):
    return 2*tf.exp(2*rho)/tf.square(1+tf.exp(rho)) * mu_hess(rho)

def flatten_all(arr):
    return tf.reshape(arr, [-1])

class BayesianFC(object):
    def __init__(self, X, num_input, num_output, *,
                name='bayes_0', 
                init_mu=tf.initializers.zeros, 
                init_rho=tf.constant_initializer(-3),
                append_vars=False):
        with tf.variable_scope(name):
            w_mu = tf.get_variable("w_mu", shape=[num_input, num_output],
                                     dtype=tf.float32, initializer=init_mu)
            w_rho = tf.get_variable("w_rho", shape=[num_input, num_output],
                                      dtype=tf.float32, initializer=init_rho)
            w_sigma = tf.log(1 + tf.exp(w_rho))
            b_mu = tf.get_variable("b_mu", shape=[1, num_output],
                                   dtype=tf.float32, initializer=init_mu)
            b_rho = tf.get_variable("b_rho", shape=[1, num_output],
                                   dtype=tf.float32, initializer=init_rho)
            b_sigma = tf.log(1 + tf.exp(b_rho))
        
            W_dist = tf.distributions.Normal(loc=w_mu, scale=w_sigma, \
                                             name="W_dist"+name)
            b_dist = tf.distributions.Normal(loc=b_mu, scale=b_sigma, \
                                             name="b_dist"+name)
            
            if append_vars:
                pass
            
            def out():
                return tf.add(tf.matmul(X, W_dist.sample(name="W_sample"+name)), \
                        b_dist.sample(name="b_sample"+name), name="a_"+name)
            def vars():
                return w_mu, w_rho, b_mu, b_rho
            def hessian():
                return mu_hess(w_rho), rho_hess(w_rho), \
                        mu_hess(b_mu), rho_hess(b_rho)
            def kl():
                return kl_div_norm(w_mu, w_sigma), \
                        kl_div_norm(b_mu, b_sigma)
                        
            self.out = out
            self.vars = vars
            self.hessian = hessian
            self.kl = kl
            
class BayesianInferenceNetwork(object):
    def __init__(self, num_features, num_actions, *,
                 num_hidden=64,
                 activ=tf.nn.relu,
                 lr=1e-3,
                 max_grad=0.5,
                 dir=os.getcwd()): 
        num_input = num_features + num_actions
        num_output = num_features   
        self.num_actions = num_actions
        self.X = tf.placeholder(shape=(None, num_input), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None, num_output), dtype=tf.float32)
        self.kl_normalize = tf.placeholder(shape=[], dtype=tf.float32)
        flatten = tf.layers.flatten
        vars, hessian, kl = [], [], []
        sample_loss = []
        with tf.variable_scope("vime"):
            layer_0 = BayesianFC(flatten(self.X), num_input, num_hidden,
                                  name='0')
            a_0 = layer_0.out()
            vars.extend(layer_0.vars())
            hessian.extend(layer_0.hessian())
            kl.extend(layer_0.kl())
            layer_1 = BayesianFC(a_0, num_hidden, num_hidden,
                                  name='1') 
            a_1 = layer_1.out()
            vars.extend(layer_1.vars())
            hessian.extend(layer_1.hessian())
            kl.extend(layer_1.kl())
            layer_2 = BayesianFC(a_1, num_hidden, num_output,
                                  name='2')    
            a_2 = layer_2.out()   
            vars.extend(layer_2.vars())
            hessian.extend(layer_2.hessian())
            kl.extend(layer_2.kl())
            a_norm = tf.nn.l2_normalize(a_2, axis=-1)
            Y_norm = tf.nn.l2_normalize(self.Y, axis=-1)
        
        hessian_flat = tf.concat([flatten_all(hess) for hess in hessian], axis=0)
        for _ in range(SAMPLES):
            sample_loss.append(log_norm_loss(a_norm, Y_norm)) # should only be one element
        pred_loss = tf.reduce_mean(tf.stack(sample_loss))
        grad_loss = tf.gradients(pred_loss, vars)
        flat_grad_loss = tf.concat([flatten_all(grad) for grad in grad_loss], axis=0)
        self.reward_prox = 1/2 * LAMBDA * tf.reduce_sum(tf.square(flat_grad_loss) / hessian_flat)
        kl_loss = tf.reduce_sum(tf.stack([tf.reduce_sum(divergence) for
                                          divergence in kl], axis=-1),
                                          name='kl_sum')/self.kl_normalize
        train_loss = pred_loss + kl_loss
        optimizer = tf.train.AdamOptimizer(lr)
        grads = tf.gradients(train_loss, vars)
        if max_grad is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad)
        grads = list(zip(grads, vars))
        self._trainer = optimizer.apply_gradients(grads)
        self.writer = tf.summary.FileWriter(dir)
        tf.summary.scalar("Prediction_Loss", pred_loss)
        tf.summary.scalar("KL Divergence", kl_loss)
        self.merged_summary = tf.summary.merge_all()
        self.epoch = 0
        self.data_size = 0
        
    def one_hot(self, t):
        depth = self.num_actions
        try:
            t_in = np.array(t)
            length = t_in.shape[0]
        except:
            t_in = np.array([t])
            length = t_in.shape[0]
        t_out = np.zeros((length, depth))
        t_out[np.arange(length), t_in.astype(int)] = 1
        return t_out
        
    def run(self, sess, state, next_state, action):
        one_hot_actions = self.one_hot(action)
        state = state[np.newaxis]
        X_in = np.concatenate((state, one_hot_actions), axis=-1)
        Y_in = next_state[np.newaxis]
        feed = {self.X:X_in, self.Y:Y_in}
        return sess.run([self.reward_prox], feed)
    
    def train(self, sess, states, next_states, actions):
        nbatch = len(states)
        self.data_size += nbatch
        nbatch_train = nbatch//MINIBATCHES
        one_hot_actions = self.one_hot(actions)
        states = np.asarray(states, dtype=np.float32)
        Y = np.asarray(next_states, dtype=np.float32)
        X = np.concatenate((states, one_hot_actions), axis=-1)
        inds = np.arange(nbatch)
        for _ in range(EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                X_in = X[mbinds]
                Y_in = Y[mbinds]
                feed = {self.X:X_in, self.Y:Y_in, self.kl_normalize:self.data_size}
                summary, _ =sess.run([self.merged_summary, self._trainer], feed)
                self.writer.add_summary(summary, self.epoch)
                self.epoch += 1
                
if __name__ == '__main__':
    model = BayesianInferenceNetwork(74, 10)
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    tf.get_default_graph().finalize()
    summary = tf.summary.FileWriter(os.getcwd(), graph=tf.get_default_graph())
    summary.flush()
    
    
    