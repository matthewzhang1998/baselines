#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 21:59:29 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, r=False):
    with tf.variable_scope(scope):
        if r:
            nin = x.get_shape()[-1].value
            w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
            b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
            return tf.einsum('ijk,kl->ijl', x, w)
        
        else:
            nin = x.get_shape()[-1].value
            w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
            b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
            return tf.matmul(x, w)+b
        
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    if len(arr.shape) == 1:
        return arr
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])