#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:55:05 2018

@author: matthewszhang
"""
import tensorflow as tf
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def MlpPolicy(tin, nh=64, layers=2, activ=tf.nn.relu, reuse=False):
    flatten = tf.layers.flatten
    embed = flatten(tin)
    for i in range(layers):
        embed = activ(fc(embed, 'em_fc'+str(i), nh=nh, init_scale=np.sqrt(2)))
    tout=embed
    return tout
            
def CnnPolicy(tin, nh=64, layers=1, activ=tf.nn.relu, reuse=False, **conv_kwargs):
    embed = nature_cnn(tin, **conv_kwargs)
    for i in range(layers):
        embed = activ(fc(embed, 'em_fc'+str(i), nh=nh, init_scale=np.sqrt(2)))
    tout = embed
    return tout

def NullPolicy(tin):
    return tf.layers.flatten(tin)