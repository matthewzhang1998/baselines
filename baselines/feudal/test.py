#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 01:21:33 2018

@author: matthewszhang
"""
import tensorflow as tf

with tf.Session() as sess:
    rnn = tf.contrib.rnn.LSTMCell(64, state_is_tuple=False)
    zs = rnn.zero_state(batch_size=1, dtype=tf.int32)
    print(sess.run([zs], {})[0].shape)