#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:04:16 2018

@author: matthewszhang
"""
import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import PdType, CategoricalPd
from gym import spaces

class HierCategoricalPd(CategoricalPd): #fixes neglogp
    def __init__(self, logits):
        super(HierCategoricalPd, self).__init__(logits)
    def neglogp(self, x):
        # ignore warning about second order derivatives: need to preserve dims
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)

class HierCategoricalPdType(PdType):
    def __init__(self, nac, npol):
        self.nac = nac
        self.npol = npol
        
    def pdclass(self):
        return HierCategoricalPd
    
    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        temp_pdparam = fc(latent_vector, 'pi', self.nac * self.npol, init_scale=init_scale, init_bias=init_bias)
        pdparam = tf.reshape(temp_pdparam, [-1, self.npol, self.nac])
        return self.pdfromflat(pdparam), pdparam
    
    def param_shape(self):
        return [self.nac * self.npol]

    def sample_shape(self):
        return [self.npol]

    def sample_dtype(self):
        return tf.int32

def make_pdtype_hier(ac_space, npol):
    assert isinstance(ac_space, spaces.Discrete)
    return HierCategoricalPdType(ac_space.n, npol)
    
    