#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:08:48 2018

@author: matthewszhang
"""
import numpy as np

SEQUENCE = [{'state':[2,1,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[2,1,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[2,1,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[2,2,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[2,2,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[2], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[1],
                          'alu_flag':[0]},
                    {'state':[1,2,0], 'ptr':[2], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,2,2], 'ptr':[2], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,2,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,0,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[2], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[1,0,2], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,0,2], 'ptr':[0], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,0,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,1,2], 'ptr':[1], 'comp_flag':[0],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]},
                    {'state':[0,1,2], 'ptr':[1], 'comp_flag':[1],
                          'stack':[], 'ptr_stack':[], 'gpr_1':[1], 'gpr_2':[0],
                          'alu_flag':[0]}]

DEPTH = 6
ONE_HOT = 6

def decode_index(state):
    re_enc = {'stack':[], 'ptr_stack':[], 'alu_flag':[0]}
    size = DEPTH + ONE_HOT
    n_nums = state.shape[0] // size
    for i in range(0, n_nums):
        start = i * size
        vec = state[start:start+size]
        oh_num = vec[DEPTH:]
        oh_variable = vec[:DEPTH]
        un_hot_num = np.argmax(oh_num, axis=0)
        un_hot_var = np.argmax(oh_variable, axis=0)
        variable = ['state', 'gpr_1', 'gpr_2', 'ptr', 'comp_flag', 'alu_flag'][un_hot_var]
        
        if variable in re_enc:
            re_enc[variable].append(un_hot_num)
        else:
            re_enc[variable] = [un_hot_num]
        
    return SEQUENCE.index(re_enc)

def decode(state):
    re_enc = {'stack':[], 'ptr_stack':[], 'alu_flag':[0]}
    size = DEPTH + ONE_HOT
    n_nums = state.shape[0] // size
    for i in range(0, n_nums):
        start = i * size
        vec = state[start:start+size]
        oh_num = vec[DEPTH:]
        oh_variable = vec[:DEPTH]
        un_hot_num = np.argmax(oh_num, axis=0)
        un_hot_var = np.argmax(oh_variable, axis=0)
        variable = ['state', 'gpr_1', 'gpr_2', 'ptr', 'comp_flag', 'alu_flag'][un_hot_var]
        
        if variable in re_enc:
            re_enc[variable].append(un_hot_num)
        else:
            re_enc[variable] = [un_hot_num]
        
    return re_enc