#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:56:20 2018

@author: matthewszhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import argparse

N_NUMS = 3

def encode(state):
    power = 0
    loc = 0
    for i in state:
        loc += i * (N_NUMS ** power)
        power += 1
    return loc

def list_mod(num):
    list_mod = []
    for i in range(1, N_NUMS):
        exp = N_NUMS - i
        val = num // (N_NUMS ** exp)
        num = num % (N_NUMS ** exp)
        list_mod.append(val)
    list_mod.append(num)
    return list_mod

def visualize_goal_progress(csv, out=None):
    if out is None:
        out = os.getcwd()
    
    df = pd.read_csv(csv)
    
    AX_1 = N_NUMS // 2
    AX_2 = N_NUMS - AX_1
    
    inr_arr = np.zeros((N_NUMS ** AX_1, N_NUMS ** AX_2))
    
    for t in range(df.shape[0]):
        for state in df:
            val = df[state][t]
            i_state = list_mod(int(state))
            loc_1, loc_2 = map(encode, (i_state[:AX_1], i_state[AX_1:]))
            inr_arr[loc_1][loc_2] = val
        
        rew_plot = plt.figure()
        ax = rew_plot.subplots()
        ax.imshow(inr_arr, cmap='hot', interpolation='nearest')
        ax.set_yticks(np.arange(N_NUMS ** AX_1))
        ax.set_yticklabels([list_mod(i)[-AX_1:] for i in range(N_NUMS ** AX_1)])
        ax.set_xticks(np.arange(N_NUMS ** AX_2))
        ax.set_xticklabels([list_mod(i)[-AX_2:] for i in range(N_NUMS ** AX_2)])
        rew_plot.savefig(osp.join(out, '{}.png'.format(t)))
        
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def program_arg_parser():
    """
    Create an argparse.ArgumentParser for run_program.py.
    """
    parser = arg_parser()
    #parser.add_argument('--dir', help='fetch directory', type=str, default=None)
    parser.add_argument('-d', '--dir', type=str, default=None, help='fetch directory')
    parser.add_argument('-o', '--out', type=str, default=None, help='outdir')
    return parser
    
if __name__ == '__main__':
    args = program_arg_parser().parse_args()
    visualize_goal_progress(csv=args.dir, out=args.out)
    
    