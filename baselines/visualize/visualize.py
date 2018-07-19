#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:08:20 2018

@author: matthewszhang
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as osp

class Delog():
    def __init__(self, path):
        if not osp.exists(path):
            raise Exception("Invalid Path")
            
        self.path = path
        self.monitor = pd.read_csv(osp.join(path, 'monitor.csv'))
        self.progress = pd.read_csv(osp.join(path, 'progress.csv'))
        
    def disp(self, var):
        assert var in self.progress
        eprewmean = self.progress[var]
        tmstep = self.progress['serial_timesteps']
        
        rew_plot = plt.gcf()
        plt.plot(tmstep, eprewmean, 'bo--')
        plt.xlabel('timestep')
        plt.ylabel(var)
        plt.show()
        rew_plot.savefig(osp.join((self.path), 'reward_over_time.png'))
  
def display(iteration, var):
    dir = osp.dirname('/home/matthewszhang/baselines/baselines/feudal/test/test/iteration-{}/'.format(iteration))
    log = Delog(dir)
    log.disp(var)
    
def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def program_arg_parser():
    """
    Create an argparse.ArgumentParser for run_program.py.
    """
    parser = arg_parser()
    parser.add_argument('--iter', help='iteration number', type=int, default='1')
    parser.add_argument('--var', help='variable to visualize', type=str, default='eprewmean')
    return parser    

def main():
    args = program_arg_parser().parse_args()
    display(args.iter, args.var)
    
if __name__ == '__main__':
    main()
    
    
    
    
        
        
        
        
    