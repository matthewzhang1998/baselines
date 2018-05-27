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
        
    def disp(self):
        eprewmean = self.progress['eprewmean']
        tmstep = self.progress['serial_timesteps']
        
        rew_plot = plt.gcf()
        plt.plot(tmstep, eprewmean, 'bo--')
        plt.xlabel('timestep')
        plt.ylabel('average reward')
        plt.show()
        rew_plot.savefig(osp.join((self.path), 'reward_over_time.png'))
  
def display(iteration):
    dir = osp.dirname('/home/matthewszhang/logs/iteration-{}/'.format(iteration))
    log = Delog(dir)
    log.disp()
    
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
    return parser    

def main():
    args = program_arg_parser().parse_args()
    display(args.iter)
    
if __name__ == '__main__':
    main()
    
    
    
    
        
        
        
        
    