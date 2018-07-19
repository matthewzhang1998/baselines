#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:14:05 2018

@author: matthewszhang
"""
import math
import argparse
import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

EXCEPTIONS = ['intrinsic_rewards_1', 'intrinsic_rewards_2']
def remove_duplicate(list_csv, list_json):
    assert len(list_csv)==len(list_json)
    
    unique_csv = []
    already_seen = []
    for i in range(len(list_csv)):
        with open(list_json[i]) as json_data:
            d = json.load(json_data)
            if d in already_seen:
                continue
            else:
                try:
                    pd.read_csv(list_csv[i])
                    already_seen.append(d)
                    unique_csv.append(list_csv[i])
                except:
                    continue
    return unique_csv, already_seen

def safe_avg(a, b, c):
    c += 1
    if math.isnan(b):
        return a, c-1
        
    return ((c-1)*a+b)/c, c
    
def aggregate(tuple_a, list_b):
    list_a, counter_a = tuple_a
    len_b = len(list_b)
    
    for i in range(min(len_b, len(list_a))):
        list_a[i], counter_a[i] = \
        safe_avg(list_a[i], list_b[i], counter_a[i])
    return (list_a, counter_a)

def moving_avg(list_a, window):
    list_a_supplement = np.concatenate(([0]*(window+1), list_a[1:]))
    list_b = []
    for i in range(len(list_a)):
        list_b.append(sum(list_a_supplement[i:window+i])/window)
    return list_b

N_LINE = 5

def visualize_all(dir=os.getcwd(), subdir=None, var='eprewmean', out=None, 
                  wt_var=0, window=10):
    if subdir is None:
        subdir = [None]
        
    for find_dir in subdir:
        temp_dir = osp.join(dir, find_dir)
        
        progress = [f for p in os.walk(temp_dir) \
                    for f in glob(osp.join(p[0], 'progress.csv'))]
        vars = [v for p in os.walk(temp_dir) \
                    for v in glob(osp.join(p[0], 'vars.json'))]
        
        progress, vars = remove_duplicate(progress, vars)
        
        if out is not None:
            dir = osp.join(dir, out)
        
        if var == ['__all__']: # deprecated
            print("Plotting all is deprecated")
            csv = progress[0]
            try:
                data = pd.read_csv(csv)
                valid_vars = list(data.columns.values)
            except:
                raise ValueError("First CSV must have data")
            for i in range(len(progress)):
                csv = progress[i]
                vardict = vars[i]
                try:
                    data = pd.read_csv(csv)
                except:
                    continue
                
                lines = []
                n_vars = len(vardict)
                start = 0
                end = start + N_LINE
                items = list(vardict.items())
                while(end < n_vars):
                    lines.append(str(items[start:end]))
                    start = end
                    end = start + N_LINE
                lines.append(str(items[start:]))
                
                n_write = len(lines)
                
                rew_plot = plt.figure()
                
                if wt_var:
                    n_plots = len(var)+1
                else:
                    n_plots = len(var)
                
                for (index,var) in enumerate(valid_vars):
                    try:
                        yvar = data[var]
                    except:
                        continue
                    ax = rew_plot.add_subplot(n_plots,1,index+1)
                    
                    tmstep = data['serial_timesteps']

                    ax.set_ylabel('{}'.format(var))
                    ax.plot(tmstep,yvar, c='r')
                   
                if wt_var:
                    ax = rew_plot.add_subplot(n_plots,1,n_plots)
                    for j in range(n_write):
                        ax.text(.05, .1 * (n_write-j), lines[j], fontsize=6)
                
                if not osp.exists(osp.join(dir, out)):
                    os.makedirs(osp.join(dir, out))
                
                plt.savefig(osp.join(dir,'{}/{}.png'.format(out,csv)))
                rew_plot.clear()
                    
        else:
            for i in range(len(progress)):
                csv = progress[i]
                vardict = vars[i]
                try:
                    data = pd.read_csv(csv)
                except:
                    continue
                
                lines = []
                n_vars = len(vardict)
                start = 0
                end = start + N_LINE
                items = list(vardict.items())
                while(end < n_vars):
                    lines.append(str(items[start:end]))
                    start = end
                    end = start + N_LINE
                lines.append(str(items[start:]))
                
                n_write = len(lines)
                
                rew_plot = plt.figure(figsize=(6,12))
                
                if wt_var:
                    n_plots = len(var)+1
                else:
                    n_plots = len(var)
                
                for index,v in enumerate(var):
                    yvar = data[v]
                    tmstep = data['serial_timesteps']
                    ax = rew_plot.add_subplot(n_plots,1,index+1)
                    
                    if index < n_plots - 2:
                        ax.set_xticks([])
                    
                    ax.set_ylabel('{}'.format(v), fontsize=8)
                    to_plot = moving_avg(yvar, window)
                    ax.plot(tmstep,to_plot, c='r')
                
                if wt_var:
                    ax = rew_plot.add_subplot(n_plots,1,n_plots)
                    ax.set_frame_on(0)
                    ax.set_axis_off()
                    for j in range(n_write):
                        ax.text(.05, .1 * (n_write-j), lines[j], fontsize=6)
                
                if not osp.exists(osp.join(dir, out)):
                    os.makedirs(osp.join(dir, out))
                
                plt.savefig(osp.join(dir,'{}/{}.png'.format(out,i)))
                plt.close()
                
def visualize_multi(dir=os.getcwd(), subdir=None, var='intrinsic_reward', out=None):    
    if subdir is None:
        subdir = [None]
        
    for find_dir in subdir:
        temp_dir = osp.join(dir, find_dir)
        
        progress = [f for p in os.walk(temp_dir) \
                    for f in glob(osp.join(p[0], 'progress.csv'))]
        vars = [v for p in os.walk(temp_dir) \
                    for v in glob(osp.join(p[0], 'vars.json'))]
        
        progress, vars = remove_duplicate(progress, vars)    
        for i in range(len(progress)):
            csv = progress[i]
            vardict = vars[i]
            try:
                data = pd.read_csv(csv)
            except:
                continue
            
            lines = []
            n_vars = len(vardict)
            start = 0
            end = start + N_LINE
            items = list(vardict.items())
            while(end < n_vars):
                lines.append(str(items[start:end]))
                start = end
                end = start + N_LINE
            lines.append(str(items[start:]))
            
            n_write = len(lines)
            
            rew_plot = plt.figure(figsize=(6,12))
            
            if wt_var:
                n_plots = len(var)+1
            else:
                n_plots = len(var)
            
            for index,v in enumerate(var):
                yvar = data[v]
                tmstep = data['serial_timesteps']
                ax = rew_plot.add_subplot(n_plots,1,index+1)
                
                if index < n_plots - 1:
                    ax.set_xticks([])
                
                ax.set_ylabel('{}'.format(v), fontsize=8)
                to_plot = moving_avg(yvar, window)
                ax.plot(tmstep,to_plot, c='r')
            
            if wt_var:
                ax = rew_plot.add_subplot(n_plots,1,n_plots)
                ax.set_frame_on(0)
                ax.set_axis_off()
                for j in range(n_write):
                    ax.text(.05, .1 * (n_write-j), lines[j], fontsize=6)
            
            if not osp.exists(osp.join(dir, out)):
                os.makedirs(osp.join(dir, out))
            
            plt.savefig(osp.join(dir,'{}/{}.png'.format(out,i)))
            rew_plot.clear()
            
def visualize_control(dir=os.getcwd(), subdir=None, var='intrinsic_reward',
                      control='bmax', window=5, out=None):
    if subdir is not None:
        dir = osp.join(dir, subdir)
    
    progress = [f for p in os.walk(dir) \
                for f in glob(osp.join(p[0], 'progress.csv'))]
    vars = [v for p in os.walk(dir) \
                for v in glob(osp.join(p[0], 'vars.json'))]
    
    progress, vars = remove_duplicate(progress, vars)
    
    if out is not None:
        dir = osp.join(dir, out)
    
    if not osp.exists(osp.join(dir, 'figs')):
        os.makedirs(osp.join(dir, 'figs'))
    colours = ['b', 'g', 'r', 'c', 'm']
    control_values = {}

    if var == '__all__':
        csv = progress[0]
        try:
            data = pd.read_csv(csv)
            valid_vars = list(data.columns.values)
        except:
            raise ValueError("First CSV must have data")
        for i in range(len(progress)):
            csv = progress[i]
            vardict = vars[i]
            try:
                data = pd.read_csv(csv)
            except:
                continue
            for var in valid_vars:
                if var not in control_values:
                    control_values[var] = {}
                try:
                    yvar = data[var].tolist()
                except:
                    continue
                
                value = vardict[control]
                if value not in control_values[var]:
                    control_values[var][value] = ([0]*int(1e3), [0]*int(1e3))
                control_values[var][value] = aggregate(control_values[var][value], yvar)
                
        for (var, plot_dict) in control_values.items():
            rew_plot = plt.gcf()
                
            ax1 = rew_plot.add_axes((0.1,0.4,0.8,0.5))
            ax1.set_title("{}, controlling for {}".format(var, control))
            ax1.set_xlabel('timestep')
            ax1.set_ylabel(var)
            tstep = list(range(1,int(1e7), int(1e4)))
            handles = []
            for index, (value, item) in enumerate(plot_dict.items()):
                means = moving_avg(item[0], window=window)
                line, = ax1.plot(tstep[1:], means[1:], c=colours[index], label="{}: {}".format(control, str(value)))
                handles.append(line)
            ax1.legend(handles)
            if not osp.exists(osp.join(dir, '{}'.format(var))):
                os.makedirs(osp.join(dir, '{}'.format(var)))
            rew_plot.savefig(osp.join(dir,'{}/{}.png'.format(var,control)))
            plt.gcf().clear()
    
    else:
        for i in range(len(progress)):
            csv = progress[i]
            vardict = vars[i]
            try:
                data = pd.read_csv(csv)
            except:
                continue
            yvar = data[var].tolist()
            value = vardict[control]
            if value not in control_values:
                control_values[value] = ([0]*int(1e3), [0]*int(1e3))
            control_values[value] = aggregate(control_values[value], yvar)
            
        rew_plot = plt.gcf()
            
        ax1 = rew_plot.add_axes((0.1,0.4,0.8,0.5))
        ax1.set_title("{}, controlling for {}".format(var, control))
        ax1.set_xlabel('timestep')
        ax1.set_ylabel(var)
        tstep = list(range(1,int(1e7), int(1e4)))
        handles = []
        for index, (_, item) in enumerate(control_values.items()):
            print(item[0])
            means = moving_avg(item[0], window=window)
            line, = ax1.plot(tstep[1:], means[1:], c=colours[index], label="{}: {}".format(control, str(value)))
            handles.append(line)
        ax1.legend(handles)
        if not osp.exists(osp.join(dir, '{}'.format(var))):
            os.makedirs(osp.join(dir, '{}'.format(var)))
        rew_plot.savefig(osp.join(dir,'{}/{}.png'.format(var,control)))
        plt.gcf().clear()

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
    parser.add_argument('-d', '--dir', nargs='+', type=str, default=None, help='fetch directory')
    parser.add_argument('--out', help='output directory', type=str, default=None)
    #parser.add_argument('--var', help='variable to visualize', type=str, default='eprewmean')
    parser.add_argument('-v', '--var', nargs='+', type=str, help='vars to visualize', default='eprewmean')
    parser.add_argument('--control', help='variable to control', type=str, default='__none__')
    parser.add_argument('--window', help='averaging factor', type=int, default=10)
    parser.add_argument('--wt', help='write_variables', type=int, default=0)
    parser.add_argument('--onegraph', help='plot on one graph', type=int, default=0)
    return parser    

def main():
    args = program_arg_parser().parse_args()
    if args.control == '__none__':
        visualize_all(subdir=args.dir, var=args.var, out=args.out, 
                      wt_var=args.wt, window=args.window)
        
    elif args.onegraph:
        visualize_multi(subdir=args.dir, var=args.var, out=args.out, wt_var=args.wt)
    
    else:
        visualize_control(subdir=args.dir, var=args.var, control=args.control, 
                          window=args.window, out=args.out)

if __name__ == '__main__':
    main()
        
            
      