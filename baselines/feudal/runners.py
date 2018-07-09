#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:20:22 2018

@author: matthewszhang
"""
import numpy as np
from baselines.common.runners import AbstractEnvRunner

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    if len(arr.shape) == 1:
        return arr
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

class FeudalRunner(AbstractEnvRunner):
    # to do -> work on making a recurrent version
    def __init__(self, env, model, nsteps, recurrent, fixed_manager=False):
        self.env = env
        self.fixed_manager = fixed_manager
        self.recurrent=recurrent
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        if self.fixed_manager:
            self.init_goal = env.goal(self.obs)
        else:
            self.init_goal = [model.init_goal] * self.nenvs
        
        # not sure why but one step is required at the beginning
        if recurrent:
            actions, goal, pi, self.states = self.model.step(self.obs, self.states, self.init_goal)
            self.states = self.states[0]
            self.obs[:], rewards, self.dones, _ = self.env.step(actions) # perform 1 step, safety
            if self.fixed_manager:
                self.init_goal = env.goal(self.obs)
            
    def run(self):
        mb_obs, mb_rewards, mb_goals, mb_actions, mb_dones, mb_pi, mb_states, \
                    mb_init_goals = [],[],[],[],[],[],[],[]
        epinfos = []
        for i in range(self.nsteps):
            mb_states.append(self.states)
            actions, goal, pi, self.states = self.model.step(self.obs, self.states, self.init_goal)
            if self.recurrent:
                self.states = self.states[0]
            mb_obs.append(self.obs.copy())
            if self.recurrent:
                mb_goals.append(goal[0][0])
            else:
                mb_goals.append(goal[0])
            mb_pi.append(pi)
            mb_actions.append(actions[0])
            mb_dones.append(self.dones)
            if self.dones[0] == True: # lose parallelism again
                self.states = self.model.initial_state
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            if self.fixed_manager:
                self.init_goal = self.env.goal(self.obs)
            mb_init_goals.append(self.init_goal)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_init_goals = np.asarray(mb_init_goals, dtype=np.float32)
        mb_pi = np.asarray(mb_pi, dtype=np.float32)
        mb_goals = np.asarray(mb_goals, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_states = np.asarray(mb_states, dtype=np.float32)
        # lose environment parallelism here -> need to fix
        
        return (*map(sf01, (mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi, mb_init_goals)),
                mb_goals, mb_states, epinfos)

class I2ARunner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        
    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_dones = [],[],[],[]
        epinfos = []
        for i in range(self.nsteps):
            actions = self.model.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions[0])
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        # lose environment parallelism here -> need to fix
        
        return (*map(sf01, (mb_obs, mb_rewards, mb_actions, mb_dones)), epinfos)
