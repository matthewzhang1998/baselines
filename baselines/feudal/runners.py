#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:20:22 2018

@author: matthewszhang
"""
import time
import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.feudal.utils import sf01
from baselines.program.decode import decode

class FeudalRunner(AbstractEnvRunner):
    # to do -> work on making a recurrent version
    def __init__(self, env, model, nsteps, recurrent, fixed_manager=False,
                 fixed_agent=False):
        self.env = env
        self.fixed_manager = fixed_manager
        self.fixed_agent = fixed_agent
        self.recurrent=recurrent
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = np.tile(model.initial_state, (nenv, *np.ones_like(model.initial_state.shape)))
        self.dones = [False for _ in range(nenv)]
        if self.fixed_manager:
            self.init_goal, self.supervised_action = env.goal(self.obs)
        else:
            self.init_goal = [model.init_goal] * nenv
            self.supervised_action = [0] * nenv
        
        # extra step in the beginning
#        actions, goal, pi, self.states = self.model.step(self.obs, self.states, self.init_goal)
#        if self.fixed_agent:
#            actions = env.action(self.obs)
#        if self.recurrent:
#            self.states = self.states
#        self.obs[:], rewards, self.dones, _ = self.env.step(actions) # perform 1 step, safety
#        print("pre_2", decode(self.obs[0]))
#        if self.fixed_manager:
#            self.init_goal = env.goal(self.obs)
        
    def run(self):
        mb_obs, mb_rewards, mb_goals, mb_actions, mb_dones, mb_pi, \
                    mb_init_goals, mb_supervised_actions = [],[],[],[],[],[],[],[]
        epinfos = []
        t_model = 0
        t_run = 0
        mb_states = self.states
        for i in range(self.nsteps):
            t_start = time.time()
            actions, goal, pi, self.states = self.model.step(self.obs, self.states, self.init_goal)
            t_step = time.time()
            t_model += (t_step - t_start)
            if self.fixed_agent:
                actions = self.env.action(self.obs)
            if self.recurrent:
                self.states = self.states[0]
            mb_obs.append(self.obs.copy())
            if self.recurrent:
                mb_goals.append(goal)
            else:
                mb_goals.append(goal)
            
            mb_pi.append(pi)
            mb_actions.append(actions)
            mb_dones.append(self.dones)
            mb_init_goals.append(self.init_goal)
            mb_supervised_actions.append(self.supervised_action)
            t_env = time.time()
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            if self.dones[0]:
                temp_obs = np.zeros((self.env.num_envs,) \
                                    + self.env.observation_space.shape, dtype=np.float32)
                temp_obs[:] = self.env.final_obs()
                mb_obs.append(temp_obs.copy())
            t_run += (t_env - t_step)
            if self.fixed_manager:
                self.init_goal, self.supervised_action = self.env.goal(self.obs)
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
        mb_supervised_actions = np.asarray(mb_supervised_actions, dtype=np.int32)
        # lose environment parallelism here -> need to fix
        
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi, mb_init_goals, mb_goals, mb_supervised_actions \
            = (np.swapaxes(arr,0,1) for arr in
            (mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi, mb_init_goals, mb_goals, mb_supervised_actions))
        print(t_model, t_run)
        
        return mb_obs, mb_rewards, mb_actions, mb_dones, mb_pi, mb_init_goals,\
                mb_goals, mb_supervised_actions, mb_states, epinfos
        
class TestRunner(AbstractEnvRunner):
        # to do -> work on making a recurrent version
    def __init__(self, env, model, recurrent, fixed_manager=False,
                 fixed_agent=False):
        self.env = env
        self.model = model
        self.recurrent = recurrent
        self.fixed_manager = fixed_manager
        self.fixed_agent = fixed_agent
        
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        if self.fixed_manager:
            self.init_goal = env.goal(self.obs)
        else:
            self.init_goal = [model.init_goal] * nenv
        
        # not sure why but one step is required at the beginning
        if self.recurrent:
            actions, goal, pi, self.states = self.model.step(self.obs, self.states, self.init_goal)
            if self.fixed_agent:
                actions = env.action(self.obs)
            self.states = self.states[0]
            self.obs[:], rewards, self.dones, _ = self.env.step(actions) # perform 1 step, safety
            if self.fixed_manager:
                self.init_goal = env.goal(self.obs)
        
        
    def run(self):
        mb_obs, mb_rewards, mb_goals, mb_actions, mb_dones, mb_pi, mb_states, \
                    mb_init_goals = [],[],[],[],[],[],[],[]
        epinfos = []
        while(1): # loop continuously
            mb_states.append(self.states)
            actions, goal, pi, self.states = self.model.step(self.obs, self.states, self.init_goal)
            if self.fixed_agent:
                actions = self.env.action(self.obs)
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
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_init_goals.append(self.init_goal)
            if self.fixed_manager:
                self.init_goal = self.env.goal(self.obs)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            if self.dones[0] == True:
                break
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
