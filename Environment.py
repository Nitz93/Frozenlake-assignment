#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:55:23 2020

@author: nithinmahadev
"""

import numpy as np
import contextlib
from itertools import product

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

        
class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        
        if state == self.absorbing_state:
            next_state = self.absorbing_state
            reward = self.r(next_state, state, action)
            return next_state, reward
        
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        #self.random_state = np.random.RandomState(seed)
        
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, gmap, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.gmap = np.array(gmap, dtype=np.float)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        
        # Indices to states (coordinates), states (coordinates) to indices 
        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)
        # TODO:
            
        # Up, left, down, right.
        self.actions = [(-1, 0),(0, -1), (1, 0), (0, 1)]
        self.goal = np.argmax(self.gmap)
        hole_idx = np.where(self.gmap.flatten() == -1)
        self.holes = hole_idx[0].tolist()
        # Precomputed transition probabilities
        self._p = np.zeros((n_states, n_states, n_actions))
        
        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])
                
                # If next_state is not valid, default to current state index
                next_state_index = self.stoi.get(next_state, state_index)
                if (state_index in self.holes) or ( state_index == self.goal or (state_index == self.absorbing_state)):
                    next_state_index = self.absorbing_state  
                
                self._p[next_state_index, state_index, action_index] = 1.0
        
    def step(self, action):
        
        if np.random.uniform(0, 1) < self.slip:
            action = np.random.choice(range(4))
        
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
    
        
    def p(self, next_state, state, action):
        # TODO:
        return self._p[next_state, state, action]
    
    def r(self, next_state, state, action):
        # TODO:
        # if state >= self.lake.size:
        #     st = self.lake.size - 1
        # else:
        #     st = state
        if state == self.goal:
            return 1
        else:
            return 0
   
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))
