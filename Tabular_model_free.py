#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:57:58 2020

@author: nithinmahadev
"""

import numpy as np
import contextlib
from itertools import product
from Environment import *
import random as rand

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    def select_action(state,e):       
        action=0
        if rand.uniform(0, 1) > (1 - e):
            action = np.argmax(q[state, :])                      
        else: 
            action =  random_state.choice(range(env.n_actions))
        return action 
    
    
    for i in range(max_episodes):
        s = env.reset()
        e = epsilon[i]
        # TODO:
        action = select_action(s,e)      
        done = False
        while not done:
            s2, reward, done  = env.step(action) 
            action2 = select_action(s2,e) 
          
            #Learning the Q-value 
            q[s, action] = q[s, action] + eta[i] * (reward + (gamma * q[s2, action2])  -  q[s, action]) 
      
            s = s2 
            action = action2 
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)  
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    def select_action(state,e):       
        action=0
        if rand.uniform(0, 1) > (1 - e):
            action = np.argmax(q[state, :])                      
        else: 
            action =  random_state.choice(range(env.n_actions))
        return action 
    
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        e = epsilon[i]
        action = select_action(s,e)
        done = False
        while not done:
            s2, reward, done = env.step(action) 
            action2 = select_action(s2, e) 
          
            #Learning the Q-value 
            q[s, action] = q[s, action] + eta[i] * (reward + (gamma * np.max(q[s2,:]))  -  q[s, action]) 
            s = s2 
            action = action2 
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
