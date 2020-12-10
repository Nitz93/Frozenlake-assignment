#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:56:44 2020

@author: nithinmahadev
"""

import numpy as np
import contextlib
from itertools import product
from Environment import *


################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:
    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum([env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in range(env.n_states)])

            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            break

    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:
    for s in range(env.n_states):
        policy[s] = np.argmax([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for a in range(env.n_actions)])

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    value = np.zeros(env.n_states)
    
    # TODO:
    for it in range(max_iterations):
        p = policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        
        if np.array_equal(p, policy):
            break
    
    print(f'iterations: {it}')
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    # TODO:    
    for it in range(max_iterations):
        delta = 0.
        
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for a in range(env.n_actions)])
    
            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for a in range(env.n_actions)])
    
    print(f'iterations: {it}')
    return policy, value
