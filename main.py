#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:03:57 2020

@author: nithinmahadev
"""
import numpy as np
import os
import contextlib
from itertools import product
from Environment import *
from Model_based_algorithms import *
from Tabular_model_free import *
from Non_tabular_model_free import *


################ Main function ################

def main():
    seed = 0
    small_lake = True
    big_lake = False
    
    
    if small_lake == True:
        # Small lake
        small_lake =   [['&', '.', '.', '.'],
                        ['.', '#', '.', '#'],
                        ['.', '.', '.', '#'],
                        ['#', '.', '.', '$']]
        
        lake = small_lake
        gmap = np.zeros((4,4))
        gmap[1,1] = -1
        gmap[1,3] = -1
        gmap[2,3] = -1
        gmap[3,0] = -1
        gmap[3,3] = 1
        
    if big_lake == True:
        big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '.', '.', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '.', '.', '.', '.', '#', '.', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '.'],
                    ['.', '#', '#', '.', '.', '.', '#', '.'],
                    ['.', '#', '.', '.', '#', '.', '#', '.'],
                    ['.', '.', '.', '#', '.', '.', '.', '$']]
    
        lake = big_lake
        gmap = np.zeros((8,8))
        gmap[2,1] = -1
        gmap[3,5] = -1
        gmap[2,4] = -1
        gmap[5,1] = -1
        gmap[5,2] = -1
        gmap[6,1] = -1
        gmap[6,4] = -1
        gmap[6,6] = -1
        gmap[7,3] = -1
        gmap[7,7] = 1        
        

    env = FrozenLake(gmap, lake, slip=0.1, max_steps=16, seed=seed)
    
    # print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    
    print('')
    
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    
    print('')
    
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    print('')
    op = policy_evaluation(env, policy, gamma, theta, max_iterations)
    print(op)
    
    # print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    op = policy_evaluation(env, policy, gamma, theta, max_iterations)
    print(op)
    
    print('')
    
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    print('')
    op = policy_evaluation(env, policy, gamma, theta, max_iterations)
    print(op)
    
    print('')
    
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')
    
    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    op = policy_evaluation(env, policy, gamma, theta, max_iterations)
    print(op)
    
    print('')
    
    print('## Linear Q-learning')
    
    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                    gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    op = policy_evaluation(env, policy, gamma, theta, max_iterations)
    print(op)

if __name__ == "__main__":
    main()