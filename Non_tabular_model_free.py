#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:00:47 2020

@author: nithinmahadev
"""
import numpy as np
import contextlib
from itertools import product
from Environment import *
import random as rand

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        # TODO:
        done = False
        if rand.uniform(0, 1) > (1 - epsilon[i]):
            action = np.argmax(q)
        else:
            action = np.random.choice(range(env.n_actions))

        while not done:
            s1, reward, done = env.step(action)
            delta = reward - q[action]
            q = s1.dot(theta)
            if rand.uniform(0, 1) > (1 - epsilon[i]):
                action1 = np.argmax(q)
            else:
                action1 = np.random.choice(range(env.n_actions))
            delta += gamma * q[action1]
            theta += (eta[i] * delta * features[action])

            action = action1
            features = s1
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

        q = features.dot(theta)
        done = False
        while not done:
            action=0
            if rand.uniform(0, 1) > (1 - epsilon[i]):
                action = np.argmax(q)
            else:
                action = np.random.choice(range(env.n_actions))
            s1, reward, done = env.step(action)
            delta = reward - q[action]
            q = s1.dot(theta)
            delta = delta + gamma * np.max(q)
            theta = theta + eta[i] * delta * features[action]
            features = s1
                                
    return theta    
