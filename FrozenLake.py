#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 00:49:23 2020

@author: nithinmahadev
"""

import numpy as np
import contextlib
from itertools import product
import random as rand

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



################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:
    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)for a in range(env.n_actions)])

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

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    def select_action(state,e):       
        action=0
        if  rand.uniform(0, 1) > (1 - e): 
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
            q[s, action] = q[s, action] + eta[i] * ((reward + gamma * np.max(q[s2,action2]))  -  q[s, action]) 
      
            s = s2 
            action = action2 
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value


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
    
    print('# Model-based algorithms')
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
    max_episodes = 3000
    eta = 0.5
    epsilon = 0.5
    
    print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    op = policy_evaluation(env, policy, gamma, theta, max_iterations)
    #print(op)
    #print(value)
    print('')
    
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')
    
    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    
    print('## Linear Q-learning')
    
    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                    gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

if __name__ == "__main__":
    main()
