################ Environment ################

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
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
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
    def __init__(self, lake, slip, max_steps, seed=None):
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
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        # Indices to states (coordinates), states (coordinates) to indices 
        self.itos = list(product(range(lake.shape[0]), range(lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # TODO:
            
        # Up, down, left, right, stay.
        self.actions = [(-1, 0),(0, -1), (1, 0), (0, 1)]
        
        # Precomputed transition probabilities
        self._p = np.zeros((self.n_states, self.n_states, self.n_actions))
        
        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])
                
                # If next_state is not valid, default to current state index
                next_state_index = self.stoi.get(next_state, state_index)
                
                self._p[next_state_index, state_index, action_index] = self.slip
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
    
        
    def p(self, next_state, state, action):
        # TODO:
        return self._p[next_state, state, action]
    
    def r(self, next_state, state, action):
        # TODO:
        if self.lake[self.itos[state]] == '$':
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
    
    # TODO:
    for _ in range(max_iterations):
        delta = 0
        p = policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        for s in range(env.n_states):
            delta = max(delta, np.abs(p[s] - policy[s]))
        
        if delta < theta:
            break
    
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    # TODO:    
    for _ in range(max_iterations):
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

    return policy, value

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    def select_action(state,e):       
        action=0
        if (1 - e) > e: 
            action = np.argmax(q[state, :]) 
        else: 
            action = np.random.choice(range(4))
        return action 
    
    
    for i in range(max_episodes):
        s = env.reset()
        e = epsilon[i]
        # TODO:
        action = select_action(s,e)      
        done = False
        while not done:
            s2, reward, done, info = env.step(action) 
            action2 = select_action(s2,e) 
          
            #Learning the Q-value 
            q[s, action] = q[s, action] + eta[i] * ((reward + gamma * q[s2, action2])  -  q[s, action]) 
      
            s = s2 
            action1 = action2 
        
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
        if (1 - e) > e: 
            action = np.argmax(q[state, :]) 
        else: 
            action = np.random.choice(range(4))
        return action
    
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        e = epsilon[i]
        action = select_action(s,e)
        done = False
        while not done:
            s2, reward, done, info = env.step(action) 
            action2 = select_action(s2, e) 
          
            #Learning the Q-value 
            q[s, action] = q[s, action] + eta[i] * ((reward + gamma * np.max(q[s2,action2]))  -  q[s, action]) 
      
            s = s2 
            action1 = action2 
        
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
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

    return theta    

################ Main function ################

 def main():
    seed = 0
    
    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    
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
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    
    print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
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
