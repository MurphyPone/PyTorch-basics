"""
Explain what the replay buffer is used for
"""

import numpy as np 
import torch 
import random 
from collections import deque

class ReplayBuffer():
    """Stores tuples of (state, action, reward, state+1, mask) which the agent samples from when TODO exploring vs. randomly acting?"""
    
    def __init__(self, size):
        """Constructor for a ReplayBuffer which takes the size of the buffer"""
        
        self.buffer = deque(maxlen=int(size))       # extends list which can be sampled from 
        self.maxSize = size 
        self.len = 0

    def sample(self, count=128): 
        """Attempts to fetch the desired amount of transitions tuples and return them separated into 
        arrays of states, actions, rewards, etc., defaults to 128
        """
        
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)   
        
        s_arr   = torch.FloatTensor(np.array( [arr[0] for arr in batch] ))  # state
        a_arr   = torch.FloatTensor(np.array( [arr[1] for arr in batch] ))  # action
        r_arr   = torch.FloatTensor(np.array( [arr[2] for arr in batch] ))  # reward
        s2_arr  = torch.FloatTensor(np.array( [arr[3] for arr in batch] ))  # state + 1
        m_arr   = torch.FloatTensor(np.array( [arr[4] for arr in batch] ))  # mask --> whether the agent is still alive or not 

        return s_arr, a_arr.unsqueeze(1), r_arr.unsqueeze(1), s2_arr, m_arr.unsqueeze(1)

    def len(self):
        """Returns the length of the ReplayBuffer"""

        return self.len 

    def store(self, s, a, r, s2, d):
        """Creates a transition tuple from a state, action, reward, state+1, and mask, 
        ensures they are formatted correctly and appends them to the buffer
        """

        def fix(x):
            if not isinstance(x, np.ndarray): return np.array(x)    # checks to make sure storing an n-dimensional array
            else: return x

        # Whereas states are provided by the environment and *likely* already float64, 
        # the action is executed by the agent so we need to ensure it's sufficiently continuous    
        data = [s, np.array(a, dtype=np.float64), r, s2, 1 - d]     
        transition = tuple(fix(x) for x in data)
        self.len = min(self.len + 1, self.maxSize)
        self.buffer.append(transition)  # replay buffer stores "transitions" that act as a memory of how it got to a given state 