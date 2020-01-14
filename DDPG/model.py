import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical 

class PolicyGradient(nn.Module):
    def __init__(self, env):
        super(PolicyGradient, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.shape[0])
        ) 

    
    def forward(self, s):
        return self.main(torch.FloatTensor(s))

class Q(nn.Module):
    def __init__(self, env):
        super(Q, self).__init__() 
        
        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        return self.main(torch.cat([s, a], 1))


    