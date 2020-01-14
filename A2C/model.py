import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical # Why categorical instead of Normal? 

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, s):
        """Accepts a state and returns an action based on the categorical distribution of distinct action"""

        logits = self.main(torch.FloatTensor(s))                    # ln(probability of a distinct action)
        dist = Categorical(logits=logits)                           # stores discrete characteristics
        a = dist.sample().numpy()                                   # TODO what does this do?  convert from torch to np?
        return a

    def get_log_prob(self, s, a ):
        """You're gonna have to ask @BCHoagland about this one"""
        orig_s_shape = s.shape
        s = s.squeeze()
        a = a.squeeze()
        if len(s.shape) == 0:
            s = s.unsqueeze(0)
        if len(a.shape) == 0:
            a = a.unsqueeze(0)

        logits = self.main(torch.torch.FloatTensor(s))
        dist = Categorical(logits=logits)
        log_p = dist.log_prob(torch.FloatTensor(np.array(a)))

        while len(orig_s_shape) > len(log_p.shape):
            log_p = log_p.unsqueeze(len(log_p.shape))

        return log_p

    
class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)            # Critic's output is only one because... gauge of the actor's policy?
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))
