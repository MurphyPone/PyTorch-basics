import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Normal 

class PolicyNetwork(nn.Module):
    def __init__(self, env):
        super(PolicyNetwork, self).__init__()

        self.actor = nn.Sequential(     ## any reason to use sequential instead of 
            nn.Linear(env.observation_space.shape[0], 64),  # input size depends on what the input is 
            nn.ReLU(),                  # use ReLU or some LU activation function since other normalizing 
                                        #functions like tanh, sigmoid, etc. 
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # TODO re-read SAC to figure out what these do 
        self.μ = nn.Sequential(
            nn.Linear(64, env.action_space.shape[0]) # action_space.size ≠ observation_space.size always
        ) 

        self.σ = nn.Sequential(
            nn.Linear(64, env.action_space.shape[0])
        ) 

    def dist(self, s): 
        main = self.actor(torch.FloatTensor(s))
        μ = self.μ(main)
        σ = torch.exp(self.σ(main))
        dist = Normal(μ, σ) # Normal(location, scale)
        return dist 

    def forward(self, x):
        dist = self.dist(x)
        return torch.tanh(dist.sample())

    def sample(self, x):
        dist = self.dist(x)
        x = dist.rsample()      # rsample here because the sample has been reparameterized s.t. it is now differentiable
        a = torch.tanh(x)       # using tanh map outputs since it's range is (-1, 1) which is arbitrarily desired

        log_p = dist.log_prob(x)
        log_p -= torch.log(1 - torch.pow(a, 2) + 1e-6) ### ????? 

        return a, log_p

class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], 64), # TODO why is the input big enough for both?
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):            # feeds in the state and action?
        return self.main(torch.cat([s, a], 1))