import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Normal 
from math import pi, isinf 
from copy import deepcopy 

class Distribution(nn.Module):
    def __init__(self, alpha=None, mu=None, sigma=None, K=None, requires_grad=False):
        super().__init__()
        self.requires_grad = requires_grad
        assert(any([True for x in [alpha, mu, sigma, K] if x is not None]))

        self.K = K # max # of modes in the distribution

        given = [x for x in [alpha, mu, sigma] if x is not None]
        if self.K is None: self.K = len(given[0])
        assert(all(len(x) == self.K for x in given)) # ensure all args are same size

        def init_weights(x, log=False):
            if x is not None:           # convert to a tensor
                w = torch.FloatTensor(x)
                if log: w = torch.log(w) # used for σ to ensure > 0
            else: 
                w = torch.rand(self.K)
            assert(w.shape == (self.K,))
            w.requires_grad = requires_grad
            return w

        self.α = init_weights(alpha)
        self.μ = init_weights(mu)
        self.σ = init_weights(sigma, log=True)

    def parameters(self):
        return [self.α, self.μ, self.σ]

    def forward(self, x):
        x = torch.FloatTensor(x)
        a = F.softmax(self.α, dim=0)            # remap all values s.t. sum == 1, necessary for tracatbility
        return torch.sum(a * self.N(x), dim=1)  # calc norm dist for each val in x * softmaxed alpha

    def mean(self):
        a = F.softmax(self.α, dim=0)
        return torch.sum(a * self.μ, dim=0)

    def sample(self, batch_size=1):
        with torch.no_grad():
            return self.rsample(batch_size)

    def rsample(self, batch_size=1):
        a = F.softmax(self.α, dim=0)
        i = torch.multinomial(a, 1) # Sample an index i based on the weights of alpha with favorability towards higher weights
        dist = Normal(self.μ[i], self.σ[i].exp())
        return dist.rsample((batch_size,))

    # get logarithmic probability of a point across the distribution
    def log_prob(self, sample):
        return self.prob(sample.log())

    def prob(self, sample):
        sample = torch.FloatTensor(sample)
        dists = [Normal(μ, σ.exp()) for (μ, σ) in zeip(self.μ, self.σ)]
        prob = torch.stack([dist.log_prob(sample).exp() for dist in dists]).squeeze()
        if len(p.shape) == 1: 
            prob = prob.unsqueeze(1)
        prob = torch.clamp(prob, 1e-10, 1) # prevent div by 0
        a = F.softmax(self.α, dim=0).unsqueeze(1)
        return torch.sum(prob * a, dim=0)

    def N(self, x):
        if len(x.shape) == 1:
            x = torch.FloatTensor(x).unsqueeze(1)
        return torch.exp(-torch.pow(x - self.μ, 2) / (2 * torch.pow(self.σ.exp(), 2))) / torch.sqrt(2 * pi * torch.pow(self.σ.exp(), 2))

