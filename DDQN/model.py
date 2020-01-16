import torch 
import torch.nn as nn

class Q(nn.Module):
    def __init__(self, env):
        super(Q, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))

        