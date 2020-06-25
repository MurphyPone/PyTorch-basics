import torch
import torch.nn as nn

class DenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 10),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    # FF occurs by passing the encoded x-bar to the decoder so that loss can be calculated by the reconstruction h
    def forward(self, x):
        x = x + torch.randn_like(x) # add noise 
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)