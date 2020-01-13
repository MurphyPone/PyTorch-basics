import torch 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn 
import torch.optim as optim 
import math 
import numpy as np 
from visualize import * 

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# print(torch.__version__)

N = 1000
x = torch.linspace(0, 2*math.pi, N)
y = torch.sin(x)

for xi in range(math.floor(math.pi *2 * 100)):
    plot_sin(xi, math.sin(xi/100), 'sin_curve', '#FF0000')

dataset = TensorDataset(x, y)
dl = DataLoader(dataset, batch_size=32, shuffle=True)

# for xi, yi in dl: 
#     print(xi ,yi)

class Linear(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(1), 
            nn.Linear(in_features=1, out_features=hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.main(x)

model = Linear()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

def vis(i):
    with torch.no_grad():
        yh = model(x.unsqueeze(1)).squeeze(1).cpu()
        for xi, yi in zip(np.array(x.cpu()), np.array(yh)):
            # print(xi.item(), yi.item())
            plot_sin(xi.item(), yi.item(), 'guess_curve', '#0000FF')

epochs = 20

for ep in range(epochs):
    sum_loss = torch.zeros(1)
    vis(ep)
    
    for xi, yi in dl: 
        yh = model(xi.unsqueeze(1))
        # print("episode " + str(ep) + "/" + str(epochs))
        # print("xi: ", xi)
        # print("yh: ", yh)
        # print("yi: ", yi.unsqueeze(1))
        # print("")

        loss = ((yh-yi.unsqueeze(1))**2).mean()
        sum_loss += loss 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(sum_loss)
