import torch 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn 
import torch.optim as optim 
import math 
import matplotlib.pyplot as plt 
import numpy as np 

torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# print(torch.__version__)

N = 1000
x = torch.linspace(0, 2*math.pi, N)
y = torch.sin(x)

plt.plot(np.array(x.cpu()), np.array(y.cpu()))
plt.savefig("./images/sinwave.png")

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
        plt.clf()
        plt.plot(np.array(x.cpu()), np.array(yh), np.array(x.cpu()), np.array(y.cpu()))
        plt.show()
        plt.savefig("./images/img_" + str(i) +".png")
        

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
