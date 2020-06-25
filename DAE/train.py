import torch 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import DenoisingAutoEncoder
from visualize import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
algo_name = 'DAE'
max_episodes = 100
batch_size = 128
lr = 1e-3           # play with this value

# load dataset from MNIST, save to ../datasets
dataset = MNIST('../datasets', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

DAE = DenoisingAutoEncoder()
optimizer = optim.Adam(DAE.parameters(), lr=lr)

for episode in range(max_episodes):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)            # squash to 1D
        output = DAE(img)

        # calculate loss and update acc.
        loss = F.binary_cross_entropy(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 10 == 0:
        img = img.view(output.size(0), 1, 28, 28)
        save_image(img, './img/'+str(episode)+'_eps.png')
    
    update_viz(episode, loss.item(), algo_name)