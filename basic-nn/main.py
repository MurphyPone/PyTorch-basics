# from https://github.com/DecipherNow/sense-rnd/blob/master/danielpcox/notebooks/ML/VeryBasicNNsInPyTorch.ipynb

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms 
import torch.optim as optim
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np 

image = Image.open("images/yoda.png")
#plt.imshow(image)

numbers = np.array(image)
#plt.imshow(numbers)

numbers[0, 0]
#print(numbers.shape)

transforms.Resize(512)(image)

transforms.ToTensor()(image).shape
imsize = 512 

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

tensornow = loader(image).unsqueeze(0)
# print(tensornow.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## fully connected layers 
        self.fc1 = nn.Linear(in_features=512*512*3, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = x.view(-1, 512*512*3)   # reshape the output of the convolutional layers to fit ->
        x = F.relu(self.fc1(x))     # fc1 -> relu 
        x = F.relu(self.fc2(x))     # fc2 -> relu
        x = self.fc3(x)             # fc3 -> 
        return x                    # output            


net = Net()

output = net(tensornow)
#print(output)

yoda = loader(Image.open("images/yoda.png")).unsqueeze(0)
benhall = loader(Image.open("images/benhall.jpg")).unsqueeze(0)
#print(yoda.shape)
#print(benhall.shape)

inputs = torch.cat((yoda, benhall), 0)
#print(inputs.shape)

onehot_labels = torch.FloatTensor([[1,0], [0,1]])
#print(onehot_labels)

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
optimizer.zero_grad()           # zero the parameter gradients

outputs = net(inputs)

loss = ((outputs-onehot_labels)**2).mean() # MSE Loss
print(loss)
loss.backward()
optimizer.step()
print(loss)
print(outputs)

net(yoda)
net(benhall)

