import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import torch.nn.functional as F 

#import tensorflow as tf

import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#---------------------------------------------------------
# Globals
#---------------------------------------------------------

download = True
batch_size = 64
epochs = 5
acc_iter = []
loss_iter = []
learning_rate = 1e-3
decay = 1e-5
feed = []

#---------------------------------------------------------
# Functions
#---------------------------------------------------------

def training_loop(model, optimizer, epoch, lr):
    model = model.to(device=device)
    for e in range(epoch):
        for (input, _) in train_loader:
            input = input.reshape(-1, 28*28)
            input = input.to(device=device)
            output = model(input)
            loss = nn.MSELoss(output, input)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {e+1} ---> Loss = {loss.item():.4f}')
        feed.append((epoch, input, output))
        


#---------------------------------------------------------
# Classes
#---------------------------------------------------------

def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1)

def init(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_normal_(net.weight)
        net.bias.data.fill_(0.001)
    if isinstance(net, nn.ConvTranspose2d):
        nn.init.xavier_normal_(net.weight)
        net.bias.data.fill_(0.001)
    

class Autoencoder_Conv(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) 
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#---------------------------------------------------------
# Main
#---------------------------------------------------------

# Download Dataset MNIST with DataLoader
#---------------------------------------
transform=T.ToTensor()
mnist_data_train = dset.MNIST(root=r'./data', train=True, download=download, transform=transform)
mnist_data_test = dset.MNIST(root=r'./data', train=False, download=download, transform=transform)

train_subset, val_subset = torch.utils.data.random_split(mnist_data_train, [51200, 8800])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_data_test, batch_size=batch_size, shuffle=True)

# Linear Modul
#---------------------------------------
model = Autoencoder_Conv()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

# Training
#---------------------------------------
num_epochs = 12
feed = []
for epoch in range(num_epochs):
    for (input, _) in train_loader:
        #input = input.reshape(-1, 28*28) #nur für linear
        output = model(input)
        loss = loss_function(output, input)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoche {epoch+1} ---> Loss = {loss.item():.4f}')
    feed.append((epoch, input, output))


# Plotting
#---------------------------------------
for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    input = feed[k][1].detach().numpy()
    output = feed[k][2].detach().numpy()
    for i, item in enumerate(input):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        #item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(output):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        #item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

plt.show()
