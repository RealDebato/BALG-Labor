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
acc_iter = []
loss_iter = []
learning_rate = 1e-3

#---------------------------------------------------------
# Functions
#---------------------------------------------------------

def check_accuracy(loader, model):
  
    num_correct = 0
    num_samples = 0
    model.eval() 
    #with torch.no_grad():
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32) 
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    acc_iter.append(acc)
    print(f'Got {num_correct} / {num_samples} correct ({100*acc:.2f}%)')

def training_loop(model, optimizer, epoch, lr):
    model = model.to(device=device)
    for e in range(epoch):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        for t, (x, y) in enumerate(train_loader):
            model.train()  
            x = x.to(device=device, dtype=torch.float32)  
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % 80 == 0:
                print(f'Epoch {e}, Iteration {t}, loss = {loss.item()}')
                loss_iter.append(loss.item())
                #print(f'accuracy on validation')
                #check_accuracy(val_loader, model)
                
        #print(f'accuracy on test')
        #check_accuracy(test_loader, model)
        lr = lr*0.95
        #print(f'learning rate {lr:.6f} at epoch {e}')


#---------------------------------------------------------
# Classes
#---------------------------------------------------------

def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1)

def init(net):
    if isinstance(net, nn.Linear):
        nn.init.normal_(net.weight, mean=0.0, std=1.0)
        net.bias.data.fill_(0)

class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.LogSoftmax(),
            nn.Linear(128, 64),
            nn.LogSoftmax(),
            nn.Linear(64, 12),
            nn.LogSoftmax(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.LogSoftmax(),
            nn.Linear(12, 64),
            nn.LogSoftmax(),
            nn.Linear(64, 128),
            nn.LogSoftmax(),
            nn.Linear(128, 28*28),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        

#---------------------------------------------------------
# Main
#---------------------------------------------------------

# Download Dataset MNIST with DataLoader
#---------------------------------------
transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
mnist_data_train = dset.MNIST(root=r'./data', train=True, download=download, transform=transform)
mnist_data_test = dset.MNIST(root=r'./data', train=False, download=download, transform=transform)

train_subset, val_subset = torch.utils.data.random_split(mnist_data_train, [51200, 8800])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_data_test, batch_size=batch_size, shuffle=True)

# Linear Modul
#---------------------------------------
model_linear = Autoencoder_Linear()
optimizer_linear = optim.SGD(model_linear.parameters(), lr=learning_rate)

# Training
#---------------------------------------

training_loop(model=model_linear, optimizer=optimizer_linear, epoch=5, lr=learning_rate)
