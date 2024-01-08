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
feed = []

#---------------------------------------------------------
# Functions
#---------------------------------------------------------


def training_loop(model, optimizer, lossfunction, trainloader, epochs, save_checkpoint=True, checkpoint_path = r'./checkpoint'):
    model = model.to(device=device)
    for e in range(epochs):
        for i, (input, labels) in enumerate(trainloader):
            input = input.to(device=device)
            output = model(input)
            loss = lossfunction(output, input)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {e+1} ---> Loss = {loss.item():.4f}')
        feed.append((epochs, input, output))
    if save_checkpoint == True:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)
    return print(f'Checkpoint set at Epoch {epochs} with Loss {loss}')
    
 

def load_checkpoint(model, optimizer, checkpoint_path = r'./checkpoint'):      # model und optimizer müssen neu initialisiert werden
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

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
    if isinstance(net, nn.Linear):
        nn.init.xavier_normal_(net.weight)
        net.bias.data.fill_(0.001)
    

class Autoencoder_Conv(nn.Module):
    # Img Size = ((Input Width or Hight - Filtersize + 2xPadding) / Stride) + 1 (bzw. Aufrunden)
    def __init__(self):
        super().__init__() 
        # Inputbild der Größe 1, 28, 28       
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 16, 14, 14 (28 - 3 + 2*1)/2)+1
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # 64, 1, 1
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
    
class FC_Classifier(nn.Module):
    def __init__(self):
        super().__init__()        
        # gefaltetes Bild mit der Größe 64, 1, 1
        self.classifier = nn.Sequential(
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 10),
            nn.LogSoftmax()
        )

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
model_ = Autoencoder_Conv()
lossfunction_ = nn.MSELoss()
optimizer_ = torch.optim.Adam(model_.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 12

# Training der Autoencoders
#---------------------------------------

training_loop(
    model=model_, 
    optimizer=optimizer_,
    lossfunction=lossfunction_,
    trainloader=train_loader,
    epochs=num_epochs,
    save_checkpoint=True
)


# Plotting
#---------------------------------------
for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.title(f'After Epoch {k}')
    plt.gray()
    feed_plot = feed
    input_plot = feed_plot[k][1].detach().to(device='cpu').numpy()
    output_plot = feed_plot[k][2].detach().to(device='cpu').numpy()
    for i, item in enumerate(input_plot):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        #item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(output_plot):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        #item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

plt.show()
