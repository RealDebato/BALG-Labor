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

# Model definition
#---------------------------------------
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
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)#,
            #nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 10)#,
            #nn.Softmax(dim=1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        encoded = self.encoder(x)
        classes = self.softmax(encoded)
        decoded = self.decoder(classes)
        classed = self.classifier(encoded)
        return decoded, classed

# Load Model
#---------------------------------------
model = torch.load(r'model_conv_autoencoder_mnist.pth')

# Create noisy image
#---------------------------------------
batch_size = 4
transform=T.ToTensor()
mnist_data = dset.MNIST(root=r'./data', train=False, download=False, transform=transform)
loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

image, label = next(iter(loader))

print(image[0].size())
test_img = []
blured_test_img = []

for i in range(batch_size):
    test_img.append(image[i].reshape(28,28,1).numpy())
    blured_test_img.append(ski.filters.gaussian(image[i].reshape(28,28,1).numpy(), sigma=1))

plt.figure(figsize=(batch_size, 2))
plt.gray()
for i in range(batch_size):
    plt.subplot(2,batch_size,i+1)
    plt.title(label[i].numpy(), fontweight='bold')
    plt.imshow(test_img[i])

for i in range(4):
    plt.subplot(2,batch_size, batch_size+i+1)
    #plt.title(label[i].numpy(), fontweight='bold')
    plt.imshow(blured_test_img[i])

plt.show()

'''plt.figure()
plt.imshow(image[0].reshape(28,28,1))
plt.show()'''
