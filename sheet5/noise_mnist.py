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


# Functions

def prepare_for_loader(inputs, targets):
    loader_data = []
    for i in range(len(inputs)):
        loader_data.append([inputs[i], targets[i]])
    return loader_data



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
t_blured_test_img = []

for i in range(batch_size):
    plot_img = image[i].reshape(28,28,1).numpy()
    test_img.append(plot_img)
    noise_img = plot_img + np.random.normal(loc=0, scale=0.2, size=(28,28,1))
    noise_img = np.where(noise_img>1, 1, noise_img)
    noise_img = np.where(noise_img<0, 0, noise_img)
    t_noise_img = torch.from_numpy(noise_img)
    t_noise_img_reshape = t_noise_img.view(1,28,28)
    blured_test_img.append(noise_img)
    t_blured_test_img.append(t_noise_img_reshape)

# Create Dataloader for eval

eval_loader = DataLoader(prepare_for_loader(t_blured_test_img, label),batch_size=1, shuffle=False)


# Load Model
#---------------------------------------
model = torch.load(r'model_conv_autoencoder_mnist.pth', map_location=torch.device('cpu'))
model.eval()
denoised = []
predictions = []
for i, (image, target) in enumerate(eval_loader):
    image = image.to(torch.float32)
    denoised_img, prediction = model(image)
    np_denoised_img = denoised_img.detach().numpy()
    np_denoised_img.astype(np.float32)
    np_denoised_img = np.transpose(np_denoised_img, (0, 2, 3, 1))
    denoised.append(np_denoised_img[0,:])
    predictions.append(prediction)



'''for noise in blured_test_img:
    
    t_noise = torch.from_numpy(noise)
    print('t from np', t_noise.size())
    t_noise_reshape = t_noise.view(1, 28, 28)
    print('reshape', t_noise_reshape.size())

    m_denoised = model(t_noise_reshape)
    print(m_denoised.size())
    m_denoised.reshape(28,28,1).detach().numpy()
    denoised.append(m_denoised)'''
    
#predictions = np.argmax(predictions.numpy(), axis=1)

plt.figure(figsize=(batch_size, 3))
plt.gray()
for i in range(batch_size):
    plt.subplot(3,batch_size,i+1)
    plt.title(label[i].numpy(), fontweight='bold')
    plt.imshow(test_img[i])

for i in range(batch_size):
    plt.subplot(3,batch_size, batch_size+i+1)
    #plt.title(label[i].numpy(), fontweight='bold')
    plt.imshow(blured_test_img[i])

for i in range(batch_size):
    plt.subplot(3,batch_size, 2*batch_size+i+1)
    #plt.title(predictions[i], fontweight='bold')
    plt.imshow(denoised[i])

plt.show()

'''plt.figure()
plt.imshow(image[0].reshape(28,28,1))
plt.show()'''
