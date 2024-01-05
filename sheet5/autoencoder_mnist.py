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



#---------------------------------------------------------
# Functions
#---------------------------------------------------------


#---------------------------------------------------------
# Classes
#---------------------------------------------------------


#---------------------------------------------------------
# Main
#---------------------------------------------------------

# Download Dataset MNIST
#---------------------------------------

train = torch.utils.data.DataLoader(
    dset.MNIST('data', train=True, download=download,
                transform=T.Compose([T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
test = torch.utils.data.DataLoader(
    dset.MNIST('data', train=False,
                transform=T.Compose([T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

