import torch
#assert '.'.join(torch.__version__.split('.')[:2]) == '1.4'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import torch.nn.functional as F  # useful stateless functions

import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


#--------------------------------------------------------------------------------------

def hist_hue(pixel_data_cifar):
    
    num_img = pixel_data_cifar.shape[0]
    pixel_data_cifar = np.reshape(pixel_data_cifar, (num_img, 1024, 3))
    hue = cv2.cvtColor(pixel_data_cifar, cv2.COLOR_RGB2HSV)[:,:,0]

    Hue_split = np.split(hue, num_img, axis=0)

    hist = []

    for img in range(0, num_img):
        hist_row, _ = np.histogram(Hue_split[img], bins=180, range=(0, 179), density=True)
        hist = np.append(hist, hist_row)

    hist = np.reshape(hist, (num_img, 180))
    
    return hist

def classification_hog(pixel_data):
    
    pixels_per_cell = (8, 8)    # --> 16 cells
    cells_per_block = (4, 4)    # --> 1 block
    orientations = 10
    num_images = pixel_data.shape[0]
    len_hog = int(orientations * 1024/(pixels_per_cell[0]*pixels_per_cell[1]))
    pixel_data_hog = pixel_data.reshape(num_images, 3, 32, 32).transpose(0,2,3,1).astype("float")
    hog_list = []
    for img in range(0, num_images):
        hog_feats = ski.feature.hog(pixel_data_hog[img, :], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L1', visualize=False, feature_vector=True, channel_axis=2)
        hog_list = np.append(hog_list, hog_feats)

    hog_list = np.reshape(hog_list, (num_images, len_hog))
    
    return hog_list

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_cifar(num_train, num_test, num_val):

    if num_train + num_val > 50000:
        num_val = 50000 - num_train

    if num_test > 10000:
        num_test = 10000


    data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')   
    data_batch_2 = unpickle(R'sheet3\CIFAR\data_batch_2.bin')   
    data_batch_3 = unpickle(R'sheet3\CIFAR\data_batch_3.bin')   
    data_batch_4 = unpickle(R'sheet3\CIFAR\data_batch_4.bin')
    data_batch_5 = unpickle(R'sheet3\CIFAR\data_batch_5.bin')
    data_batch_test = unpickle(R'sheet3\CIFAR\test_batch.bin')

    pixel_data_batch_1 = np.asarray(data_batch_1[b'data'])
    pixel_data_batch_2 = np.asarray(data_batch_2[b'data'])
    pixel_data_batch_3 = np.asarray(data_batch_3[b'data'])
    pixel_data_batch_4 = np.asarray(data_batch_4[b'data'])
    pixel_data_batch_5 = np.asarray(data_batch_5[b'data'])
    

    pixel_data = np.concatenate((pixel_data_batch_1, pixel_data_batch_2, pixel_data_batch_3, pixel_data_batch_4, pixel_data_batch_5), axis=0)


    labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])   
    labels_data_batch_2 = np.asarray(data_batch_2[b'labels'])   
    labels_data_batch_3 = np.asarray(data_batch_3[b'labels'])   
    labels_data_batch_4 = np.asarray(data_batch_4[b'labels'])  
    labels_data_batch_5 = np.asarray(data_batch_5[b'labels'])  
    

    labels_data = np.concatenate((labels_data_batch_1, labels_data_batch_2, labels_data_batch_3, labels_data_batch_4, labels_data_batch_5))


    pixel_data_test = np.asarray(data_batch_test[b'data'])[0:num_test,:]
    labels_data_test = np.asarray(data_batch_test[b'labels']) [0:num_test]

    pixel_data_train = pixel_data[0:num_train,:]
    labels_data_train = labels_data[0:num_train]

    pixel_data_val = pixel_data[50001-num_val:-1,:]
    labels_data_val = labels_data[50001-num_val:-1]

    return pixel_data_train, labels_data_train, pixel_data_val, labels_data_val, pixel_data_test, labels_data_test

def prepare_for_loader(inputs, targets):
    loader_data = []
    for i in range(inputs.shape[0]):
        loader_data.append([inputs[i,:], targets[i]])
    return loader_data

def normalize_std(data):
    data = np.asarray(data, dtype=np.float64)
    mean = np.mean(data, dtype=np.float64)
    data -= mean

    max_hist = np.max(data)
    min_hist = np.min(data)
    std = np.std(data)

    '''if max_hist > np.abs(min_hist):
        scale = max_hist
    else:
        scale = np.abs(min_hist)'''

    data_norm = data/std

    return data_norm



#--------------------------------------------------------------------------------------
USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

#--------------------------------------------------------------------------------------

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # http://pytorch.org/docs/master/nn.html#torch-nn-init 
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, x):
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores
    
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train(model, optimizer, epochs=5):
    """
    Returns nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(trainloader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update!!!!!!!!
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 200 == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e,t, loss.item()))
                check_accuracy(validationloader, model)
                print()
        check_accuracy(testloader,model)
        print()



#----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #torch.manual_seed(0)
    torch.multiprocessing.freeze_support()
    
    pixel_data_train, labels_data_train, pixel_data_val, labels_data_val, pixel_data_test, labels_data_test = unpickle_cifar(10000, 1000, 2000)

    pixel_data_train = np.concatenate((normalize_std(hist_hue(pixel_data_train)), normalize_std(classification_hog(pixel_data_train))), axis=1)
    pixel_data_val = np.concatenate((normalize_std(hist_hue(pixel_data_val)), normalize_std(classification_hog(pixel_data_val))), axis=1)
    pixel_data_test = np.concatenate((normalize_std(hist_hue(pixel_data_test)), normalize_std(classification_hog(pixel_data_test))), axis=1)

    data_load_train = prepare_for_loader(pixel_data_train, labels_data_train)
    data_load_val = prepare_for_loader(pixel_data_val, labels_data_val)
    data_load_test = prepare_for_loader(pixel_data_test, labels_data_test)

    batch_size = 64

    trainloader = DataLoader(data_load_train, batch_size=batch_size, shuffle=True, num_workers=1)
    validationloader = DataLoader(data_load_val, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(data_load_test, batch_size=batch_size, shuffle=True, num_workers=1)

    hidden_layer_size = 100
    learning_rate = 1e-2
    model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train(model, optimizer, epochs = 20)



