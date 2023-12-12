import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import torch.nn.functional as F 

import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


acc_iter = []
loss_iter = []

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

def normalize_minmax(data):
    data = np.asarray(data, dtype=np.float64)
    mean = np.mean(data, dtype=np.float64)
    data -= mean

    max_hist = np.max(data)
    min_hist = np.min(data)
    

    if max_hist > np.abs(min_hist):
        scale = max_hist
    else:
        scale = np.abs(min_hist)

    data_norm = data/scale

    return data_norm



#--------------------------------------------------------------------------------------
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#--------------------------------------------------------------------------------------

def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1)

def init(net):
    if isinstance(net, nn.Linear):
        nn.init.xavier_uniform_(net.weight)
        net.bias.data.fill_(0)

class MultiLayerFC(nn.Module):
    def __init__(self, input_size, hidden_input, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_input[0]),
            nn.LogSoftmax(),
            nn.Linear(hidden_input[0], hidden_input[1]),
            nn.LogSoftmax(),
            nn.Linear(hidden_input[1], hidden_input[2]),
            nn.LogSoftmax(),
            nn.Linear(hidden_input[2], num_classes),
            nn.LogSoftmax()
            )

    def forward(self, x):
        x = flatten(x)
        scores = self.layers(x)
        return scores
    
    
    
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


def train(model, optimizer, epochs=5, lr=1e-4):
    model = model.to(device=device)
    for e in range(epochs):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        for t, (x, y) in enumerate(trainloader):
            model.train()  
            x = x.to(device=device, dtype=torch.float32)  
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % 75 == 0:
                print(f'Epoch {e}, Iteration {t}, loss = {loss.item()}')
                loss_iter.append(loss.item())
                print(f'accuracy on validation')
                check_accuracy(validationloader, model)
                
        print(f'accuracy on test')
        #check_accuracy(testloader,model)
        lr = lr*0.9
        print(f'learning rate {lr:.6f} at epoch {e}')
        



#----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.multiprocessing.freeze_support()


    pixel_data_train, labels_data_train, pixel_data_val, labels_data_val, pixel_data_test, labels_data_test = unpickle_cifar(48000, 2000, 10000)

    data_load_train = prepare_for_loader(normalize_minmax(pixel_data_train), labels_data_train)
    data_load_val = prepare_for_loader(normalize_minmax(pixel_data_val), labels_data_val)
    data_load_test = prepare_for_loader(normalize_minmax(pixel_data_test), labels_data_test)

    batch_size = 64

    trainloader = DataLoader(data_load_train, batch_size=batch_size)
    validationloader = DataLoader(data_load_val, batch_size=batch_size)
    testloader = DataLoader(data_load_test, batch_size=batch_size)

    abc = enumerate(trainloader)

    hidden_layers = [300, 100, 30, 20]
    learning_rate = 1e-4
    model = MultiLayerFC(3072, hidden_layers, 10)
    model.layers.apply(init)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train(model, optimizer, epochs = 20, lr=learning_rate)

    
    fig, ax1 = plt.subplots()

    fig.suptitle('4 Layer - LogSoftmax - lr = 1e-4 (Pixels)')

    color = 'tab:blue'
    ax1.set_ylabel('acc', color=color)
    ax1.plot(range(0, len(acc_iter)), acc_iter, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('loss', color=color)
    ax2.plot(range(0, len(loss_iter)), loss_iter, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.show()





