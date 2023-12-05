import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


'''training_batch = [[9, 15, 8, 5, 5, 2],[14, 17, 7, 6, 1, 2],[12, 12, 5, 8, 4, 3],[19, 18, 6, 7, 2, 2],[10, 11, 6, 6, 4, 1]]
test_batch = [[4, 1, 7],[2, 1, 1],[8, 7, 5]]
training_batch = np.asarray(training_batch)
test_batch = np.asarray(test_batch)'''


def cifar_data_to_hue(pixel_data_cifar):
    num_img = pixel_data_cifar.shape[0]
    pixel_data_cifar = np.reshape(pixel_data_cifar, (num_img, 1024, 3))
    hue = cv2.cvtColor(pixel_data_cifar, cv2.COLOR_RGB2HSV)[:,:,0]

    return hue

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')   # --> Train
pixel_data_img0_5 = data_batch_1[b'data']
hue = cifar_data_to_hue(pixel_data_img0_5)
print('max', np.max(hue))
print('min', np.min(hue))
print('End')