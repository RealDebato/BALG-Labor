import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time

#---------------------------------------------------------------------------------------------------
# functions

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def image_from_cifar10(data_vektor):
   
    img_flattend = np.split(data_vektor, 3)
    red = img_flattend[0]
    green = img_flattend[1] 
    blue = img_flattend[2]

    red = np.split(red, 32)
    green = np.split(green, 32)
    blue = np.split(blue, 32)
    
    img = np.dstack((red, green, blue))

    return img.astype(np.uint8)

#---------------------------------------------------------------------------------------------------
# globals

nr_bild = 999


#---------------------------------------------------------------------------------------------------
# main

label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
print(label_decoder[b'label_names'][6])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')

pixel_data_batch_1 = data_batch_1[b'data']
labels_data_batch_1 = data_batch_1[b'labels']


Bild1 = image_from_cifar10(pixel_data_batch_1[nr_bild,:])

#---------------------------------------------------------------------------------------------------
# output

plt.figure
plt.imshow(Bild1)
plt.title(label_decoder[b'label_names'][labels_data_batch_1[nr_bild]])


#---------------------------------------------------------------------------------------------------
# end

plt.show()
print('La fin')
cv2.waitKey(0)
