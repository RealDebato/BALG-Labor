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

nr_bild = 0


#---------------------------------------------------------------------------------------------------
# main

label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
print(label_decoder[b'label_names'][6])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')

pixel_data_batch_1 = data_batch_1[b'data']
labels_data_batch_1 = data_batch_1[b'labels']


Bild1 = image_from_cifar10(pixel_data_batch_1[nr_bild,:])

Entscheidungsraum_x = np.linspace(0, 3072, 1)
Entscheidungsraum_y = np.linspace(0, 256, 1)
Entscheidungsraum_z = np.linspace(0, 9999, 1)
Entscheidungsraum = np.dstack(Entscheidungsraum_x, Entscheidungsraum_y, Entscheidungsraum_z)

for Bild_nummer in range(0, 9999):
    Entscheidungsraum[:,:,Bild_nummer] = [Entscheidungsraum_x, pixel_data_batch_1[Bild_nummer,:], Bild_nummer]



# Wie sehen die Entscheidungsräume aus??
# Features sollen die Pixel sein, d. h.
# Entscheidungsraum x: Pixelnummer entsprechend aus dem .bin
# Entscheidungsraum y: Pixelwert
# Entscheidungsraum z: Bildnummer


#---------------------------------------------------------------------------------------------------
# output

'''plt.figure
plt.imshow(Bild1)
plt.title(label_decoder[b'label_names'][labels_data_batch_1[nr_bild]])'''

plt.figure
plt.imshow(Entscheidungsraum)
plt.title('Entscheidungsraum')


#---------------------------------------------------------------------------------------------------
# end

plt.show()
print('La fin')
cv2.waitKey(0)
