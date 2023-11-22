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
#print(label_decoder[b'label_names'][6])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')

pixel_data_batch_1 = np.asarray(data_batch_1[b'data'])
pixel_data_batch_1 = np.split(pixel_data_batch_1, 10000)    # hier steht jeweils ein Array mit allen Pixelwerten aus einem Bild (1. Array --> 1. Bild, 2. Array --> 2. Bild, usw.)
labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])


print(pixel_data_batch_1.size)

Bild1 = image_from_cifar10(pixel_data_batch_1[nr_bild,:])


# Wie sehen die Entscheidungsräume aus??
# Features sollen die Pixel sein, d. h. ich machs einfach und nehm den Mittelwert aller Pixel im gleichen Farbraum

feature_space_x_axis = np.arange(0, 1023)   #red
feature_space_y_axis = np.arange(0, 1023)   #green
feature_space_z_axis = np.arange(0, 1023)   #blue

feature_space = np.full((1024, 1024, 1024), fill_value=-1)

X, Y, Z = np.meshgrid(feature_space_x_axis, feature_space_y_axis, feature_space_z_axis)



# feature space mit den Trainings-Daten befüllen
for img_number in range(0, 10000):
    feature_space_x_data = (np.split(pixel_data_batch_1[img_number], 3))[0]
    feature_space_y_data = (np.split(pixel_data_batch_1[img_number], 3))[1]
    feature_space_z_data = (np.split(pixel_data_batch_1[img_number], 3))[2]
    feature_space[feature_space_x_axis, feature_space_y_axis, feature_space_z_axis] = labels_data_batch_1[img_number] # Hier ist der feature space schon 4-dimensional



#---------------------------------------------------------------------------------------------------
# output

feature_space_x_axis = np.arange(0, 1023)   #red
feature_space_y_axis = np.arange(0, 1023)   #green
feature_space_z_axis = np.arange(0, 1023)   #blue

'''plt.figure
plt.imshow(Bild1)
plt.title(label_decoder[b'label_names'][labels_data_batch_1[nr_bild]])'''

'''plt.figure
plt.imshow(Entscheidungsraum)
plt.title('Entscheidungsraum')'''


#---------------------------------------------------------------------------------------------------
# end

plt.show()
print('La fin')
cv2.waitKey(0)
