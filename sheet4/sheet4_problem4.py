import numpy as np
import math
import os
import shutil
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time
import torch
import torchvision


#---------------------------------------------------------------------------------------------------
# classes
#---------------------------------------------------------------------------------------------------
  


#---------------------------------------------------------------------------------------------------
# functions
#---------------------------------------------------------------------------------------------------

def cifar_data_to_hue(pixel_data_cifar):
    num_img = pixel_data_cifar.shape[0]
    pixel_data_cifar = np.reshape(pixel_data_cifar, (num_img, 1024, 3))
    hue = cv2.cvtColor(pixel_data_cifar, cv2.COLOR_RGB2HSV)[:,:,0]

    return hue

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
    pixel_data = pixel_data.reshape(num_images, 3, 32, 32).transpose(0,2,3,1).astype("float")
    hog_list = []
    for img in range(0, num_images):
        hog_feats = ski.feature.hog(pixel_data[img, :], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L1', visualize=False, feature_vector=True, channel_axis=2)
        hog_list = np.append(hog_list, hog_feats)
    
    hog_list = np.reshape(hog_list, (num_images, len_hog))
    
    return hog_list

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
# file structure
#---------------------------------------------------------------------------------------------------
'''try:    
    shutil.rmtree(R'sheet4\data')
except:
    pass

try:
    os.makedirs(R'sheet4\data')

    open(R'sheet4\data\pixel_data_train.npy', 'x')
    open(R'sheet4\data\pixel_data_test.npy', 'x')
    open(R'sheet4\data\pixel_data_validation.npy', 'x')

    open(R'sheet4\data\hue_data_train.npy', 'x')
    open(R'sheet4\data\hue_data_test.npy', 'x')
    open(R'sheet4\data\hue_data_validation.npy', 'x')

    open(R'sheet4\data\hog_data_train.npy', 'x')
    open(R'sheet4\data\hog_data_test.npy', 'x')
    open(R'sheet4\data\hog_data_validation.npy', 'x')

    open(R'sheet4\data\features_data_train.npy', 'x')
    open(R'sheet4\data\features_data_test.npy', 'x')
    open(R'sheet4\data\features_data_validation.npy', 'x')

except:
    pass'''
#---------------------------------------------------------------------------------------------------
# pre-processing data
#---------------------------------------------------------------------------------------------------

numbers_train = 5000
numbers_validate = 1000
numbers_test = 500



label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
#print(label_decoder[b'label_names'])

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
pixel_data_batch_test = np.asarray(data_batch_test[b'data'])

pixel_data = np.concatenate((pixel_data_batch_1, pixel_data_batch_2, pixel_data_batch_3, pixel_data_batch_4, pixel_data_batch_5), axis=0)

pixel_data_training = pixel_data[0:numbers_train, :]
pixel_data_testing = pixel_data_batch_test[0:numbers_test, :]
pixel_data_validation = pixel_data[50000-numbers_validate:-1, :]
#print('Pixel Data Batch 1 ',pixel_data_training)
#print('Pixel Data Batch 1 shape ',pixel_data_training.shape)

labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])   
labels_data_batch_2 = np.asarray(data_batch_2[b'labels'])   
labels_data_batch_3 = np.asarray(data_batch_3[b'labels'])   
labels_data_batch_4 = np.asarray(data_batch_4[b'labels'])  
labels_data_batch_5 = np.asarray(data_batch_5[b'labels'])  
labels_data_batch_test = np.asarray(data_batch_test[b'labels'])  

labels_data = np.concatenate((labels_data_batch_1, labels_data_batch_2, labels_data_batch_3, labels_data_batch_4, labels_data_batch_5))

labels_data_training = labels_data_batch_1[0:numbers_train]
labels_data_testing = labels_data_batch_2[0:numbers_test]
labels_data_validation = labels_data_batch_3[numbers_train+1:-1]

mean_pixel_data_training = np.mean(pixel_data_training, axis=0)

pixel_data_training_normalized = np.subtract(pixel_data_training, mean_pixel_data_training)
pixel_data_testing_normalized = np.subtract(pixel_data_testing, mean_pixel_data_training)
pixel_data_validation_normalized = np.subtract(pixel_data_validation, mean_pixel_data_training)

pixel_data_training_normalized = np.divide(pixel_data_training_normalized, 255.)
pixel_data_testing_normalized = np.divide(pixel_data_testing_normalized, 255.)
pixel_data_validation_normalized = np.divide(pixel_data_validation_normalized, 255)

'''print('Start writing pixels')
np.save(R'sheet4\data\pixel_data_train.npy', pixel_data_training_normalized)
np.save(R'sheet4\data\pixel_data_test.npy', pixel_data_testing_normalized)
np.save(R'sheet4\data\pixel_data_validation.npy', pixel_data_validation_normalized)'''

# new features ------------------------------------------------
# hue
hue_training = hist_hue(pixel_data_training)
hue_testing = hist_hue(pixel_data_testing)
hue_validation = hist_hue(pixel_data_validation)

mean_hue = np.mean(hue_training, axis=0)

hue_training_normalized = np.subtract(hue_training, mean_hue)
hue_testing_normalized = np.subtract(hue_testing, mean_hue)
hue_validation_normalized = np.subtract(hue_validation, mean_hue)

hue_training_normalized = np.divide(hue_training_normalized, 255.)
hue_testing_normalized = np.divide(hue_testing_normalized, 255.)
hue_validation_normalized = np.divide(hue_validation_normalized, 255)

'''print('Start writing hue')
np.save(R'sheet4\data\hue_data_train.npy', hue_training_normalized)
np.save(R'sheet4\data\hue_data_test.npy', hue_testing_normalized)
np.save(R'sheet4\data\hue_data_validation.npy', hue_validation_normalized)'''

# hog
hog_training = classification_hog(pixel_data_training)
hog_testing = classification_hog(pixel_data_testing)
hog_validation = classification_hog(pixel_data_validation)

mean_hog = np.mean(hog_training)

hog_training_normalized = np.subtract(hog_training, mean_hog)
hog_testing_normalized = np.subtract(hog_testing, mean_hog)
hog_validation_normalized = np.subtract(hog_validation, mean_hog)

max_hog = np.amax(hog_training_normalized)
min_hog = np.abs(np.amin(hog_training_normalized))
if max_hog > min_hog:
    scale = max_hog
else:
    scale = min_hog

hog_training_normalized = np.divide(hog_training_normalized, scale)
hog_testing_normalized = np.divide(hog_testing_normalized, scale)
hog_validation_normalized = np.divide(hog_validation_normalized, scale)

'''print('Start writing hog')
np.save(R'sheet4\data\hog_data_train.npy', hog_training_normalized)
np.save(R'sheet4\data\hog_data_test.npy', hog_testing_normalized)
np.save(R'sheet4\data\hog_data_validation.npy', hog_validation_normalized)'''


# concatenat features
features_train = np.concatenate((hue_training_normalized, hog_training_normalized), axis=1)
features_test = np.concatenate((hue_testing_normalized, hog_testing_normalized), axis=1)
features_validate = np.concatenate((hue_validation_normalized, hog_validation_normalized), axis=1)

'''print('Start writing features')
np.save(R'sheet4\data\features_data_train.npy', features_train)
np.save(R'sheet4\data\features_data_test.npy', features_test)
np.save(R'sheet4\data\features_data_validation.npy', features_validate)'''



print('Ende')

