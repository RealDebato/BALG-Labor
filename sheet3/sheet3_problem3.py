import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


#---------------------------------------------------------------------------------
# classes
#---------------------------------------------------------------------------------





#---------------------------------------------------------------------------------
# functions
#---------------------------------------------------------------------------------
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def norm_img(data_train, data_test):
    mean_img = np.mean(data_train, axis=0)
    data_train_norm = data_train - mean_img
    data_test_norm = data_test - mean_img
    data_train_norm = np.divide(data_train_norm, 255.)
    data_test_norm = np.divide(data_test_norm, 255.)

    return data_train_norm, data_test_norm

def features_HOG(data_train, labels_train):

def features_hist_hsv(data_train, labels_train):

def rgb_to_hsv(pixel_data):


#---------------------------------------------------------------------------------
# data
#---------------------------------------------------------------------------------

training = unpickle(R'sheet3\CIFAR\data_batch_1.bin')
testing = unpickle(R'sheet3\CIFAR\test_batch.bin')
validation = unpickle(R'sheet3\CIFAR\data_batch_2.bin')

data_train = np.asarray(training[b'data'])
labels_train = np.asarray(training[b'labels'])
data_test = np.asarray(testing[b'data'])
labels_test = np.asarray(testing[b'labels'])
data_validate = np.asarray(validation[b'data'])[0:1000]
labels_validate = np.asarray(validation[b'labels'])[0:1000]


#---------------------------------------------------------------------------------
# main
#---------------------------------------------------------------------------------







#---------------------------------------------------------------------------------
# output
#---------------------------------------------------------------------------------







#---------------------------------------------------------------------------------
# end
#---------------------------------------------------------------------------------


plt.show()
print('La fin')
cv2.waitKey(0)