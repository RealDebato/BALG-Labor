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

def rgb_to_hue(pixel_data):
    
    r, g, b = np.split(pixel_data, 3, axis=1)
    rgb = np.stack((r, g, b), axis=2)
    max_color_channel = np.argmax(rgb, axis=2) 

    Hue = np.zeros_like(max_color_channel)
    Hue[max_color_channel==0] = (rgb[..., 1]-rgb[..., 2])[max_color_channel==0]
    Hue[max_color_channel==1] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==1]
    Hue[max_color_channel==2] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==2]

    return Hue




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

data_train_hue = rgb_to_hue(data_train)
data_test_hue = rgb_to_hue(data_test)
data_validate_hue = rgb_to_hue(data_validate)

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