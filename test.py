import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


training_batch = [[9, 15, 8, 5, 3, 2],[14, 17, 7, 6, 1, 2],[12, 12, 5, 8, 4, 3],[19, 18, 6, 7, 2, 2],[10, 11, 6, 6, 4, 1]]
test_batch = [[4, 1, 7],[2, 1, 1],[8, 7, 5]]
training_batch = np.asarray(training_batch)
test_batch = np.asarray(test_batch)


num_img, num_pixel = training_batch.shape
#training_batch /= 255
r, g, b = np.split(training_batch, 3, axis=1)
print(r, g, b)
rgb = np.stack((r, g, b), axis=2)
print('RGB:', rgb)
max_color_channel = np.argmax(rgb, axis=2) 
print('max Color:', max_color_channel)

Hue_max_r = rgb[:,:, 1]-rgb[:,:, 2]
Hue_max_g = rgb[:,:, 2]-rgb[:,:, 0]
Hue_max_b = rgb[:,:, 0]-rgb[:,:, 1]

Hue = np.zeros_like(max_color_channel)
Hue[max_color_channel==0] = (rgb[..., 1]-rgb[..., 2])[max_color_channel==0]
Hue[max_color_channel==1] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==1]
Hue[max_color_channel==2] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==2]

print('Hue:', Hue)
