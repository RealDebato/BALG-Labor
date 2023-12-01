import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


training_batch = [[9, 15, 8, 5, 5, 2],[14, 17, 7, 6, 1, 2],[12, 12, 5, 8, 4, 3],[19, 18, 6, 7, 2, 2],[10, 11, 6, 6, 4, 1]]
test_batch = [[4, 1, 7],[2, 1, 1],[8, 7, 5]]
training_batch = np.asarray(training_batch)
test_batch = np.asarray(test_batch)

pixel_data = training_batch

r, g, b = np.split(pixel_data, 3, axis=1)
rgb = np.stack((r, g, b), axis=2)
max_color_channel = np.argmax(rgb, axis=2) 
num_img = pixel_data.shape[0]

Hue = np.zeros_like(max_color_channel)
Hue[max_color_channel==0] = (rgb[..., 1]-rgb[..., 2])[max_color_channel==0]
Hue[max_color_channel==1] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==1]
Hue[max_color_channel==2] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==2]
print(Hue.shape)

Hue = np.where((Hue<0, Hue+256))
Hue = np.asarray(Hue)
Hue_split = np.split(Hue, num_img, axis=1)
print(Hue_split)
print(len(Hue_split))
hist = []

for img in range(0, num_img):
    hist_row, _ = np.histogram(Hue_split[img], bins=256)
    hist = np.append(hist, hist_row)

hist = np.reshape(hist, (num_img, 256))

print(hist)
print(hist.shape)