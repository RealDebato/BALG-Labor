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

pixel_data = training_batch

num_images = pixel_data.shape[0]
print(num_images)
r, g, b = np.split(pixel_data, 3, axis=1)
rgb = np.stack((r, g, b), axis=2)
#rgb = np.split(rgb, num_images, axis=0)
print(rgb)    
hog = ski.feature.hog(rgb, orientations=1, pixels_per_cell=(1, 2), cells_per_block=(1, 1), visualize=False, channel_axis=2)
print(hog)
