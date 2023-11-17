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

def plot_image_to_3D(plot_3d):

    plot_3d = np.asarray(plot_3d)

    x = np.linspace(0, plot_3d.shape[1]-1, plot_3d.shape[1])
    y = np.linspace(0, plot_3d.shape[0]-1, plot_3d.shape[0])
    X, Y = np.meshgrid(x , y)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, plot_3d, cmap=plt.cm.Blues, linewidth=0)

#---------------------------------------------------------------------------------------------------
# globals



#---------------------------------------------------------------------------------------------------
# images

img = cv2.imread('sheet2\Test_Images\overlapping_circles.png', 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#---------------------------------------------------------------------------------------------------
# main

dmap = cv2.distanceTransform(img, cv2.DIST_L2, 3)
local_max = ski.feature.peak_local_max(dmap, min_distance=80)

background = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
seed = np.zeros_like(img)
seed[tuple(local_max.T)] = 255
#seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)))
flood_area = cv2.subtract(background, seed)
flood_area = np.abs(255-flood_area)

flood_area = cv2.subtract(background, seed)

_, labels = cv2.connectedComponents(seed)
labels = labels + 1

labels[flood_area==255] = 0


labels = np.int32(labels)
img = cv2.merge((img, np.zeros_like(img), np.zeros_like(img)))


watershed = cv2.watershed(img, labels)
plt.imshow(watershed.astype(np.uint8))
plt.show

#---------------------------------------------------------------------------------------------------
# output

plot_image_to_3D(watershed)

cv2.imshow('Watershed', watershed.astype(np.uint8))

plt.imshow(watershed)
plt.title("Watershed")
plt.show




#---------------------------------------------------------------------------------------------------
# main-end

plt.show
print('La fin')
cv2.waitKey(0)
