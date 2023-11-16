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
    ax.plot_surface(X, Y, plot_3d, cmap=plt.cm.gray, linewidth=0)

#---------------------------------------------------------------------------------------------------
# globals



#---------------------------------------------------------------------------------------------------
# images

img = cv2.imread('sheet2\Test_Images\overlapping_circles.png', 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#---------------------------------------------------------------------------------------------------
# main

dmap = cv2.distanceTransform(img, cv2.DIST_L2, 3)
#_, foreground_save = cv2.threshold(dmap, 0.45 * dmap.max(), 255, 0)                 # Funktioniert nicht, weil die Distanzwerte zu verschieden sind
#background_save = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
#unbekannt = cv2.subtract(background_save, foreground_save)
#dmap_blur = cv2.medianBlur(dmap.astype(np.uint8), 27)
local_max = ski.feature.peak_local_max(dmap, min_distance=80)
local_max[:, [0, 1]] = local_max[:, [1, 0]]
print(local_max)
img_local_max = dmap
for coord in local_max:
    cv2.circle(img_local_max, coord, 10, 255, -1)

#---------------------------------------------------------------------------------------------------
# output

plot_image_to_3D(img_local_max)

cv2.imshow('Orginal', img)
cv2.imshow('Distance Map', dmap.astype(np.uint8))
#plt.imshow(dmap)



#---------------------------------------------------------------------------------------------------
# main-end

plt.show()
print('La fin')
cv2.waitKey(0)
