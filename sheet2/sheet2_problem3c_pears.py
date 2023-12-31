import numpy as np
import math
import scipy
from scipy import ndimage
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

img = cv2.imread(R'sheet2\FurtherImages\pears.png', 0)

#---------------------------------------------------------------------------------------------------
# main

img_blured = ski.filters.gaussian(img, 5)
gradient = cv2.morphologyEx(img_blured, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
gradient = np.uint8(gradient*255)

watershed_on_gradient = ski.segmentation.watershed(gradient, watershed_line=True)

#---------------------------------------------------------------------------------------------------
# output

plt.figure()
plt.imshow(img)
plt.title('Orginal')

plt.figure()
plt.imshow(gradient)
plt.title('Gradient')

plt.figure()
plt.imshow(watershed_on_gradient)
plt.title('Watershed on gradient map')

plot_image_to_3D(gradient)

#---------------------------------------------------------------------------------------------------
# main-end

plt.show()
print('La fin')
cv2.waitKey(0)