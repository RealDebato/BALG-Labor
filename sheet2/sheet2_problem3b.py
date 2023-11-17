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

img = cv2.imread(R'sheet2\FurtherImages\pills.jpg', 0)
_, img_b = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

seed_b = ski.morphology.erosion(img_b, ski.morphology.square(7)).astype(np.double)
footprint = img_b.astype(np.double)
img_reconstruction_b = ski.morphology.reconstruction(seed_b, footprint, 'dilation').astype(np.double)

#seed_b = ski.morphology.dilation(img_reconstruction_b, ski.morphology.square(2)).astype(np.double)
#footprint = img_reconstruction_b.astype(np.double)
#img_reconstruction_b = ski.morphology.reconstruction(seed_b, footprint, 'erosion').astype(np.double)

img_reconstruction_b = np.uint8(img_reconstruction_b)
holeless = ski.morphology.remove_small_holes(img_reconstruction_b, area_threshold=3)
img_reconstruction_b[holeless==1] = 255

#img_reconstruction_b = np.uint8(img_reconstruction_b)
#dirtless = ski.morphology.remove_small_objects(img_reconstruction_b, min_size=10)
#img_reconstruction_b[dirtless==0] = 0

#img_reconstruction_b = ski.morphology.closing(img_reconstruction_b, np.ones((3, 3), np.uint8))
#img_reconstruction_b = ski.morphology.opening(img_reconstruction_b, np.ones((3, 3), np.uint8))



#---------------------------------------------------------------------------------------------------
# main
def watershed_full(img, switch):
    dmap = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    local_max = ski.feature.peak_local_max(dmap, min_distance=10)

    seed = np.zeros_like(img)
    seed[tuple(local_max.T)] = 255
    flood_area = cv2.subtract(img, seed)

    _, labels = cv2.connectedComponents(seed)
    labels = labels + 1

    labels[flood_area==255] = 0

    labels = np.int32(labels)
    img = cv2.merge((img, np.zeros_like(img), np.zeros_like(img)))

    watershed = cv2.watershed(img, labels)
    seg_img = (watershed/watershed.max()) * 255
    seg_img = np.uint8(seg_img)
    if switch==0:
        return seg_img
    elif switch==1:
        return watershed.astype(np.int32)
 

#---------------------------------------------------------------------------------------------------
# output

segmented_img_uint8 = watershed_full(img_reconstruction_b, 0)
#segmented_img_int32 = watershed_full(img, 1)

#cv2.imshow('Watershed', segmented_img_uint8)
cv2.imshow('Binary', img_b)
cv2.imshow('Recon Binary', img_reconstruction_b.astype(np.uint8))
cv2.imshow('Orginal', img)
#cv2.imshow(segmented_img_uint8)

#plot_image_to_3D(segmented_img_int32)

plt.figure()
plt.imshow(segmented_img_uint8)
plt.show()

#plot_image_to_3D(labels)




#---------------------------------------------------------------------------------------------------
# main-end

plt.show()
print('La fin')
cv2.waitKey(0)
