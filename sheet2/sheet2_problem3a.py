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

img = cv2.imread(R'sheet2\Test_Images\same_circles.png', 0)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#---------------------------------------------------------------------------------------------------
# main
def watershed_full(img, switch):
    dmap = cv2.distanceTransform(img, cv2.DIST_L2, 3)                   # Distance map wird erstellt
    local_max = ski.feature.peak_local_max(dmap, min_distance=10)       # Locale Maximas aus Distance map

    seed = np.zeros_like(img)
    seed[tuple(local_max.T)] = 255                                      # Locale Maximas in Seed-Bild einfügen
    flood_area = cv2.subtract(img, seed)                                # Flood area entspricht der Zone zwischen der Maximas und dem schwarzen (=0) Hintergrund
                                                                        # hier wird geflutet

    _, labels = cv2.connectedComponents(seed)                           # Alles Maximas nummerieren
    labels = labels + 1                                                 # Labels werden um +1 erhöht, sodass die Label-Nummer 0 frei wird

    labels[flood_area==255] = 0                                         # Alles was jetzt im Labelbild 0 ist wird geflutet

    labels = np.int32(labels)
    img = cv2.merge((img, np.zeros_like(img), np.zeros_like(img)))      # OpenCV watershed ist doof und braucht zwingend ein 3-channel Bild

    watershed = cv2.watershed(img, labels)
    seg_img = (watershed/watershed.max()) * 255
    seg_img = np.uint8(seg_img)
    if switch==0:
        return seg_img
    elif switch==1:
        return watershed.astype(np.int32)
 

#---------------------------------------------------------------------------------------------------
# output

segmented_img_uint8 = watershed_full(img, 0)
segmented_img_int32 = watershed_full(img, 1)

cv2.imshow('Watershed', segmented_img_uint8)
cv2.imshow('Orginal', img)

#plot_image_to_3D(segmented_img_int32)

plt.figure()
plt.imshow(segmented_img_int32)
plt.show()

#plot_image_to_3D(labels)




#---------------------------------------------------------------------------------------------------
# main-end

plt.show()
print('La fin')
cv2.waitKey(0)
