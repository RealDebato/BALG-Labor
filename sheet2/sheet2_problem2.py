import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import time


#---------------------------------------------------------------------------------------------------
# Images

img = cv2.imread('sheet2\Test_Images\particle1.jpg', 0)
img = np.abs(255 - img)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#plt.imshow(img)
cv2.imshow('Img', img)

img2 = cv2.imread('sheet2/Test_Images/electrop.jpg', 0)
img2 = np.abs(255 - img2)
#_, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
#plt.imshow(img2)
cv2.imshow('Img2', img2)

r = 1
kernel = np.ones((3, 3), np.uint8)

#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for binary images
# Startpunkte mit Erosion bestimmen
'''
def recon_by_dilation_binary(img, kernel,  start_iter):

    start = cv2.erode(img, kernel, iterations=start_iter)
    cv2.imshow('Start1', start)

    # Reconstruction by dilation
    Recon = 255 * (np.logical_and(img, cv2.dilate(start, kernel, iterations=1)).astype(np.uint8))
    Recon_old = 255* (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))
    i = 0

    while Recon.all() == Recon_old.all():
        Recon = 255 * (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))
        Recon_old = 255 * (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))
        i = i + 1
        if i == 100:
            break

    return Recon

t1 = time.time()
cv2.imshow('Binary Reconstruction', recon_by_dilation_binary(img, kernel, 7))
t2 = time.time()
print('Time recon binary', t2 - t1)
#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for grey value images

def recon_by_dilation_grey(img, kernel, start_iter):
    start = cv2.erode(img, kernel, iterations=start_iter)
    cv2.imshow('Start2', start)

    # Reconstruction by dilation
    Recon = np.minimum(img, cv2.dilate(start, kernel, iterations=1)).astype(np.uint8)
    Recon_old = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8)
    i = 0

    while Recon.all() == Recon_old.all():
        Recon = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8)
        Recon_old = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8)
        i = i + 1
        if i == 100:
            break

    return Recon

t3 = time.time()
cv2.imshow('Grey Reconstruction', recon_by_dilation_grey(img2, kernel, 7))
t4 = time.time()
print('Time recon grey', t4 - t3)

'''




#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction scikit-images
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for grey value images - scikit

sci_erode = ski.morphology.erosion(img2, ski.morphology.square(15))
cv2.imshow('sci erode', sci_erode)

cv2.imshow('Scikit recon', ski.morphology.reconstruction(sci_erode, img2, 'dilation'))


print('Ende')
cv2.waitKey(0)
