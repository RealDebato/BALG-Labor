import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import time

#---------------------------------------------------------------------------------------------------
# Globals

v_time_selfmade_binary = []
v_time_selfmade_grey = []

v_time_scikit_binary = []
v_time_scikit_grey = []

kernel = np.ones((3, 3), np.uint8)


#---------------------------------------------------------------------------------------------------
# Functions

def recon_by_dilation_binary(img, kernel,  start_iter):

    start = cv2.erode(img, kernel, iterations=start_iter)
    #cv2.imshow('Start1', start)

    # Reconstruction by dilation
    t0_selfmade = time.time()
    Recon = 255 * (np.logical_and(img, cv2.dilate(start, kernel, iterations=1)).astype(np.uint8))
    Recon_old = 255* (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))
    i = 0

    while np.array_equal(Recon, Recon_old) == False:
        Recon = 255 * (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))
        Recon_old = 255 * (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))

    
    t1_selfmade = time.time()
    return Recon, t1_selfmade - t0_selfmade

def recon_by_dilation_grey(img, kernel, start_iter):
    img = img.astype(np.double)
    start = cv2.erode(img, kernel, iterations=start_iter)
    #cv2.imshow('Start2', start)

    # Reconstruction by dilation
    t0_selfmade = time.time()
    Recon = np.minimum(img, cv2.dilate(start, kernel, iterations=1)).astype(np.double)
    Recon_old = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.double)
    i = 0

    while np.array_equal(Recon, Recon_old) == False:
        Recon = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.double)
        Recon_old = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.double)
        i = i + 1
        if i == 100:
            break
    t1_selfmade = time.time()
    return Recon.astype(np.uint8), t1_selfmade - t0_selfmade

#---------------------------------------------------------------------------------------------------
# Images

img = cv2.imread('sheet2\Test_Images\particle1.jpg', 0)
img = np.abs(255 - img)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#plt.imshow(img)
cv2.imshow('Binary', img)

img2 = cv2.imread('sheet2/Test_Images/electrop.jpg', 0)
img2 = np.abs(255 - img2)
#plt.imshow(img2)
cv2.imshow('Grey', img2)

#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for binary images

img_binary_recon_selfmade, time_binary_selfmade = recon_by_dilation_binary(img, kernel, 15)
v_time_selfmade_grey = np.append(v_time_selfmade_grey, time_binary_selfmade)

#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for grey value images

img_grey_recon_selfmade, time_grey_selfmade = recon_by_dilation_grey(img2, kernel, 7)
v_time_selfmade_grey = np.append(v_time_selfmade_grey, time_grey_selfmade)

#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction scikit-images
#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for binary images - scikit
seed_b = ski.morphology.erosion(img, ski.morphology.square(30)).astype(np.double)

footprint = img.astype(np.double)

t0_scikit_binary = time.time()
scikit_reconstruction_b = ski.morphology.reconstruction(seed_b, footprint, 'dilation').astype(np.double)
t1_scikit_binary = time.time()

v_time_scikit_binary = np.append(v_time_scikit_binary, t1_scikit_binary - t0_scikit_binary)

#---------------------------------------------------------------------------------------------------
# Geodesic Reconstruction for grey value images - scikit

seed_g = ski.morphology.erosion(img2, ski.morphology.square(15)).astype(np.double)
footprint = img2.astype(np.double)

t0_scikit_grey = time.time()
scikit_reconstruction_g = ski.morphology.reconstruction(seed_g, footprint, 'dilation').astype(np.double)
t1_scikit_grey = time.time()

v_time_scikit_grey = np.append(v_time_scikit_grey, t1_scikit_grey - t0_scikit_grey)


#---------------------------------------------------------------------------------------------------
# Bildausgabe

cv2.imshow('Binary Reconstruction', img_binary_recon_selfmade)
print('Time binary selfmade', time_binary_selfmade)

cv2.imshow('Grey Reconstruction', img_grey_recon_selfmade)
print('Time grey selfmade', time_grey_selfmade)

cv2.imshow('seed binary', seed_b.astype(np.uint8))
cv2.imshow('Scikit binary recon', scikit_reconstruction_b.astype(np.uint8))
print('Time scikit binary', t1_scikit_binary - t0_scikit_binary)

cv2.imshow('seed grey', seed_g.astype(np.uint8))
cv2.imshow('Scikit grey recon', scikit_reconstruction_g.astype(np.uint8))
print('Time scikit grey', t1_scikit_grey - t0_scikit_grey)


print('Ende')
cv2.waitKey(0)
