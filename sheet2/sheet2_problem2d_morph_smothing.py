import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import time

#---------------------------------------------------------------------------------------------------
# functions

def recon_opening(img, kernel, start_iter):
    img = img.astype(np.double)
    
    start = cv2.erode(img, kernel, iterations=start_iter)
    #cv2.imshow('Start2', start)

    # Reconstruction by dilation
    t0_selfmade = time.time()
    Recon = np.minimum(img, cv2.dilate(start, kernel, iterations=1)).astype(np.double)
    Recon_old = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.double)

    i = 0

    while np.array_equal(Recon, Recon_old) == False:
        Recon = np.minimum(img, cv2.dilate(Recon_old, kernel, iterations=1)).astype(np.double)
        Recon_old = np.minimum(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.double)

        i = i + 1
        if i == 100:
            break
    t1_selfmade = time.time()
    return Recon.astype(np.uint8), t1_selfmade - t0_selfmade    

def recon_closing(img, kernel, start_iter):
    img = img.astype(np.double)
    
    start = cv2.dilate(img, kernel, iterations=start_iter)
    #cv2.imshow('Start2', start)

    # Reconstruction by dilation
    t0_selfmade = time.time()
    Recon = np.maximum(img, cv2.erode(start, kernel, iterations=1)).astype(np.double)
    Recon_old = np.maximum(img, cv2.erode(Recon, kernel, iterations=1)).astype(np.double)

    i = 0

    while np.array_equal(Recon, Recon_old) == False:
        Recon = np.maximum(img, cv2.erode(Recon_old, kernel, iterations=1)).astype(np.double)
        Recon_old = np.maximum(img, cv2.erode(Recon, kernel, iterations=1)).astype(np.double)

        i = i + 1
        if i == 100:
            break
    t1_selfmade = time.time()
    return Recon.astype(np.uint8), t1_selfmade - t0_selfmade    

#---------------------------------------------------------------------------------------------------
# globals

kernel = np.ones((3, 3), np.uint8)

#---------------------------------------------------------------------------------------------------
# images

img = cv2.imread('sheet2\Test_Images\electrop.jpg')
inv_img = np.abs(255 - img)
cv2.imshow('Orginal', img)
cv2.imshow('Inv Orginal', inv_img)

#---------------------------------------------------------------------------------------------------
# main

t0 = time.time()

recon_by_closing, time_recon_closing = recon_closing(img, kernel, 7)
cv2.imshow('Recon closing', recon_by_closing)

smoothing_by_reconstrucion_cl_op, _ = recon_opening(recon_by_closing, kernel, 7)
cv2.imshow('smooth cl-op', smoothing_by_reconstrucion_cl_op)

t1 = time.time()

print('Time smothing by recon (closing->opening)', t1 - t0)
time_smooth_cl_op = t1-t0


t0 = time.time()

recon_by_opening, time_recon_opening = recon_opening(inv_img, kernel, 7)
cv2.imshow('Recon opening', recon_by_opening)

smoothing_by_reconstrucion_op_cl, _ = recon_closing(recon_by_opening, kernel, 7)
cv2.imshow('smooth op-cl', smoothing_by_reconstrucion_op_cl)

t1 = time.time()

print('Time smothing by recon (opening->closing)', t1 - t0)
time_smooth_op_cl = t1-t0

#---------------------------------------------------------------------------------------------------
# Output
x_axis = ['Recon closing', 'Recon opening', 'smoothing_cl_op', 'smoothing_op_cl']
y_axis = [time_recon_closing, time_recon_opening, time_smooth_cl_op, time_smooth_op_cl]
plt.stem(x_axis, y_axis, 'r')
plt.show()

#---------------------------------------------------------------------------------------------------
# main-end

print('La fin')
cv2.waitKey(0)