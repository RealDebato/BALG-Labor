import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import time

#---------------------------------------------------------------------------------------------------
# functions

def recon_opening(img, kernel, start_iter):                 # Recon opening wie in 2c, Seed ist eine Erosion (Gleiches gilt für closing vice versa)
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
# Zeitmessung jeweils ohne die Seed-Erstellung
# Zeitmessungen für smoothing mit opening closing by reconstruction vice versa (2x)
# Zeitmessung closing by reconstruction and opening by reconstruction

t0 = time.time()

recon_by_closing, time_recon_closing = recon_closing(img, kernel, 7)
smoothing_by_reconstrucion_cl_op, _ = recon_opening(recon_by_closing, kernel, 7)
t1 = time.time()
time_smooth_cl_op = t1 - t0


t0 = time.time()
recon_by_opening, time_recon_opening = recon_opening(inv_img, kernel, 7)
smoothing_by_reconstrucion_op_cl, _ = recon_closing(recon_by_opening, kernel, 7)
t1 = time.time()
time_smooth_op_cl = t1 - t0

# smothing mit morphological opening and closing vice versa (2x)

t0 = time.time()
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
op_cl = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
t1 = time.time()
time_op_cl = t1 - t0
print(time_op_cl)
print(t0)
print(t1)


t0 = time.time()
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
cl_op = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=3)
t1 = time.time()
time_cl_op = t1 - t0
print(time_cl_op)
print(t0)
print(t1)

#---------------------------------------------------------------------------------------------------
# Output
cv2.imshow('Recon opening', recon_by_opening)
cv2.imshow('Recon closing', recon_by_closing)
cv2.imshow('smooth op-cl', smoothing_by_reconstrucion_op_cl)
cv2.imshow('smooth cl-op', smoothing_by_reconstrucion_cl_op)
cv2.imshow('Morph-smooth op->cl', op_cl)
cv2.imshow('Morph-smooth cl->op', cl_op)


x_axis = ['Recon closing', 'Recon opening', 'Recon-smoothing cl->op', 'Recon-smoothing op->cl', 'Morph-smooth cl->op', 'Morph-smooth op->cl']
y_axis = [time_recon_closing, time_recon_opening, time_smooth_cl_op, time_smooth_op_cl, time_cl_op, time_op_cl]
plt.stem(x_axis, y_axis, 'r')
plt.show()

#---------------------------------------------------------------------------------------------------
# main-end

print('La fin')
cv2.waitKey(0)