import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import time
from sheet2_problem2 import recon_by_dilation_grey

#---------------------------------------------------------------------------------------------------
# Images

img100 = cv2.imread('sheet2\image sizes\electrop100.jpg')
img120 = cv2.imread('sheet2\image sizes\electrop120.jpg')
img150 = cv2.imread('sheet2\image sizes\electrop150.jpg')
img200 = cv2.imread('sheet2\image sizes\electrop200.jpg')
img500 = cv2.imread('sheet2\image sizes\electrop500.jpg')

kernel = np.ones(3, 3).astype(np.uint8)

#---------------------------------------------------------------------------------------------------
# times

time100a = []
time120a = []
time150a = []
time200a = []
time500a = []

time100b = []
time120b = []
time150b = []
time200b = []
time500b = []

#---------------------------------------------------------------------------------------------------
# Messung 100
# selfmade
for i in range(0, 50):
    img_grey_recon_selfmade, time_grey_selfmade = recon_by_dilation_grey(img100, kernel, 7)
    time100a = np.append(time100a, time_grey_selfmade)

# scikit
for i in range(0, 50):
    seed_g = ski.morphology.erosion(img100, ski.morphology.square(15)).astype(np.double)
    footprint = img100.astype(np.double)

    t0_scikit_grey = time.time()
    scikit_reconstruction_g = ski.morphology.reconstruction(seed_g, footprint, 'dilation').astype(np.double)
    t1_scikit_grey = time.time()

    time100b = np.append(time100b, t1_scikit_grey - t0_scikit_grey)

#---------------------------------------------------------------------------------------------------
# Messung 120
# selfmade
for i in range(0, 50):
    img_grey_recon_selfmade, time_grey_selfmade = recon_by_dilation_grey(img120, kernel, 7)
    time120a = np.append(time120a, time_grey_selfmade)

# scikit
for i in range(0, 50):
    seed_g = ski.morphology.erosion(img120, ski.morphology.square(15)).astype(np.double)
    footprint = img120.astype(np.double)

    t0_scikit_grey = time.time()
    scikit_reconstruction_g = ski.morphology.reconstruction(seed_g, footprint, 'dilation').astype(np.double)
    t1_scikit_grey = time.time()

    time120b = np.append(time120b, t1_scikit_grey - t0_scikit_grey)

#---------------------------------------------------------------------------------------------------
# Messung 150
# selfmade
for i in range(0, 50):
    img_grey_recon_selfmade, time_grey_selfmade = recon_by_dilation_grey(img150, kernel, 7)
    time150a = np.append(time150a, time_grey_selfmade)

# scikit
for i in range(0, 50):
    seed_g = ski.morphology.erosion(img150, ski.morphology.square(15)).astype(np.double)
    footprint = img150.astype(np.double)

    t0_scikit_grey = time.time()
    scikit_reconstruction_g = ski.morphology.reconstruction(seed_g, footprint, 'dilation').astype(np.double)
    t1_scikit_grey = time.time()

    time150b = np.append(time150b, t1_scikit_grey - t0_scikit_grey)

#---------------------------------------------------------------------------------------------------
# Messung 200
# selfmade
for i in range(0, 50):
    img_grey_recon_selfmade, time_grey_selfmade = recon_by_dilation_grey(img200, kernel, 7)
    time200a = np.append(time200a, time_grey_selfmade)

# scikit
for i in range(0, 50):
    seed_g = ski.morphology.erosion(img200, ski.morphology.square(15)).astype(np.double)
    footprint = img200.astype(np.double)

    t0_scikit_grey = time.time()
    scikit_reconstruction_g = ski.morphology.reconstruction(seed_g, footprint, 'dilation').astype(np.double)
    t1_scikit_grey = time.time()

    time200b = np.append(time200b, t1_scikit_grey - t0_scikit_grey)


#---------------------------------------------------------------------------------------------------
# Messung 500
# selfmade
for i in range(0, 50):
    img_grey_recon_selfmade, time_grey_selfmade = recon_by_dilation_grey(img500, kernel, 7)
    time500a = np.append(time500a, time_grey_selfmade)

# scikit
for i in range(0, 50):
    seed_g = ski.morphology.erosion(img500, ski.morphology.square(15)).astype(np.double)
    footprint = img500.astype(np.double)

    t0_scikit_grey = time.time()
    scikit_reconstruction_g = ski.morphology.reconstruction(seed_g, footprint, 'dilation').astype(np.double)
    t1_scikit_grey = time.time()

    time500b = np.append(time500b, t1_scikit_grey - t0_scikit_grey)


# selfmade
meanTime100a = np.mean(time100a)
meanTime120a = np.mean(time120a)
meanTime150a = np.mean(time150a)
meanTime200a = np.mean(time200a)
meanTime500a = np.mean(time500a)

Time_selfmade = [meanTime100a, meanTime120a, meanTime150a, meanTime200a, meanTime500a]

# scikit
meanTime100b = np.mean(time100b)
meanTime120b = np.mean(time120b)
meanTime150b = np.mean(time150b)
meanTime200b = np.mean(time200b)
meanTime500b = np.mean(time500b)

Time_scikit = [meanTime100b, meanTime120b, meanTime150b, meanTime200b, meanTime500b]


x_Axis = np.arange(0, 5)

plt.plot(Time_selfmade, x_Axis, 'r', Time_scikit, x_Axis, 'b')
plt.show()

