import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt

# Reconstruction by erosion

img = cv2.imread('sheet2\Test_Images\particle1.jpg', 0)
img = np.abs(255 - img)
plt.imshow(img)

cv2.imshow('Orginal', img)

r = 1
kernel = np.ones((3, 3), np.uint8)

def gray_erosion(img, r):
    img = np.array(img)
    img = np.pad(array=img, pad_width=r, mode='edge')
    filtered_img = img * 0.0
    R_width = np.linspace(0, img.shape[1]-1, img.shape[1])         
    R_hight = np.linspace(0, img.shape[0]-1, img.shape[0])         
    X, Y = np.meshgrid(R_width, R_hight)    
    for y in range(r, img.shape[0] - r):
        for x in range(r, img.shape[1] - r):
            pos_kernel = []
            pos_kernel = np.logical_and(np.abs((X - x)) <=r,  np.abs((Y - y)) <= r) # Hier stimmt was nicht
            print(pos_kernel)
            px_kernel = img[pos_kernel]
            print(px_kernel)
            filtered_img[y, x] = np.min(px_kernel)
            print(filtered_img[y, x])
    return filtered_img

def ero_recon(img, kernel, eps):
    
    i = 0
    
    erosion_img = cv2.erode(img, kernel, iterations=1)
    while True:
        if np.abs(np.subtract(img, erosion_img)).all() > eps:
            erosion_img = cv2.erode(img, kernel, iterations=1)
            i = i + 1
        elif i > 100:
            print('Laufzeit zu lang')
            break
        else:
            return erosion_img

#---------------------------------------------------------------------------------------------------
# Startpunkte mit Erosion bestimmen

start = cv2.erode(img, kernel, iterations=15)

cv2.imshow('Start', start)


# Reconstruction by dilation
Recon = 255 * (np.logical_and(img, cv2.dilate(start, kernel, iterations=1)).astype(np.uint8))
Recon_old = np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8)
i = 0

while Recon.all() == Recon_old.all():
    Recon = 255 * (np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8))
    Recon_old = np.logical_and(img, cv2.dilate(Recon, kernel, iterations=1)).astype(np.uint8)
    i = i + 1
    if i == 100:
        break
    

cv2.imshow('Recon', Recon)

plt.imshow(img)



#cv2.imshow('Diff', img - Recon)



#cv2.imshow('Erosion', Erosion)


print('Ende')
cv2.waitKey(0)
cv2.destroyAllWindows() 