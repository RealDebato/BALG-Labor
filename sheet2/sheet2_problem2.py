import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('sheet2\Test_Images\Retina1.jpg', 0)

cv2.imshow('Orginal', img)

r = 1
kernel = np.ones((r, r), np.uint8)

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

Erosion = gray_erosion(img, r).astype(np.uint8)

#Erosion_plot = plt.imshow(Erosion)
cv2.imshow('Erosion', Erosion)


print('Ende')
cv2.waitKey(0)
cv2.destroyAllWindows() 