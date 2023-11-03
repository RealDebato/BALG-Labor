
import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt



Kermit = np.array([[5, 10, 50, 20, 3, 9],
                  [15, 155, 180, 155, 12, 21],
                  [100, 200, 255, 200, 100, 84],
                  [35, 155, 180, 155, 16, 54],
                  [55, 23, 50, 30, 7, 67]])


r = 1

pad_Kermit = np.pad(array=Kermit, pad_width=r, mode='constant', constant_values=0)

print(pad_Kermit.shape[0])
print(pad_Kermit.shape[1])

R_hight = np.arange(0, pad_Kermit.shape[0])
R_width = np.arange(0, pad_Kermit.shape[1])

X, Y = np.meshgrid(R_width, R_hight)
print('--------------------------------')
print('X =')
print(X)
print('--------------------------------')
print('Y =')
print(Y)

print(X*X)
# Meshgrid in kreisförmigen Kern wandeln
kernel = np.sqrt(X*X + Y*Y)
print('--------------------------------')
print('kernel =')
print(kernel)

def px_kernel_circle(img, x, y, r=3):
    img = np.array(img)
    R_hight = np.arange(0, img.shape[0])
    R_width = np.arange(0, img.shape[1])
    X, Y = np.meshgrid(R_width, R_hight)
    px_kernel = img[ np.sqrt((X - x)**2 + (Y - y)**2) <= r ]
    return px_kernel

print('Test')
print(px_kernel_circle(Kermit, 1, 1, 1))
# Leeres Bild erstellen mit shape of Kermit

Zeros = np.zeros(pad_Kermit.shape[0], pad_Kermit.shape[1])

for x in range(r, pad_Kermit.shape[0] - r):
    for y in range(r, pad_Kermit.shape[1] - r):
        print('gw[x, y] =', pad_Kermit[x , y], '[', x, ',', y, ']')
        # kernel mit Bild überlagern


        