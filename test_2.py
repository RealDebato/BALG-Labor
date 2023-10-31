import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt


Kermit = cv2.imread('_Smiley.png', 0)
cv2.imshow('Orginal Kermit', Kermit)
#Kermit = np.ones((100, 100)).astype('uint8')

def hist_n(img):
    histogram = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            GV = img[i, j]
            histogram[GV] = histogram[GV] + 1
    histogram_normal = histogram/histogram.sum()
    return histogram_normal

Histn_Kermit = hist_n(Kermit)
max_px = max(Histn_Kermit)

def Entropy(p):                                         # Argument sind die Wahrscheinlichkeiten als Vektor 
       E = -(p[p > 0] * np.log2(p[p > 0])).sum()
       return E

print(Entropy(Histn_Kermit))

def entropy_filter_slow(img, r):                 # img[col, row]
       Histn_img = hist_n(img)
       p = []
       core = 0
       # Randberech erweitern
       img = np.pad(array=img, pad_width=r, mode='constant', constant_values=0)
    
       for i in range(r, img.shape[0] - r):
              for j in range(r, img.shape[1] - r):
                     for k in range(-r, r):
                            for l in range(-r, r):
                                   gw = img[i + k, j + l]
                                   p = np.append(p, Histn_img[gw])
                     core = core + Entropy(p)
                     img[i, j] = core

       return img

Kermit_Entropy_filtered = entropy_filter_slow(Kermit, 1)
cv2.imshow('Entropie-Bild', Kermit_Entropy_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows() 