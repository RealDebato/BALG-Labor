import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt


Kermit = cv2.imread('_Smiley.png', 0)
cv2.imshow('Orginal Kermit', Kermit)
#Kermit = np.ones((100, 100)).astype('uint8')

'''Kermit = np.array([[0, 0, 155, 0, 0],
                  [0, 155, 200, 155, 0],
                  [155, 200, 255, 200, 155],
                  [0, 155, 200, 155, 0],
                  [0, 0, 155, 0, 0]])'''



def hist_n(img):
    histogram = np.zeros(256)
    
    if img.ndim == 2:
       for i in range(0, img.shape[0]):
              for j in range(0, img.shape[1]):
                     GV = img[i, j]
                     GV = int(GV)                     
                     histogram[GV] = histogram[GV] + 1                     
       histogram_normal = histogram/histogram.sum()
       return histogram_normal
    elif img.ndim == 1:
       for i in range(0, img.shape[0]):
              GV = img[i]
              GV = int(GV)                       # Warum hier nötig aber nicht bei if img.ndim == 2 ???
              histogram[GV] = histogram[GV] + 1              
       histogram_normal = histogram/histogram.sum()
       return histogram_normal

Histn_Kermit = hist_n(Kermit)
max_hist = max(Histn_Kermit)

def Entropy(p):                                         # Argument sind die Wahrscheinlichkeiten als Vektor 
       E = -(p[p > 0] * np.log2(p[p > 0])).sum()
       return E

print(Entropy(Histn_Kermit))

#filter_kern_q = np.array([1, 1, 1],
#                         [1, 1, 1],
#                         [1, 1, 1])

def entropy_filter_slow(img, r):                 # img[col, row]
       p = []
       core = 0
       # Randberech erweitern
       img = np.pad(array=img, pad_width=r, mode='constant', constant_values=0)
       #print('Img Pad=', img)
    
       for i in range(r, img.shape[0] - r):
              #print('i =', i)
              for j in range(r, img.shape[1] - r):
                     #print('j =', j)
                     for k in range(-r, r + 1):
                            #print('k =', k)
                            for n in range(-r, r + 1):
                                   #print('n =', n)
                                                                  # 1. Filterposition pro Col muss komplett berechnet werden
                                   gw = img[i + k, j + n]
                                   #print('gw(i + k, j + n) =', gw)
                                   gw = int(gw)
                                   p = np.append(p, gw)
                                   #print('p =', p)

                                   
                     histn_p = hist_n(p)
                     core = core + Entropy(histn_p)
                     img[i, j] = core

       return img

Kermit_Entropy_filtered = entropy_filter_slow(Kermit, 1)
print('Filtered Kermit=', Kermit_Entropy_filtered)

cv2.imshow('Entropie-Bild', Kermit_Entropy_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows() 