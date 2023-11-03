import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt


#Kermit = cv2.imread('_Smiley.png', 0)
#cv2.imshow('Orginal Kermit', Kermit)
#Kermit = np.ones((100, 100)).astype('uint8')

Kermit = np.array([[5, 10, 50, 20, 3, 9],
                  [15, 155, 180, 155, 12, 21],
                  [100, 200, 255, 200, 100, 84],
                  [35, 155, 180, 155, 16, 54],
                  [55, 23, 50, 30, 7, 67]])

print('Orginal Kermit\n',Kermit)

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

def entropy_local(img, x, y, r):
       R = np.arange(-r, r + 1)
       X, Y = np.meshgrid(R, R)

       # Meshgrid in kreisförmigen Kern wandeln
       kernel = img[ np.sqrt((X - x)**2 + (Y - y)**2) <= r ]
   


       
Kermit_Entropy_filtered = entropy_filter(Kermit, 1)
print('Filtered Kermit=\n', Kermit_Entropy_filtered)

#cv2.imshow('Entropie-Bild', Kermit_Entropy_filtered)
print('Ende')
#cv2.waitKey(0)
#cv2.destroyAllWindows() 