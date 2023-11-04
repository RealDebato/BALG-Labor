import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt


Kermit = cv2.imread('_Smiley.png', 0)
cv2.imshow('Orginal Kermit', Kermit)
#Kermit = np.ones((100, 100)).astype('uint8')

#Kermit = np.array([[5, 10, 50, 20, 3, 9],
#                  [15, 155, 180, 155, 12, 21],
#                  [100, 200, 255, 200, 100, 84],
#                  [35, 155, 180, 155, 16, 54],
#                  [55, 23, 50, 30, 7, 67]])

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

def Entropy(p):                                         # Argument sind die Wahrscheinlichkeiten als Vektor 
       E = -np.sum(p[p > 0] * np.log2(p[p > 0]))
       return E

def pad_delete(padded_img, pad_width):                  # Entfernt den Rahmen des Arrays mit Dicke pad_width (r)
    padded_img = np.delete(padded_img, np.arange(0, pad_width), axis=0)
    padded_img = np.delete(padded_img, np.arange(padded_img.shape[0] - pad_width, padded_img.shape[0]), axis=0)
    padded_img = np.delete(padded_img, np.arange(0, pad_width), axis=1)
    padded_img = np.delete(padded_img, np.arange(padded_img.shape[1] - pad_width, padded_img.shape[1]), axis=1)
    return padded_img

def px_kernel_circle(img, x, y, r=3):                   # Gibt die px-werte im kernel an der Pos. x, y aus (px-Werte sind ungeordnet)
       img = np.array(img)
       R_width = np.linspace(0, img.shape[1]-1, img.shape[1])
       R_hight = np.linspace(0, img.shape[0]-1, img.shape[0])
       X, Y = np.meshgrid(R_width, R_hight)
       px_kernel = img[np.sqrt((X - x)**2 + (Y - y)**2) <= r]
       return px_kernel

def entropy_filter_slow(img, r=3):
       img = np.pad(array=Kermit, pad_width=r, mode='edge')
       filtered_img = img
       for y in range(img.shape[0]):
          for x in range(img.shape[1]):
               filtered_img[y, x] = Entropy(hist_n(px_kernel_circle(img, x, y, r))) * 32   # * 32 als Angleichung an uint8 
       filtered_img = pad_delete(filtered_img, r)
       return filtered_img
               
def entropy_filter_faster(img, r=3):
       
          

Kermit_Entropy_filtered = entropy_filter_slow(Kermit, 1)
print('Filtered Kermit, slow=\n', Kermit_Entropy_filtered)

cv2.imshow('Entropie-Bild', Kermit_Entropy_filtered)
print('Ende')
cv2.waitKey(0)
cv2.destroyAllWindows() 