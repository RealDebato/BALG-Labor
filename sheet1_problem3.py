# Sheet 1, Problem 3
# Compute the local entropy for a circle of radius r. Filter welcher die Entropie eines Bildes bestimmt


import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt

# Bild erstellen von welchem die lokale Entropie ermittelt wird

#img1 = np.random.randint(255, size=(100, 100))
Eye = np.eye(500) * 255
Kermit = cv2.imread('_Kermit.png', 0)
#Smiley = cv2.imread('_Smiley.png', 0)
#cv2.imshow('Orginal Eye', Eye)
cv2.imshow('Orginal Kermit', Kermit)
#cv2.imshow('Orginal Smiley', Smiley)

def hist(img):
    histogram = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            GV = img[i, j]
            histogram[GV] = histogram[GV] + 1
    return histogram

Hist_Kermit = hist(Kermit)
pixelanzahl = Kermit.shape[0] * Kermit.shape[1]
max_px = max(Hist_Kermit)

def Entropy(x):
    if 0 < x/255 < 1:  
        return x * math.log(1/x, 2)
    else:
        return 0


def entropy_filter_slow(img, r):                 # img[col, row]
    core = 0
    # Randberech erweitern
    img = np.pad(array=img, pad_width=r, mode='constant', constant_values=0)
    #cv2.imshow('Mit Rand', img)
        


    for i in range(r, img.shape[0] - r):
        for j in range(r, img.shape[1] - r):
            for k in range(-r, r):
                for l in range(-r, r):
                    p = img[i + k, j + l]
                    core = core + Entropy(p)
            img[i, j] = core

    return img

#entropy_eye = entropy_filter_slow(Eye, 1)
entropy_kermit = entropy_filter_slow(Kermit, 1)
#entropy_smiley = entropy_filter_slow(Smiley, 1)
#cv2.imshow('Entropy Eye', entropy_eye)                    
cv2.imshow('Entropy Kermit', entropy_kermit)
#cv2.imshow('Entropy Smiley', entropy_smiley)

Hist_Kermit_entropy = hist(entropy_kermit)

plt.style.use('_mpl-gallery')

fig, ax = plt.subplots()

ax.stairs(Hist_Kermit, linewidth=2.5)

ax.set(xlim=(0, 256), xticks=np.arange(1, 256),
       ylim=(0, max_px), yticks=np.arange(0, max_px, 0.1))

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows() 

'''def entropy_filter(img, r):
    i = 0
    j = 0
    u = 0
    entropy_core = 0
    entropy_col_1 = 0
    entropy_col_2 = 0
    entropy_col_3 = 0
    entropy_col_4 = 0
    for u in range(0, 2 * r - 1):               # Bedingung alle Filterkerne sind symmetrisch, d. h. r ist ungerade aus N
        for v in range(0, 2 * r - 1):           # Werden gerade r zugelassen muss muss zuvor abgefragt werden ob r gerade/ungerade
            entropy_core = entropy_core + Entropy(img[u, v] / 255)
            if u == 0:
                entropy_col_1 = entropy_col_1 + Entropy(img[u, v] / 255)
            elif u == 1:
                entropy_col_2 = entropy_col_2 + Entropy(img[u, v] / 255)
            elif u == 2:
                entropy_col_3 = entropy_col_3 + Entropy(img[u, v] / 255)

            
        
    for i in range(1 + r, img.shape[0] - r):            # Img[col, row]
        for j in range(1 + r , img.shape[1] - r):       
            for k in range(-r, r):
                entropy_col_1 = entropy_col_1 + Entropy(img[i + k, j - r - 1])
                entropy_col_4 = entropy_col_4 + Entropy(img[i + k, j + r])

            entropy_core = entropy_core - entropy_col_1 + entropy_col_4
            img[i, j] = entropy_core

    return img'''
    
   


                



