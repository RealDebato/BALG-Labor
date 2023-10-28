# Sheet 1, Problem 3
# Compute the local entropy for a circle of radius r. Filter welcher die Entropie eines Bildes bestimmt


import numpy as np
import math
import scipy as sp

# Bild erstellen von welchem die lokale Entropie ermittelt wird

img1 = np.random.randint(255, size=(100, 100))
img2 = np.eye(100) * 255
print(img1)
print(img1[0, 1])
print(img1.shape[1])

def Entropy(x):
    if 0 < x < 1:  
        return x * math.log(1/x, 2)
    else:
        return 0
    

def entropy_filter(img, r):
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

    return img
    
    

print(entropy_filter(img1, 2))
                



