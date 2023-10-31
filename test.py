
import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt



Kermit = cv2.imread('_Kermit.png', 0)

def hist(img):
    histogram = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            GV = img[i, j]
            histogram[GV] = histogram[GV] + 1
    return histogram

pixelanzahl = Kermit.shape[0] * Kermit.shape[1]
print(pixelanzahl)
Hist_Kermit = hist(Kermit)
Hist_Kermit = Hist_Kermit / pixelanzahl
max_px = max(Hist_Kermit)
print(Hist_Kermit)

plt.style.use('_mpl-gallery')

fig, ax = plt.subplots()

ax.stairs(Hist_Kermit, linewidth=2.5)

ax.set(xlim=(0, 256), xticks=np.arange(1, 256),
       ylim=(0, max_px), yticks=np.arange(0, max_px, 0.1))

plt.show()



