
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

# LUT erstellen



hist, _ = np.histogram(Kermit, 256, [0,256], False)
maxAnzahl_px = np.sum(hist)
print(maxAnzahl_px)
print(hist)
unique, counts = np.unique(Kermit, return_counts=True)
print(unique, counts)

LUT = dict(zip(unique, counts/maxAnzahl_px))
#LUT = np.array([unique, counts/maxAnzahl_px])
print('--------------------------------------------')
print('LUT')
print(LUT)
P = []
for i in [5, 10, 50, 10]:
    P = np.append(P, LUT[i])
print('P')
print(P)



'''maxPossibleElements = (2*r)**2
print('max =', maxPossibleElements)
pldp = np.zeros([maxPossibleElements+1,maxPossibleElements+1])
print('--------------------------------------------')
print(pldp)
pldp[1,1] = 0
for i in range(1,maxPossibleElements+1):
    for j in range(1,i):
        relativeFrequency = j/i
        print('j/i=', j,'/', i,'=',j/i)
        pldp[j,i] = relativeFrequency * np.log2(relativeFrequency)
print('--------------------------------------------')
print(pldp)'''
