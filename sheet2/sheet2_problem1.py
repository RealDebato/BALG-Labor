import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('sheet2\Test_Images\city.jpg', 0)
cv2.imshow('Orginal', img)

img_linear =

img_morph =

img_tophat =





#cv2.imshow('Entropie-Bild', Kermit_Entropy_filtered)
#print('Laufzeit', Kermit_Entropy_filtered)
#print('Laufzeit =', t1-t0 )
print('Ende')
cv2.waitKey(0)
cv2.destroyAllWindows() 