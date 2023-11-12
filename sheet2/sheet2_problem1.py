import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('sheet2\Test_Images\Retina1.jpg')

cv2.imshow('Orginal', img)


r = 3



#lin_laplace = img*0
#morph_laplace = img*0
kernel = np.ones((r, r), np.uint8)

# Ohne Smoothing
#--------------------------------------------------------------------------------------------------
# Linear Laplacian
lin_laplace = cv2.Laplacian(img, ddepth=0, ksize=r)
S_lin = img - (0.3 * lin_laplace).astype(np.uint8)

# Morphological Laplacian --->  0.5 * (Dilatation + Erosion) - img
morph_laplace = 0.5 * (cv2.dilate(img, kernel, iterations=1) + cv2.erode(img, kernel, iterations=1))
S_morph = img - (0.3 * morph_laplace).astype(np.uint8)

# Tophat
WTH = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
BTH = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
S_tophat = img - (1 * WTH).astype(np.uint8) + (1 * BTH).astype(np.uint8)


cv2.imshow('Lin Laplace', S_lin)
cv2.imshow('Morph Laplace', S_morph)
cv2.imshow('Tophat', S_tophat)


# Mit Smoothing
#--------------------------------------------------------------------------------------------------
smooth_gaussian = cv2.GaussianBlur(img, (r, r), 0)
cv2.imshow('Gaussian', smooth_gaussian)

# Linear Laplacian
lin_laplace_S = cv2.Laplacian(smooth_gaussian, ddepth=0, ksize=r)
S_lin_smooth = smooth_gaussian - (0.3 * lin_laplace_S).astype(np.uint8)

# Morphological Laplacian --->  0.5 * (Dilatation + Erosion) - img
morph_laplace_S = 0.5 * (cv2.dilate(smooth_gaussian, kernel, iterations=1) + cv2.erode(smooth_gaussian, kernel, iterations=1))
S_morph_smooth = img - (0.3 * morph_laplace_S).astype(np.uint8)

# Tophat
WTH_S = cv2.morphologyEx(smooth_gaussian, cv2.MORPH_TOPHAT, kernel)
BTH_S = cv2.morphologyEx(smooth_gaussian, cv2.MORPH_BLACKHAT, kernel)
S_tophat_smooth = smooth_gaussian - (1 * WTH).astype(np.uint8) + (1 * BTH).astype(np.uint8)

cv2.imshow('Lin Laplace smooth', S_lin_smooth)
cv2.imshow('Morph Laplace smooth', S_morph_smooth)
cv2.imshow('Tophat smooth', S_tophat_smooth)


#cv2.imshow('Diff', img-S_tophat)

#print('Laufzeit', laplace_img)
#print('Laufzeit =', t1-t0 )
print('Ende')
cv2.waitKey(0)
cv2.destroyAllWindows() 


# Bild und Methode müssen zusammenpassen
# lowContrast.jpg ist der Tophatfilter am geeignetsten
# city.jpg ist der Lineare Laplace am geeignetsten
# retina1.jpg ist der morphologische Laplace am geeignetsten
# c ist Bildabhängig