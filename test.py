import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time




A = np.array([[ 7367819. , 8234613.,  8265938.],
 [ 9296495. , 9275766. , 9434138.],
 [ 6132123. , 6213317. , 6264519.],
 [ 5620181.,  6887595.,  7043724.],
 [ 6209436. , 6224860. , 6289059.],
 [ 6915525. , 6619657. , 7187072.],
 [ 6913128.  ,      0. , 6927035.],
 [ 7330454.,  6674827.,  6604806.],
 [ 8718731. , 8462096.,  8392504.],
 [10346119. ,10563440., 11016134.]])

print(A)


k_min = np.partition(np.ndarray.flatten(A), 3)[:3]
print(k_min)
index_k_min_old = np.zeros_like(A, dtype=np.bool_)

for min in k_min:
    index_k_min = A == min
    index_k_min = np.logical_or(index_k_min, index_k_min_old)
    index_k_min_old = index_k_min

nearest_classes = np.argwhere(index_k_min==True)[:,0]

sol_class = np.argmax(np.bincount(nearest_classes))

print('nearest classes', nearest_classes)
print('solution', sol_class)