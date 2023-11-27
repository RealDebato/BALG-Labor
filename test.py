import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


training_batch = [[2, 5, 8],[4, 7, 1],[2, 2, 0],[9, 8, 3],[0, 1, 4]]
test_batch = [[4, 1, 7],[2, 1, 1],[8, 7, 5]]
training_batch = np.asarray(training_batch)
test_batch = np.asarray(test_batch)


dists = np.sum(np.square(training_batch), axis=1) + np.sum(np.square(test_batch), axis=1)[:, np.newaxis] - 2 * np.dot(test_batch, training_batch.T)
print(np.sum(np.square(training_batch), axis=1))
print(np.sum(np.square(test_batch), axis=1)[:, np.newaxis])
print(np.sum(np.square(training_batch), axis=1) + np.sum(np.square(test_batch), axis=1)[:, np.newaxis])
print(2 * np.dot(test_batch, training_batch.T))
print(dists)



