import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


#---------------------------------------------------------------------------------------------------
# functions

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def image_from_cifar10(data_vektor):
   
    img_flattend = np.split(data_vektor, 3)
    red = img_flattend[0]
    green = img_flattend[1] 
    blue = img_flattend[2]

    red = np.split(red, 32)
    green = np.split(green, 32)
    blue = np.split(blue, 32)
    
    img = np.dstack((red, green, blue))

    return img.astype(np.uint8)

#---------------------------------------------------------------------------------------------------
# globals

nr_bild = 29


#---------------------------------------------------------------------------------------------------
# main

data_reduction = 1000       # max. 10.000

label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
print(label_decoder[b'label_names'])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')

pixel_data_batch_1 = np.asarray(data_batch_1[b'data'])
pixel_data_batch_1 = pixel_data_batch_1[0:data_reduction, :]

labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])
labels_data_batch_1 = labels_data_batch_1[0:data_reduction]

# Einteilung in 4 Quartiele für 4-fold cross-validation
split = 4
split_col = pixel_data_batch_1.shape[0]/split
split_col = int(split_col)

# Testdaten
pixel_data_batch_1_0_25 = pixel_data_batch_1[0:split_col, :]
pixel_data_batch_1_25_50 = pixel_data_batch_1[split_col:2*split_col, :]
pixel_data_batch_1_50_75 = pixel_data_batch_1[2*split_col:3*split_col, :]
pixel_data_batch_1_75_100 = pixel_data_batch_1[3*split_col:4*split_col, :]

# Trainingsdaten
pixel_data_batch_1_0_75 = pixel_data_batch_1[0:3*split_col, :]
pixel_data_batch_1_25_100 = pixel_data_batch_1[split_col:4*split_col, :]
pixel_data_batch_1_without_25_50 = np.delete(pixel_data_batch_1, np.s_[split_col:2*split_col], 1)
pixel_data_batch_1_without_50_75 = np.delete(pixel_data_batch_1, np.s_[split_col:2*split_col], 1)

Bild1 = image_from_cifar10(pixel_data_batch_1[nr_bild,:])


# Wie sehen die Entscheidungsräume aus??
# Features sollen die Pixel sein, d. h. eine Dim ist ein Pixel [0:255]
# Anzahl feature spaces entspricht Anzahl Labels = 0:9
# Differenzen zu jedem feature space / label berechenen

# Zum Trainig werden nur 75% aller Bilder aus Batch 1 verwendet. Die restlichen 25% werden zum Testing verwendet 

# feature space 0 --> airplane
class_0 = np.where(labels_data_batch_1==0)
pixel_data_class_0 = np.asarray(pixel_data_batch_1[class_0, :][0])

# feature space 1 --> automobile
class_1 = np.where(labels_data_batch_1==1)
pixel_data_class_1 = np.asarray(pixel_data_batch_1[class_1, :][0])

# feature space 2 --> bird
class_2 = np.where(labels_data_batch_1==2)
pixel_data_class_2 = np.asarray(pixel_data_batch_1[class_2, :][0])

# feature space 3 --> cat
class_3 = np.where(labels_data_batch_1==3)
pixel_data_class_3 = np.asarray(pixel_data_batch_1[class_3, :][0])

# feature space 4 --> deer
class_4 = np.where(labels_data_batch_1==4)
pixel_data_class_4 = np.asarray(pixel_data_batch_1[class_4, :][0])

# feature space 5 --> dog
class_5 = np.where(labels_data_batch_1==5)
pixel_data_class_5 = np.asarray(pixel_data_batch_1[class_5, :][0])

# feature space 6 --> frog
class_6 = np.where(labels_data_batch_1==6)
pixel_data_class_6 = np.asarray(pixel_data_batch_1[class_6, :][0])

# feature space 7 --> horse
class_7 = np.where(labels_data_batch_1==7)
pixel_data_class_7 = np.asarray(pixel_data_batch_1[class_7, :][0])

# feature space 8 --> ship
class_8 = np.where(labels_data_batch_1==8)
pixel_data_class_8 = np.asarray(pixel_data_batch_1[class_8, :][0])

# feature space 9 --> truck
class_9 = np.where(labels_data_batch_1==9)
pixel_data_class_9 = np.asarray(pixel_data_batch_1[class_9, :][0])

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Distanzberechnung für nur ein Testbild für jede Klasse einzeln
# Delta zu Class 0
#def delta_class(training, test):



k = 10

delta = []
delta_sum = []


# Differenz von Test Img zu Class X-------------------------
def distance_kNN(test, training_class, k):
    training_class = np.asarray(training_class)
    training_class = training_class.astype(np.double)
    test = np.asarray(test)
    test = test.astype(np.double)
    delta = []
    delta_sum = []
    for number_trainings_img in range(0, training_class.shape[0]):
        # new row for new image to compare 
        diff = (test - training_class[number_trainings_img]) **2
        delta = np.append(delta, diff)

    delta = np.split(delta, training_class.shape[0])

    for number_trainings_img in range(0, training_class.shape[0]):        # delta_sum[0] entspricht der Differenz zwischen dem Test bild und dem 1. Trainingsbild
        delta_sum_single = delta[number_trainings_img].sum()
        delta_sum = np.append(delta_sum, delta_sum_single)
    
    index_k_nearest = np.argpartition(delta_sum, k)
    k_nearest_class_0 = delta_sum[index_k_nearest[:k]]

    '''sorted_delta_sum = np.sort(delta_sum, axis=None, kind='heapsort')
    k_nearest_class_0 = sorted_delta_sum[0:k]'''
    return k_nearest_class_0



nearest_class_0 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_0, 3)
nearest_class_1 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_1, 3)
nearest_class_2 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_2, 3)
nearest_class_3 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_3, 3)
nearest_class_4 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_4, 3)
nearest_class_5 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_5, 3)
nearest_class_6 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_6, 3)
nearest_class_7 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_7, 3)
nearest_class_8 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_8, 3)
nearest_class_9 = distance_kNN(pixel_data_batch_1[0], pixel_data_class_9, 3)

stack_nearest_classes = np.stack((nearest_class_0, nearest_class_1, nearest_class_2, nearest_class_3, nearest_class_4, nearest_class_5, nearest_class_6, nearest_class_7, nearest_class_8, nearest_class_9))

k_min = np.partition(np.ndarray.flatten(stack_nearest_classes), k)[:k]
print(k_min)
index_k_min_old = np.zeros_like(stack_nearest_classes, dtype=np.bool_)

for min in k_min:
    index_k_min = stack_nearest_classes == min
    index_k_min = np.logical_or(index_k_min, index_k_min_old)
    index_k_min_old = index_k_min

nearest_classes = np.argwhere(index_k_min==True)[:,0]
print('Nearest Classes', nearest_classes)

solution_class = np.argmax(np.bincount(nearest_classes))    # Was wenn es zwei max. Bins gibt? Idee: Die Ausgabe erfolgt als Wahrscheinlichkeit für jede Klasse
print('Solution', solution_class)

solution_correct = solution_class == labels_data_batch_1[0]

print(solution_correct)


#---------------------------------------------------------------------------------------------------
# output


'''plt.figure
plt.imshow(Bild1)
plt.title(label_decoder[b'label_names'][labels_data_batch_1[nr_bild]])'''



#---------------------------------------------------------------------------------------------------
# end

plt.show()
print('La fin')
cv2.waitKey(0)
