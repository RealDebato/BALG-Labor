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

def softmax(x):
    ex = np.exp(x)
    prob = ex/np.sum(ex)
    return prob

def compute_distances(test_batch, training_batch):
    test_batch = np.asarray(test_batch)
    training_batch = np.asarray(training_batch)
    num_test = test_batch.shape[0]
    print('num test', num_test)
    num_train = training_batch.shape[0]
    print('num train', num_train)
    dists = np.zeros((num_test, num_train))
    dists = np.sum(np.square(training_batch), axis=1) + np.sum(np.square(test_batch), axis=1)[:, np.newaxis] - 2 * np.dot(test_batch, training_batch.T)
    # dist ist array mit dist zwischen (test, training)
    return dists

def predict_labels(dists, labels_train, k=1):
        dists = np.asarray(dists)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test) 
        for i in range(num_test):   # für jedes Testbild werden die k-nächsten Klassen in y_pred ausgegeben
            sorted_dist = np.argsort(dists[i])  # np.argpartition sollte schneller sein?
            print('sorted dist.shape', sorted_dist.shape)
            closest_y = list(labels_train[sorted_dist[0:k]])
            print('sorted dist 0:k',sorted_dist[0:k])
            print('Closest_y', closest_y)
            y_pred[i] = (np.argmax(np.bincount(closest_y))) #predicted class ist die häufigste der k-nächsten Klassen
            
        return y_pred

def validate_prediction(prediction, labels_test):
    accuracy = np.zeros(len(labels_test))
    for i in range(0, len(labels_test)):
        accuracy[i] = prediction[i] == labels_test[i] # Vergleich zwischen prediction und tatsächlichem label
    return accuracy


    
#---------------------------------------------------------------------------------------------------
# globals



#---------------------------------------------------------------------------------------------------
# data

data_reduction = 20       # max. 10.000
if data_reduction > 10000:
    data_reduction = 10000


label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
#print(label_decoder[b'label_names'])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')

pixel_data_batch_1 = np.asarray(data_batch_1[b'data'])
pixel_data_batch_1 = pixel_data_batch_1[0:data_reduction, :]
print('Pixel Data Batch 1 ',pixel_data_batch_1)
print('Pixel Data Batch 1 shape ',pixel_data_batch_1.shape)

labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])
labels_data_batch_1 = labels_data_batch_1[0:data_reduction]
print('Labels Data Batch 1 ',labels_data_batch_1)
print('Labels Data Batch 1 shape ',labels_data_batch_1.shape)

# Einteilung in 4 Quartiele für 4-fold cross-validation
split = 4
split_col = pixel_data_batch_1.shape[0]/split
split_col = int(split_col)




# Trainingsdaten
pixel_data_batch_1_0_75 = pixel_data_batch_1[0:3*split_col, :]
mean_pixel_data_batch_1_training = np.mean(pixel_data_batch_1, axis=0)
pixel_data_batch_1_0_75_normalized = np.subtract(pixel_data_batch_1_0_75, mean_pixel_data_batch_1_training)

# Testdaten
pixel_data_batch_1_75_100 = pixel_data_batch_1[3*split_col:4*split_col, :]
pixel_data_batch_1_75_100_normalized = np.subtract(pixel_data_batch_1_75_100, mean_pixel_data_batch_1_training)

# Labels Test
labels_data_batch_1_75_100 = labels_data_batch_1[3*split_col:4*split_col]

# Labels Training
labels_data_batch_1_0_75 = labels_data_batch_1[0:3*split_col]


#--------------------------------------------------------------
# Testing
print('Data.shape', pixel_data_batch_1_0_75_normalized.shape)
distances = compute_distances(pixel_data_batch_1_75_100_normalized, pixel_data_batch_1_0_75_normalized)
distances = np.array(distances)
print(distances.shape)

prediced_labels_from_test = predict_labels(distances, labels_data_batch_1_0_75, k=3)
print(prediced_labels_from_test)

accuracy_k = np.mean(validate_prediction(prediced_labels_from_test, labels_data_batch_1_75_100))
print(accuracy_k)

#---------------------------------------------------------------------------------------------------
# cross valitation

'''num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

data_train_folds = []
lbl_train_folds = []

data_train_folds = np.array_split(pixel_data_batch_1_0_75_normalized, num_folds)
lbl_train_folds = np.array_split(labels_data_batch_1_0_75, num_folds)
k_to_accuracies = {}

for k in k_choices: # mit verschiedenen k testen
    k_to_accuracies[k] = []
    for num_knn in range(0, num_folds): # Test und Training in die Folds unterteilen
        data_test = data_train_folds[num_knn]
        lbl_test = lbl_train_folds[num_knn]
        data_train = data_train_folds
        lbl_train = lbl_train_folds

        temp = np.delete(data_train, num_knn, 0)
        data_train = np.concatenate((temp), axis=0)
        lbl_train = np.delete(lbl_train, num_knn, 0)
        lbl_train = np.concatenate((lbl_train), axis=0)

        # für k und fold das kNN berechnene und accuracy bestimmen
        dists = compute_distances(data_test, data_train)
        y_test_pred = predict_labels(dists, lbl_train, k)

        num_correct = np.sum(y_test_pred == lbl_test)
        accuracy = float(num_correct) / data_test.shape[0]
        k_to_accuracies[k].append(accuracy)

print("Printing 5-fold accuracies for varying values of k:")
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print(f'k = {k}, accuracy = {accuracy}')

plt.figure(figsize=(14, 4))
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])

plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.xticks(np.arange(min(k_choices), max(k_choices), 2))
plt.ylabel('Cross-validation accuracy')
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.grid(True)
plt.show()'''




# output

'''plt.figure
plt.plot(k_vec, accuracy_k, 'r')
plt.title('Accuracy(k)')'''

'''plt.figure
plt.imshow(Bild1)
plt.title(label_decoder[b'label_names'][labels_data_batch_1[nr_bild]])'''



#---------------------------------------------------------------------------------------------------
# end

plt.show()
print('La fin')
cv2.waitKey(0)
