import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


#---------------------------------------------------------------------------------------------------
# classes
#---------------------------------------------------------------------------------------------------

class SoftmaxClassifier():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.W = None           # s = W*x

    def train(self, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, reset_weights=False):
        num_train, dim = self.x_train.shape         # num_train = Anzahl Trainingsbilder, dim = 3072px
        num_classes = np.max(self.y_train) + 1      # Klassen 0...9, d. h. insg. 10 Klassen
        if self.W is None or reset_weights == True:
            self.W = 0.001 * np.random.randn(dim, num_classes)  # zufällige Gewichtungen für alles Klassen erstellen

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Sample batch_size elements from the training data and their corresponding labels to use in this round of gradient descent. ---> mini-batch
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = self.x_train[indices]
            y_batch = self.y_train[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)   # Grad is derivative of W 
            loss_history.append(loss)

            # perform parameter update
            self.W += -learning_rate * grad     # W wird entlang des Gradienten verschoben
            #print(f'iteration {it} / {num_iters}: loss {loss}')

        return loss_history

    def loss(self, X_batch, y_batch, reg):
        loss = 0.0
        dW = np.zeros_like(self.W)      # Gradient von W

        num_train = X_batch.shape[0]
        scores = X_batch.dot(self.W)    # s = W*x
        #scores -= np.max(scores, axis=1, keepdims=True) # der höchste score entspricht dem Ergebnis des classificators
        exp_scores = np.exp(scores)   
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax: Scores wird in Wahrscheinlichkeit (0,1) umgewandelt
        correct_logprobs = -np.log(probs[range(num_train), y_batch])    # cross-entropy loss (115)
        loss = np.sum(correct_logprobs) / num_train     # gemittelter Loss über correct_logprobs
        loss += 0.5 * reg * np.sum(self.W * self.W)     # loss function / squared error function

        # Differenz/Gradient der Gewichtungen W bestimmen
        dscores = probs     # Hierfür werden die Wahrscheinlichenkeiten verwendet
        dscores[range(num_train), y_batch] -= 1     # Differenz zum tatsächlichen score
        dscores /= num_train                        # mittlere Differenz
        dW = np.dot(X_batch.T, dscores)             # s = W*x <=> W = x.T*s
        dW += reg * self.W                          # dW = gradient wird leicht (um reg) verschoben, um loss zu verringern

        return loss, dW

    def check_accuracy(self, test_data_indices=None):
        if test_data_indices is None:
            test_data_indices = range(self.x_test.shape[0])     
        num_correct = 0
        num_test = len(test_data_indices)
        for i, idx in enumerate(test_data_indices):     # indices der Testdaten
            scores = self.x_test[idx].dot(self.W)       # score von jedem Testbild wird mit aktuellem W berechnet
            y_pred = np.argmax(scores)                  # Klasse mit höchstem score gewinnt --> predicted class
            if y_pred == self.y_test[idx]:              # immer wenn predicted class == true class zähl hoch
                num_correct += 1
        acc = float(num_correct) / num_test             # Verhältnis correct prediction / Anzahl Testbilder
        msg = f'Got {num_correct} / {num_test} correct; ' f'accuracy is {(acc * 100):.2f}%'
        print(msg)
        return acc


#---------------------------------------------------------------------------------------------------
# functions
#---------------------------------------------------------------------------------------------------

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
    #print('num test', num_test)
    num_train = training_batch.shape[0]
    #print('num train', num_train)
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
            #print('sorted dist.shape', sorted_dist.shape)
            closest_y = list(labels_train[sorted_dist[0:k]])
            #print('sorted dist 0:k',sorted_dist[0:k])
            #print('Closest_y', closest_y)
            y_pred[i] = (np.argmax(np.bincount(closest_y))) #predicted class ist die häufigste der k-nächsten Klassen
            
        #print(y_pred) 
        return y_pred

def validate_prediction(prediction, labels_test):
    accuracy = np.zeros(len(labels_test))
    for i in range(0, len(labels_test)):
        accuracy[i] = prediction[i] == labels_test[i] # Vergleich zwischen prediction und tatsächlichem label
    return accuracy

def classification_hog(pixel_data):
    
    pixels_per_cell = (8, 8)    # --> 16 cells
    cells_per_block = (4, 4)    # --> 1 block
    orientations = 10
    num_images = pixel_data.shape[0]
    len_hog = int(orientations * 1024/(pixels_per_cell[0]*pixels_per_cell[1]))
    pixel_data = pixel_data.reshape(num_images, 3, 32, 32).transpose(0,2,3,1).astype("float")
    hog_list = []
    for img in range(0, num_images):
        hog_feats = ski.feature.hog(pixel_data[img], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L1', visualize=False, feature_vector=True, channel_axis=2)
        hog_list = np.append(hog_list, hog_feats)
    
    hog_list = np.reshape(hog_list, (num_images, len_hog))
    
    return hog_list

def hist_hue(pixel_data):
    
    r, g, b = np.split(pixel_data, 3, axis=1)
    rgb = np.stack((r, g, b), axis=2)
    max_color_channel = np.argmax(rgb, axis=2) 
    num_img = pixel_data.shape[0]

    Hue = np.zeros_like(max_color_channel)
    Hue[max_color_channel==0] = ((rgb[..., 1]-rgb[..., 2])[max_color_channel==0]) * 60 
    Hue[max_color_channel==1] = (2 + (rgb[..., 2]-rgb[..., 0])[max_color_channel==1]) * 60
    Hue[max_color_channel==2] = (4 + (rgb[..., 2]-rgb[..., 0])[max_color_channel==2]) * 60

    Hue = np.asarray(Hue)
    #Hue = np.where(Hue<0, Hue+360, Hue)
    Hue[Hue<0] = Hue[Hue<0] + 360
    Hue = np.asarray(Hue)

    Hue_split = np.split(Hue, num_img, axis=0)

    hist = []

    for img in range(0, num_img):
        hist_row, _ = np.histogram(Hue_split[img], bins=360, range=(0, 359), density=True)
        hist = np.append(hist, hist_row)

    hist = np.reshape(hist, (num_img, 360))
    
    return hist
    
#---------------------------------------------------------------------------------------------------
# globals



#---------------------------------------------------------------------------------------------------
# data

numbers_train = 5000
numbers_test = 1000
numbers_validate = 100


label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
#print(label_decoder[b'label_names'])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_1.bin')   # --> Train
data_batch_2 = unpickle(R'sheet3\CIFAR\data_batch_2.bin')   # --> Test
data_batch_3 = unpickle(R'sheet3\CIFAR\data_batch_3.bin')   # --> Validate

pixel_data_batch_1 = np.asarray(data_batch_1[b'data'])
pixel_data_batch_2 = np.asarray(data_batch_2[b'data'])
pixel_data_batch_3 = np.asarray(data_batch_3[b'data'])


pixel_data_training = pixel_data_batch_1[0:numbers_train, :]
pixel_data_testing = pixel_data_batch_2[0:numbers_test, :]
pixel_data_validation = pixel_data_batch_3[0:numbers_validate, :]
#print('Pixel Data Batch 1 ',pixel_data_training)
#print('Pixel Data Batch 1 shape ',pixel_data_training.shape)

labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])   # --> Train
labels_data_batch_2 = np.asarray(data_batch_2[b'labels'])   # --> Test
labels_data_batch_3 = np.asarray(data_batch_3[b'labels'])   # --> Validate


labels_data_training = labels_data_batch_1[0:numbers_train]
labels_data_testing = labels_data_batch_2[0:numbers_test]
labels_data_validation = labels_data_batch_3[0:numbers_validate]
#print('Labels Data Batch 1 ',labels_data_training)
#print('Labels Data Batch 1 shape ',labels_data_training.shape)


mean_pixel_data_training = np.mean(pixel_data_training, axis=0)

pixel_data_training_normalized = np.subtract(pixel_data_training, mean_pixel_data_training)
pixel_data_testing_normalized = np.subtract(pixel_data_testing, mean_pixel_data_training)
pixel_data_validation_normalized = np.subtract(pixel_data_validation, mean_pixel_data_training)

pixel_data_training_normalized = np.divide(pixel_data_training_normalized, 255.)
pixel_data_testing_normalized = np.divide(pixel_data_testing_normalized, 255.)
pixel_data_validation_normalized = np.divide(pixel_data_validation_normalized, 255)

# new features ------------------------------------------------
# hue
hue_training = hist_hue(pixel_data_training)
hue_testing = hist_hue(pixel_data_testing)
hue_validation = hist_hue(pixel_data_validation)

mean_hue = np.mean(hue_training, axis=0)

hue_training_normalized = np.subtract(hue_training, mean_hue)
hue_testing_normalized = np.subtract(hue_testing, mean_hue)
hue_validation_normalized = np.subtract(hue_validation, mean_hue)

hue_training_normalized = np.divide(hue_training_normalized, 255.)
hue_testing_normalized = np.divide(hue_testing_normalized, 255.)
hue_validation_normalized = np.divide(hue_validation_normalized, 255)

# hog
hog_training = classification_hog(pixel_data_training)
hog_testing = classification_hog(pixel_data_testing)
hog_validation = classification_hog(pixel_data_validation)

mean_hog = np.mean(hog_training)

hog_training_normalized = np.subtract(hog_training, mean_hog)
hog_testing_normalized = np.subtract(hog_testing, mean_hog)
hog_validation_normalized = np.subtract(hog_validation, mean_hog)

max_hog = np.amax(hog_training_normalized)
min_hog = np.abs(np.amin(hog_training_normalized))
if max_hog > min_hog:
    scale = max_hog
else:
    scale = min_hog

hog_training_normalized = np.divide(hog_training_normalized, scale)
hog_testing_normalized = np.divide(hog_testing_normalized, scale)
hog_validation_normalized = np.divide(hog_validation_normalized, scale)


print('shape hue training', hue_training_normalized.shape)
print('shape hog training', hog_training_normalized.shape)


# concatenat features
features_train = np.concatenate((hue_training_normalized, hog_training_normalized), axis=1)
features_test = np.concatenate((hue_testing_normalized, hog_testing_normalized), axis=1)
features_validate = np.concatenate((hue_validation_normalized, hog_validation_normalized), axis=1)

#--------------------------------------------------------------
# Testing

distances = compute_distances(features_test, features_train)
distances = np.array(distances)
#print(distances.shape)

prediced_labels_from_test = predict_labels(distances, labels_data_training, k=10)
#print(prediced_labels_from_test)

accuracy_k = np.mean(validate_prediction(prediced_labels_from_test, labels_data_testing))
print('Accuracy kNN:')
print(accuracy_k)

#---------------------------------------------------------------------------------------------------
# cross valitation

'''num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20]

data_train_folds = []
lbl_train_folds = []

data_train_folds = np.array_split(pixel_data_training_normalized, num_folds)
lbl_train_folds = np.array_split(labels_data_training, num_folds)
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

# softmax ---------------------------------------------------------------------------------

# Pixel Data
softmax = SoftmaxClassifier(pixel_data_training_normalized, labels_data_training, pixel_data_testing_normalized, labels_data_testing)
softmax.train(learning_rate=1e-3, reg=1e-5, num_iters=1000, batch_size=300)
print('Accuracy Softmax pixel:')
acc = softmax.check_accuracy()


# Hue + Hog
softmax = SoftmaxClassifier(features_train, labels_data_training, features_test, labels_data_testing)
softmax.train(learning_rate=1e-1, reg=1e-5, num_iters=1000, batch_size=300)
print('Accuracy Softmax Hue + HOG:')
acc = softmax.check_accuracy()

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
