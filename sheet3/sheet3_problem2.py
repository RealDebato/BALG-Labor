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
            print(f'iteration {it} / {num_iters}: loss {loss}')

        return loss_history

    def loss(self, X_batch, y_batch, reg):
        loss = 0.0
        dW = np.zeros_like(self.W)

        num_train = X_batch.shape[0]
        scores = X_batch.dot(self.W)    # s = W*x
        scores -= np.max(scores, axis=1, keepdims=True) # der höchste score entspricht dem Ergebnis des classificators
        exp_scores = np.exp(scores)     # softmax: Scores wird in Wahrscheinlichkeit (0,1) umgewandelt
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_train), y_batch])    # cross-entropy loss (115)
        loss = np.sum(correct_logprobs) / num_train     # loss function (120)
        loss += 0.5 * reg * np.sum(self.W * self.W)

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

def compute_distances(test_batch, training_batch):
    test_batch = np.asarray(test_batch)
    training_batch = np.asarray(training_batch)
    num_test = test_batch.shape[0]
    num_train = training_batch.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sum(np.square(training_batch), axis=1) + np.sum(np.square(test_batch), axis=1)[:, np.newaxis] - 2 * np.dot(test_batch, training_batch.T)
        
    return dists

def predict_labels(dists, labels_train, k=1):
        dists = np.asarray(dists)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            sorted_dist = np.argsort(dists[i])
            closest_y = list(labels_train[sorted_dist[0:k]])
            y_pred[i] = (np.argmax(np.bincount(closest_y)))
            
        return y_pred

def validate_prediction(prediction, labels_test):
    accuracy = np.zeros(len(labels_test))
    for i in range(0, len(labels_test)):
        accuracy[i] = prediction[i] == labels_test[i]
    return accuracy

def norm_img(data_train, data_test):
    mean_img = np.mean(data_train, axis=0)
    data_train_norm = data_train - mean_img
    data_test_norm = data_test - mean_img
    data_train_norm = np.divide(data_train_norm, 255.)
    data_test_norm = np.divide(data_test_norm, 255.)

    return data_train_norm, data_test_norm



#---------------------------------------------------------------------------------------------------
# globals



#---------------------------------------------------------------------------------------------------
# data

data_reduction = 10000       # max. 10.000
if data_reduction > 10000:
    data_reduction = 10000


label_decoder = unpickle(R'sheet3\CIFAR\batches.meta.txt')             # Index = label nummer 
#print(label_decoder[b'label_names'])

data_batch_1 = unpickle(R'sheet3\CIFAR\data_batch_2.bin')

pixel_data_batch_1 = np.asarray(data_batch_1[b'data'])
pixel_data_batch_1 = pixel_data_batch_1[0:data_reduction, :]


labels_data_batch_1 = np.asarray(data_batch_1[b'labels'])
labels_data_batch_1 = labels_data_batch_1[0:data_reduction]

# Einteilung in 4 Quartiele für 4-fold cross-validation
split = 4
split_col = pixel_data_batch_1.shape[0]/split
split_col = int(split_col)




# Trainingsdaten
pixel_data_batch_1_0_75 = pixel_data_batch_1[0:3*split_col, :]

# Testdaten
pixel_data_batch_1_75_100 = pixel_data_batch_1[3*split_col:4*split_col, :]

# Labels Test
labels_data_batch_1_75_100 = labels_data_batch_1[3*split_col:4*split_col]

# Labels Training
labels_data_batch_1_0_75 = labels_data_batch_1[0:3*split_col]

data_train_norm, data_test_norm = norm_img(pixel_data_batch_1_0_75, pixel_data_batch_1_75_100)



#--------------------------------------------------------------
# Testing



#---------------------------------------------------------------------------------------------------
# softmax

softmax = SoftmaxClassifier(data_train_norm, labels_data_batch_1_0_75, data_test_norm, labels_data_batch_1_75_100)
softmax.train(learning_rate=1e-1, reg=1e-6, num_iters=1000, batch_size=400)
acc = softmax.check_accuracy()

#print(np.random.randn(data_train_norm.shape[1], 10))

# output





#---------------------------------------------------------------------------------------------------
# end

plt.show()
print('La fin')
cv2.waitKey(0)
