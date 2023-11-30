import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


#---------------------------------------------------------------------------------
# classes
#---------------------------------------------------------------------------------

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, data, lbl):
        self.data_train = data
        self.lbl_train = lbl

    def predict(self, data, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(data)
        else:
            raise ValueError(f'Invalid value for {num_loops} num_loops')
        return self.predict_labels(dists, k=k)

    def compute_distances(self, data):
        num_test = data.shape[0]
        num_train = self.data_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(np.sum(np.square(self.data_train), axis=1) + np.sum(np.square(data), axis=1)[:, np.newaxis] - 2 * np.dot(data, self.data_train.T))
        pass
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.lbl_train[sorted_dist[0:k]])
            pass
            y_pred[i] = (np.argmax(np.bincount(closest_y)))
            pass
        return y_pred

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
        dW = np.zeros_like(self.W)      # Gradient von W

        num_train = X_batch.shape[0]
        scores = X_batch.dot(self.W)    # s = W*x
        scores -= np.max(scores, axis=1, keepdims=True) # der höchste score entspricht dem Ergebnis des classificators
        exp_scores = np.exp(scores)     # softmax: Scores wird in Wahrscheinlichkeit (0,1) umgewandelt
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_train), y_batch])    # cross-entropy loss (115)
        loss = np.sum(correct_logprobs) / num_train     # loss function (120)
        loss += 0.5 * reg * np.sum(self.W * self.W)     #

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

#---------------------------------------------------------------------------------
# functions
#---------------------------------------------------------------------------------
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def norm_img(data_train, data_test):
    mean_img = np.mean(data_train, axis=0)
    data_train_norm = data_train - mean_img
    data_test_norm = data_test - mean_img
    data_train_norm = np.divide(data_train_norm, 255.)
    data_test_norm = np.divide(data_test_norm, 255.)

    return data_train_norm, data_test_norm

def rgb_to_hue(pixel_data):
    
    r, g, b = np.split(pixel_data, 3, axis=1)
    rgb = np.stack((r, g, b), axis=2)
    max_color_channel = np.argmax(rgb, axis=2) 

    Hue = np.zeros_like(max_color_channel)
    Hue[max_color_channel==0] = (rgb[..., 1]-rgb[..., 2])[max_color_channel==0]
    Hue[max_color_channel==1] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==1]
    Hue[max_color_channel==2] = (rgb[..., 2]-rgb[..., 0])[max_color_channel==2]

    return Hue

def classification_hog(pixel_data):
    
    pixels_per_cell = (8, 8)    # --> 16 cells
    cells_per_block = (4, 4)    # --> 1 block
    orientations = 9
    num_images = pixel_data.shape[0]
    len_hog = int(orientations * 1024/(pixels_per_cell[0]*pixels_per_cell[1]))
    pixel_data = pixel_data.reshape(num_images, 3, 32, 32).transpose(0,2,3,1).astype("float")
    hog_list = []
    for img in range(0, num_images):
        hog = ski.feature.hog(pixel_data[img], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2', visualize=False, feature_vector=True, channel_axis=2)
        hog_list = np.append(hog_list, hog, axis=0)
    
    hog_list = np.reshape(hog_list, (num_images, len_hog))
    
    return hog_list
#---------------------------------------------------------------------------------
# data
#---------------------------------------------------------------------------------

training = unpickle(R'sheet3\CIFAR\data_batch_1.bin')
testing = unpickle(R'sheet3\CIFAR\test_batch.bin')
validation = unpickle(R'sheet3\CIFAR\data_batch_2.bin')

data_train = np.asarray(training[b'data'])[0:10]
labels_train = np.asarray(training[b'labels'])[0:10]
data_test = np.asarray(testing[b'data'])[0:10]
labels_test = np.asarray(testing[b'labels'])[0:10]
data_validate = np.asarray(validation[b'data'])[0:10]
labels_validate = np.asarray(validation[b'labels'])[0:10]

# features----------------------------------------------------
data_train_hue = rgb_to_hue(data_train)
data_test_hue = rgb_to_hue(data_test)
data_validate_hue = rgb_to_hue(data_validate)

data_train_hog = classification_hog(data_train)
data_test_hog = classification_hog(data_test)
data_validate_hog = classification_hog(data_validate)

features_train = np.concatenate((data_train_hue, data_train_hog), axis=1)
features_test = np.concatenate((data_test_hue, data_test_hog), axis=1)
features_validate = np.concatenate((data_validate_hue, data_validate_hog), axis=1)

#---------------------------------------------------------------------------------
# main
#---------------------------------------------------------------------------------







#---------------------------------------------------------------------------------
# output
#---------------------------------------------------------------------------------







#---------------------------------------------------------------------------------
# end
#---------------------------------------------------------------------------------


plt.show()
print('La fin')
cv2.waitKey(0)