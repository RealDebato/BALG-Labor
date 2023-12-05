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




class multilayer_layer_perceptron():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.W = None 

    def activation(self, scores, mode='softmax'):
        if mode == 'softmax':
            pass
        elif mode == 'relu':
            pass
        elif mode == 'step':
            pass
        elif mode == 'sigmoid':
            pass
        pass

    def train(self, learning_rate=1e-3, reg=1e-5, epochs=30, bias):

        pass


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
    



#---------------------------------------------------------------------------------------------------
# functions
#---------------------------------------------------------------------------------------------------

def cifar_data_to_hue(pixel_data_cifar):
    num_img = pixel_data_cifar.shape[0]
    pixel_data_cifar = np.reshape(pixel_data_cifar, (num_img, 1024, 3))
    hue = cv2.cvtColor(pixel_data_cifar, cv2.COLOR_RGB2HSV)[:,:,0]

    return hue

def hist_hue(pixel_data_cifar):
    
    num_img = pixel_data_cifar.shape[0]
    pixel_data_cifar = np.reshape(pixel_data_cifar, (num_img, 1024, 3))
    hue = cv2.cvtColor(pixel_data_cifar, cv2.COLOR_RGB2HSV)[:,:,0]

    Hue_split = np.split(hue, num_img, axis=0)

    hist = []

    for img in range(0, num_img):
        hist_row, _ = np.histogram(Hue_split[img], bins=180, range=(0, 179), density=True)
        hist = np.append(hist, hist_row)

    hist = np.reshape(hist, (num_img, 180))
    
    return hist

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
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------
# pre-processing data
#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------
# training
#---------------------------------------------------------------------------------------------------

