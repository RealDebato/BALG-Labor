import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download


class SoftmaxHOG(object):
    def __init__(self):
        pass


def preprocess_data(data_train, data_test):
    mean_img = np.mean(data_train, axis=0)
    data_train_norm = data_train - mean_img
    data_test_norm = data_test - mean_img
    data_train_norm = np.divide(data_train_norm, 255.)
    data_test_norm = np.divide(data_test_norm, 255.)

    return data_train_norm, data_test_norm


def classifyKNN(data_train, lbl_train, data_test, lbl_test):
    classifier = SoftmaxHOG()


if __name__ == '__main__':
    dataset_dir = './data/cifar-10-batches-py'
    data_train, lbl_train, data_test, lbl_test = load_dataset(dataset_dir, plot=False)

    # Data subsampling
    num_training = 5000
    mask = list(range(num_training))
    data_train = data_train[mask]
    lbl_train = lbl_train[mask]

    num_test = 1000
    mask = list(range(num_test))
    data_test = data_test[mask]
    lbl_test = lbl_test[mask]

    data_train = np.reshape(data_train, (data_train.shape[0], -1))
    data_test = np.reshape(data_test, (data_test.shape[0], -1))

    print('\nSubsampling successful!')
    print(f'Train data shape: {data_train.shape}')
    print(f'Test data shape: {data_test.shape}')

    data_train_norm, data_test_norm = preprocess_data(data_train, data_test)
