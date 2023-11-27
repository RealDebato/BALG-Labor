import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download


class SoftmaxHOG(object):
    def __init__(self):
        pass


def load_dataset(dataset_dir, plot):
    # Download dataset
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    download.download_and_extract(url, download_dir)
    data_train, lbl_train, data_test, lbl_test = data_utils.load_CIFAR10(dataset_dir)

    # Checking the size of the training and testing data
    print('Loading dataset successful!')
    print('Training data shape: ', data_train.shape)
    print('Training labels shape: ', lbl_train.shape)
    print('Test data shape: ', data_test.shape)
    print('Test labels shape: ', lbl_test.shape)

    # Visualizing dataset samples
    if plot:
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            index = np.flatnonzero(lbl_train == y)
            index = np.random.choice(index, samples_per_class, replace=False)
            for i, idx in enumerate(index):
                plt_index = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_index)
                plt.imshow(data_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.suptitle('Dataset samples')
        plt.show()

    return data_train, lbl_train, data_test, lbl_test


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
