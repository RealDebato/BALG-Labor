import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download


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


def CrossValidation(data_train, lbl_train):
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    data_train_folds = []
    lbl_train_folds = []

    data_train_folds = np.array_split(data_train, num_folds)
    lbl_train_folds = np.array_split(lbl_train, num_folds)
    k_to_accuracies = {}

    for k in k_choices:
        k_to_accuracies[k] = []
        for num_knn in range(0, num_folds):
            data_test = data_train_folds[num_knn]
            lbl_test = lbl_train_folds[num_knn]
            data_train = data_train_folds
            lbl_train = lbl_train_folds

            temp = np.delete(data_train, num_knn, 0)
            data_train = np.concatenate((temp), axis=0)
            lbl_train = np.delete(lbl_train, num_knn, 0)
            lbl_train = np.concatenate((lbl_train), axis=0)

            classifier = KNearestNeighbor()
            classifier.train(data_train, lbl_train)
            dists = classifier.compute_distances(data_test)
            y_test_pred = classifier.predict_labels(dists, k)

            num_correct = np.sum(y_test_pred == lbl_test)
            accuracy = float(num_correct) / num_test
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
    plt.show()


def classifyKNN(data_train, lbl_train, data_test, lbl_test, set_k=3):
    classifier = KNearestNeighbor()
    classifier.train(data_train, lbl_train)
    dists = classifier.compute_distances(data_test)
    y_test_pred = classifier.predict_labels(dists, k=set_k)

    num_correct = np.sum(y_test_pred == lbl_test)
    accuracy = float(num_correct) / num_test
    print('\nResults:')
    print(f'Got {num_correct} / {num_test} correct => accuracy: {accuracy * 100}%')


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

    classifyKNN(data_train, lbl_train, data_test, lbl_test, set_k=3)
    CrossValidation(data_train, lbl_train)

    # Choosing best value of k based on cross-validation results
    best_k = 10
    classifyKNN(data_train, lbl_train, data_test, lbl_test, set_k=best_k)
