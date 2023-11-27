import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download


class SoftmaxClassifier():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.W = None

    def train(self, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, reset_weights=False):
        ''' Train the classifier using stochastic gradient descent. '''
        num_train, dim = self.x_train.shape
        num_classes = np.max(self.y_train) + 1
        if self.W is None or reset_weights == True:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Sample batch_size elements from the training data and their corresponding labels to use in this round of gradient descent.
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = self.x_train[indices]
            y_batch = self.y_train[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W += -learning_rate * grad
            print(f'iteration {it} / {num_iters}: loss {loss}')

        return loss_history

    def loss(self, X_batch, y_batch, reg):
        ''' Compute the loss function and its derivative. '''
        # initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        # compute the loss and the gradient
        num_train = X_batch.shape[0]
        scores = X_batch.dot(self.W)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_train), y_batch])
        loss = np.sum(correct_logprobs) / num_train
        loss += 0.5 * reg * np.sum(self.W * self.W)

        dscores = probs
        dscores[range(num_train), y_batch] -= 1
        dscores /= num_train
        dW = np.dot(X_batch.T, dscores)
        dW += reg * self.W

        return loss, dW

    def check_accuracy(self, test_data_indices=None):
        ''' Check the accuracy of the classifier on the test data. '''
        if test_data_indices is None:
            test_data_indices = range(self.x_test.shape[0])
        num_correct = 0
        num_test = len(test_data_indices)
        for i, idx in enumerate(test_data_indices):
            scores = self.x_test[idx].dot(self.W)
            y_pred = np.argmax(scores)
            if y_pred == self.y_test[idx]:
                num_correct += 1
        acc = float(num_correct) / num_test
        msg = f'Got {num_correct} / {num_test} correct; ' f'accuracy is {(acc * 100):.2f}%'
        print(msg)
        return acc


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


def sample_data(data_train, lbl_train, data_test, lbl_test, num_training=5000, num_test=1000):
    # Data subsampling
    mask = list(range(num_training))
    data_train = data_train[mask]
    lbl_train_sub = lbl_train[mask]

    mask = list(range(num_test))
    data_test = data_test[mask]
    lbl_test_sub = lbl_test[mask]

    data_train_sub = np.reshape(data_train, (data_train.shape[0], -1))
    data_test_sub = np.reshape(data_test, (data_test.shape[0], -1))

    print('\nSubsampling successful!')
    print(f'Train data shape: {data_train.shape}')
    print(f'Test data shape: {data_test.shape}')

    return data_train_sub, lbl_train_sub, data_test_sub, lbl_test_sub


if __name__ == '__main__':
    dataset_dir = './data/cifar-10-batches-py'
    data_train, lbl_train, data_test, lbl_test = load_dataset(dataset_dir, plot=False)

    data_train_sub, lbl_train_sub, data_test_sub, lbl_test_sub = sample_data(data_train, lbl_train, data_test, lbl_test, num_training=50000, num_test=10000)
    data_train_norm, data_test_norm = preprocess_data(data_train_sub, data_test_sub)

    softmax = SoftmaxClassifier(data_train_norm, lbl_train_sub, data_test_norm, lbl_test_sub)
    softmax.train(learning_rate=1e-3, reg=1e-6, num_iters=1500, batch_size=200)
    acc = softmax.check_accuracy()




