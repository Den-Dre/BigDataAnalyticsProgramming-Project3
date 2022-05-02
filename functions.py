# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                  TODO: Implement the functions in this file                 #
#                                                                             #
###############################################################################


import util
import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heapreplace

def numpy_nn_get_neighbors(xtrain, xtest, k):
    """
    Compute the `k` nearest neighbors in `xtrain` of each instance in `xtest`

    This method should return a pair `(indices, distances)` of (N x k)
    matrices, with `N` the number of rows in `xtest`. The `j`th column (j=0..k)
    should contain the indices of and the distances to the `j`th nearest
    neighbor for each row in `xtest` respectively.
    """
    indices = np.zeros((xtest.shape[0], k), dtype=int)
    distances = np.zeros((xtest.shape[0], k), dtype=float)
    max_dist = -1

    for row_idx, test_vector in enumerate(xtest):
        pq = []
        for i, train_vector in enumerate(xtrain):
            distance = np.linalg.norm(test_vector - train_vector)
            if len(pq) < k:
                # Add negative distance to sort pq by least distance
                heappush(pq, (i, -distance))
                max_dist = -pq[0][1]
            elif distance < max_dist:
                heapreplace(pq, (i, -distance))
                max_dist = -pq[0][1]
        indices[row_idx] = [tup[0] for tup in pq]
        distances[row_idx] = [-tup[1] for tup in pq]
        # https://stackoverflow.com/a/12974504/15482295
        # indices[row_idx], distances[row_idx] = [list(x) for x in zip(*pq)]  # minus must still be added...
    print(distances[:k+1])
    return indices, distances


def compute_accuracy(ytrue, ypredicted):
    """
    Return the fraction of correct predictions.
    """
    acc = 0.0
    return acc


def time_and_accuracy_task(dataset, k, n, seed):
    """
    Measure the time and accuracy of ProdQuanNN, NumpyNN, and SklearnNN on `n`
    randomly selected examples

    Make sure to keep the output format (a tuple of dicts with keys 'pqnn',
            'npnn', and 'sknn') unchanged!
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)
    pqnn, npnn, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
                                             cache_partitions=True)

    accuracies = {"pqnn": 0.0, "npnn": 0.0, "sknn": 0.0}
    times = {"pqnn": 0.0, "npnn": 0.0, "sknn": 0.0}

    # TODO use the methods in the base class `BaseNN` to classify the instances
    # in `xsample`. Then compute the accuracy with your implementation of
    # `compute_accuracy` above using the true labels `ysample` and your
    # predicted values.
    pqnn.get_neighbors(xtest, k)
    # indices, distances = npnn.get_neighbors(xtest, k)
    # print('Indices: ', indices)
    # print('Distances: ',  distances)

    return accuracies, times


def distance_absolute_error_task(dataset, k, n, seed):
    """
    Compute the mean absolute error between the distances computed by product
    quantization and the distances computed by scikit-learn.

    Return a single real value.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
                                          cache_partitions=True)

    mean_abs_dist = 0.0  # TODO compute this value

    return mean_abs_dist


def retrieval_task(dataset, k, n, seed):
    """
    How often is scikit-learn's nearest neighbor in the top `k` neighbors of
    ProdQuanNN?

    Important note: neighbors with the exact same distance to the test instance
    are considered the same!

    Return a single real value between 0 and 1.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    retrieval_rate = 1.0  # all present in top-k of ProdQuanNN
    retrieval_rate = 0.0  # none present in top-k of ProdQuanNN

    # TODO compute retrieval_rate

    return retrieval_rate


def hyperparam_task(dataset, k, n, seed):
    """
    Optimize the hyper-parameters `npartitions` and  `nclusters` of ProdQuanNN.
    Produce a plot that shows how each parameter setting affects the NN
    classifier's accuracy.

    What is the effect on the training time?

    Make sure `n` is large enough. Keep `k` fixed.

    You do not have to return anything in this function. Use this place to do
    the hyper-parameter optimization and produce the plots. Feel free to add
    additional helper functions if you want, but keep them all in this file.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    # TODO optimize the hyper parameters of ProdQuanNN and produce plot


def plot_task(dataset, k, n, seed):
    """
    This is a fun function for you to play with and visualize the resutls of
    your implementations (emnist and emnist_orig only).
    """
    if dataset != "emnist" and dataset != "emnist_orig":
        raise ValueError("Can only plot emnist and emnist_orig")

    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)

    if n > 10:
        n = 10
        print(f"too many samples to plot, showing only first {n}")

    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
                                          cache_partitions=True)
    pqnn_index, _ = pqnn.get_neighbors(xsample, k)
    sknn_index, _ = sknn.get_neighbors(xsample, k)

    # `emnist` is a transformed dataset, load the original `emnist_orig` to
    # plot the result (the instances are in the same order)
    if dataset == "emnist":
        xtrain, xtest, ytrain, ytest = util.load_dataset("emnist_orig")
        xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    for index, title in zip([pqnn_index, sknn_index], ["pqnn", "sknn"]):
        fig, axs = plt.subplots(xsample.shape[0], 1 + k)
        fig.suptitle(title)
        for i in range(xsample.shape[0]):
            lab = util.decode_emnist_label(ysample[i])
            axs[i, 0].imshow(xsample[i].reshape((28, 28)).T, cmap="binary")
            axs[i, 0].set_xlabel(f"label {lab}")
            for kk in range(k):
                idx = index[i, kk]
                lab = util.decode_emnist_label(ytrain[idx])
                axs[i, kk + 1].imshow(xtrain[idx].reshape((28, 28)).T, cmap="binary")
                axs[i, kk + 1].set_xlabel(f"label {lab} ({idx})")
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0, 0].set_title("Query")
        for kk, ax in enumerate(axs[0, 1:]):
            ax.set_title(f"Neighbor {kk}")
    plt.show()
