# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                  TODO: Implement the functions in this file                 #
#                                                                             #
###############################################################################
import logging
from collections import Counter
from time import time
from heapq import heappush, heapreplace
from statistics import mode

import matplotlib.pyplot as plt
import numpy as np

import util


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
    return indices, distances


def compute_accuracy(ytrue, ypredicted):
    """
    Return the fraction of correct predictions.
    """
    cnt = 0
    for x, y in zip(ytrue, ypredicted):
        # Predict the most occurring class among the neighbors:
        print(f'x: {x}, y: {y}')
        if Counter(y)[x] >= len(y) / 2:
            cnt += 1
        # if x == mode(y):  # mode == most frequent prediction--task retrieval
        #     cnt += 1
    acc = cnt / len(ytrue)
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
    nns = util.get_nn_instances(dataset, xtrain, ytrain,
                                             cache_partitions=True)

    accuracies = {"pqnn": 0.0, "npnn": 0.0, "sknn": 0.0}
    times = {"pqnn": 0.0, "npnn": 0.0, "sknn": 0.0}

    # TODO use the methods in the base class `BaseNN` to classify the instances
    #   in `xsample`. Then compute the accuracy with your implementation of
    #   `compute_accuracy` above using the true labels `ysample` and your
    #   predicted values.

    for nn, str_nn in zip(nns, times.keys()):
        start_time = time()
        indices, dist = nn.get_neighbors(xsample, k)
        times[str_nn] = time() - start_time
        print(f'{str_nn}: {indices}')
        # print(f'{str_nn}: {dist}')

        predicted_labels = [[ytrain[n_idx] for n_idx in indices_row] for indices_row in indices]
        accuracies[str_nn] = compute_accuracy(ysample, predicted_labels)

    return accuracies, times


def distance_absolute_error_task(dataset, k, n, seed):
    """
    Compute the mean absolute error between the distances computed by product
    quantization and the distances computed by scikit-learn.

    Return a single real value.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain, cache_partitions=True)

    _, distances_pqnn = pqnn.get_neighbors(xsample, k=k)
    _, distances_sknn = sknn.get_neighbors(xsample, k=k)

    # mean_abs_dist = np.sum(np.absolute(distances_sknn - distances_pqnn)) / distances_pqnn.size
    mean_abs_dist = np.mean(np.absolute(distances_sknn - distances_pqnn))

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

    pqnn, npnn, sknn = util.get_nn_instances(dataset, xtrain, ytrain, cache_partitions=True, nclusters=8, npartitions=4)

    indices_sknn, distances_sknn = sknn.get_neighbors(xsample)
    indices_pqnn, _ = pqnn.get_neighbors(xsample, k)

    retrieval_rate = 0.0

    for ti_idx, ti in enumerate(xsample):
        # calculate the *exact* distances of the k-NN returned by pqnn
        exact_pqnn_distances = [np.linalg.norm(ti - xtrain[i]) for i in indices_pqnn[ti_idx]]
        if distances_sknn[ti_idx][0] in exact_pqnn_distances:
            # "Important note: neighbors with the exact same distance to the test instance
            # are considered the same!", thus we add at most 1 to the retrieval rate count:
            retrieval_rate += 1


    # old, incorrect interpretation
    # for sknn_neighbor, pqnn_neighbors in zip(indices_sknn, indices_pqnn):
    #     if sknn_neighbor[0] in pqnn_neighbors:
    #         retrieval_rate += 1

    # retrieval_rate = 1.0  # all present in top-k of ProdQuanNN
    # retrieval_rate = 0.0  # none present in top-k of ProdQuanNN

    retrieval_rate /= indices_pqnn.shape[0]
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
    from util import PROD_QUAN_SETTINGS

    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    npartitions_val = PROD_QUAN_SETTINGS[dataset]['npartitions']
    p_step = max(int(npartitions_val / 10), 1)
    npartitions_vals = np.arange(max(npartitions_val - 5 * p_step, 1), npartitions_val + 6 * p_step, p_step)
    npartitions_vals = list(filter(lambda x: x > 0, npartitions_vals))

    nclusters_val = PROD_QUAN_SETTINGS[dataset]['nclusters']
    c_step = max(int(nclusters_val / 10), 1)
    nclusters_vals = np.arange(max(nclusters_val - 5 * c_step, 1), nclusters_val + 5 * c_step, c_step)
    nclusters_vals = list(filter(lambda x: x > 0, nclusters_vals))

    times = {}
    accuracies = {}
    logger = logging.getLogger()

    for npartitions in npartitions_vals:
        for nclusters in nclusters_vals:
            print(f'Using {npartitions} partitions and {nclusters} clusters... ')
            start_time = time()
            try:  # TODO fix this: this fails when n is not sufficiently large
                pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=npartitions, nclusters=nclusters)
            except ValueError as e:
                logger.error(f'{npartitions} partitions and {nclusters} clusters gave error: {e}')
                continue

            elapsed_time = time() - start_time
            indices, _ = pqnn.get_neighbors(xsample, k)
            predicted_labels = [[ytrain[n_idx] for n_idx in indices_row] for indices_row in indices]

            if npartitions in times:
                times[npartitions][nclusters] = elapsed_time
                accuracies[npartitions][nclusters] = compute_accuracy(ysample, predicted_labels)
            else:
                times[npartitions] = {nclusters: elapsed_time}
                accuracies[npartitions] = {nclusters: compute_accuracy(ysample, predicted_labels)}

    plot_dict(times, 'Execution time')
    plot_dict(accuracies, 'Accuracy')

    # TODO optimize the hyper parameters of ProdQuanNN and produce plot


def plot_dict(dict, ylabel):
    _, ax = plt.subplots()
    for npartitions in dict.keys():
        ax.plot(dict[npartitions].keys(), dict[npartitions].values(), label=f'{npartitions} partitions')
    plt.title(f'{ylabel} in function of number of clusters')
    plt.ylabel(f'{ylabel} (s)')
    plt.xlabel('Number of clusters')
    plt.legend()
    plt.show()


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
