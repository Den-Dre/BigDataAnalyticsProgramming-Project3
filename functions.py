# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                  TODO: Implement the functions in this file                 #
#                                                                             #
###############################################################################
import logging
import multiprocessing.managers
import pickle
import sys
from heapq import heappush, heapreplace, nlargest
from math import log, ceil
from os.path import join, dirname
from pickle import dump, HIGHEST_PROTOCOL
from statistics import mode
from time import time

import matplotlib.pyplot as plt
import numpy as np

from numpy import float32

import util

RESULTS_DIR = join(dirname(__file__), 'results')
CHOSEN_HYPERPARAMS = {
    'spambase': (16, 16),
    'covtype': (16, 32),
    'emnist_orig': (16, 32),
    'emnist': (16, 32),
    'higgs': (8, 64),
}

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

    for row_idx, test_vector in enumerate(xtest):
        pq = []
        for i, train_vector in enumerate(xtrain):
            distance = np.linalg.norm(test_vector - train_vector)
            if len(pq) < k:
                heappush(pq, (-distance, i))     # Add negative distance to sort pq by least distance (heapq is a min-heap)
            elif distance < -pq[0][0]:           # current is smaller than current largest distance (= minimum element of pq)
                heapreplace(pq, (-distance, i))  # pop pq[0] (minimum value) and push xtest[i]
        for idx, (neg_distance, i) in enumerate(nlargest(k, pq, key=lambda x: x[0])):
            indices[row_idx][idx] = i
            distances[row_idx][idx] = -neg_distance
    return indices, distances


def compute_accuracy(ytrue, ypredicted):
    """
    Return the fraction of correct predictions.
    """
    cnt = 0
    for x, y in zip(ytrue, ypredicted):
        # Predict the most occurring class among the neighbors:
        if x == mode(y):  # mode == most frequent prediction
            cnt += 1
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
    npartitions, nclusters = CHOSEN_HYPERPARAMS[dataset]
    nns = util.get_nn_instances(dataset, xtrain, ytrain, cache_partitions=True, npartitions=npartitions, nclusters=nclusters)

    accuracies = {"pqnn": 0.0, "npnn": 0.0, "sknn": 0.0}
    times = {"pqnn": 0.0, "npnn": 0.0, "sknn": 0.0}

    for nn, str_nn in zip(nns, times.keys()):
        if str_nn == "pqnn": continue
        start_time = time()
        indices, dist = nn.get_neighbors(xsample, k)
        print(f'{str_nn} neighbors: {indices}')
        print(f'{str_nn} distances: {dist}')
        times[str_nn] = time() - start_time  # Measure prediction time

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

    npartitions, nclusters = CHOSEN_HYPERPARAMS[dataset]
    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain, cache_partitions=False, npartitions=npartitions, nclusters=nclusters)

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
    retrieval_rate = 1.0  # all present in top-k of ProdQuanNN
    retrieval_rate = 0.0  # none present in top-k of ProdQuanNN
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    npartitions, nclusters = CHOSEN_HYPERPARAMS[dataset]
    print(f'Using {(npartitions, nclusters)=}')
    print(f'Using {k=}')
    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain, cache_partitions=False, npartitions=npartitions, nclusters=nclusters)

    indices_sknn, distances_sknn = sknn.get_neighbors(xsample,k)
    indices_pqnn, _ = pqnn.get_neighbors(xsample, k)
    retrieval_rate = 0
    unique_distances = 0
    seen_neighbors = []

    for ti_idx, ti in enumerate(xsample):
        # "Important note: neighbors with the exact same distance to the test instance
        # are considered the same!", thus we add at most 1 to the retrieval rate count
        # per unique neighbor, so that distances that we've seen multiple times don't
        # affect the retrieval rate (as these are assumed to be the same neighbors):
        if distances_sknn[ti_idx][0] in seen_neighbors:
            continue

        # calculate the *exact* distances of the k-NN returned by pqnn
        exact_pqnn_distances = np.array([np.linalg.norm(np.subtract(ti, xtrain[i])) for i in indices_pqnn[ti_idx]], dtype=float32)
        min_diff = min(np.abs(np.subtract(exact_pqnn_distances, distances_sknn[ti_idx][0])))
        unique_distances += 1

        # Account for some rounding errors by considering distances which differ less than 1e-7 as the same distance.
        # The error margin 1e-7 is chosen such that if we set `k` to its maximal value (`k=nsamples`), the retrieval
        # rate will be equal to 1.0
        if min_diff < 1e-7:  # exactly or approximately the same distances
            seen_neighbors.append(distances_sknn[ti_idx][0])
            retrieval_rate += 1

    print(f'total: {indices_pqnn.shape[0]=}, {unique_distances=}, {retrieval_rate=}')
    retrieval_rate /= unique_distances
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
    base = 2

    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    nexamples = len(xtrain)
    nfeatures = len(xtrain[0])
    print(f'Nb. of examples: {nexamples}')
    print(f'Nb. of features: {nfeatures}')

    npartitions_vals = [pow(base,x) for x in range(ceil(log(nfeatures, base)))]
    nclusters_vals = [pow(base,x) for x in range(ceil(log(nexamples, base))) if x < pow(base, x) < 10000]

    print(f'Values of partitions: {npartitions_vals}')
    print(f'Values of clusters: {nclusters_vals}')

    accuracies = {}
    times = {}
    logger = logging.getLogger()
    file_name = f'{dataset}-hyp{npartitions_vals[0]}-{npartitions_vals[-1]}--{nclusters_vals[0]}-{nclusters_vals[-1]}'

    for npartitions in npartitions_vals:
        for nclusters in nclusters_vals:
            print(f'Using {npartitions} partitions and {nclusters} clusters... ')
            start_time = time()
            try:
                pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain,  npartitions=npartitions, nclusters=nclusters)
                elapsed_time = time() - start_time  # Measure effect on training time, not prediction time
            except ValueError as e:
                logger.warning(f'{npartitions} partitions and {nclusters} clusters gave error: {e}')
                continue

            indices, _ = pqnn.get_neighbors(xsample, k)
            predicted_labels = [[ytrain[n_idx] for n_idx in indices_row] for indices_row in indices]

            if npartitions in times:
                times[npartitions][nclusters] = elapsed_time
                accuracies[npartitions][nclusters] = compute_accuracy(ysample, predicted_labels)
            else:
                times[npartitions] = {nclusters: elapsed_time}
                accuracies[npartitions] = {nclusters: compute_accuracy(ysample, predicted_labels)}
            try:
                save_results(times, accuracies, file_name)
            except Exception as e:
                logger.warning(f'Error while saving results: {e}')

    # save and plot results
    print('Results of times:')
    print(times)
    print('Results of accuracies:')
    print(accuracies)
    save_results(times, accuracies, file_name)
    plot_dict(times, 'Execution time', file_name=file_name)
    plot_dict(accuracies, 'Accuracy', file_name=file_name)

def run_parallel_experiments(times, accuracies, dataset, xtrain, ytrain, xsample, ysample, k, file_name, npartitions, nclusters ):
    """
    A wrapper providing the same functionality as hyperparam_task
    to be used to conduct experiments using multiple cores (multiprocessing)
    """
    logger = logging.getLogger()

    print(f'Using {npartitions} partitions and {nclusters} clusters... ')
    start_time = time()
    try:
        pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=npartitions, nclusters=nclusters)
        elapsed_time = time() - start_time
    except ValueError as e:
        logger.warning(f'{npartitions} partitions and {nclusters} clusters gave error: {e}')
        return

    indices, _ = pqnn.get_neighbors(xsample, k)
    predicted_labels = [[ytrain[n_idx] for n_idx in indices_row] for indices_row in indices]

    print(f'For {npartitions, nclusters} if test returns {npartitions in dict(times).keys()} with dict: {dict(times)}')
    if npartitions in dict(times).keys():
        tmp = times[npartitions]
        tmp[nclusters] = elapsed_time
        times[npartitions] = tmp

        tmp = accuracies[npartitions]
        tmp[nclusters] = compute_accuracy(ysample, predicted_labels)
        accuracies[npartitions] = tmp
    else:
        times[npartitions] = {nclusters: elapsed_time}
        accuracies[npartitions] = {nclusters: compute_accuracy(ysample, predicted_labels)}
    try:
        save_results(times, accuracies, file_name)
    except Exception as e:
        logger.warning(f'Error while saving results: {e}')
    return times, accuracies

def plot_dict(d, ylabel, file_name=None, base=2):
    """
    Plot the results provided in a dictionary object

    :param d: the experiment results to be plottted
    :param ylabel: the label to give to the y axis
    :param file_name: the name of the file to which the plot will be saved
    :param base: the base for the logarithmic axes
    """
    _, ax = plt.subplots()
    for npartitions in d.keys():
        ax.plot(list(d[npartitions].keys())[:], list(d[npartitions].values())[:], label=f'{npartitions} partitions')
    ax.set_xscale('log', base=base)
    ax.set_yscale('log', base=10)
    plt.title(f'Accuracy in function of number of clusters')
    plt.ylabel(f'{ylabel}')
    plt.xlabel('Number of clusters')
    plt.legend()
    if file_name:
        plt.savefig(join(RESULTS_DIR, 'plots', f'{ylabel}--{file_name}'))
    try:
        plt.show()
    except Exception as e:
        logging.getLogger('plotter').warning(e)

def save_results(times, accuracies, file_name):
    """
    Save the experiment results to a file

    :param times: the execution times obtained by an experiment
    :param accuracies: the accuracies obtained by an experiment
    :param file_name: the file name of the file to which the data will be saved
    """
    if isinstance(times, multiprocessing.managers.DictProxy):
        times = dict(times)
    if isinstance(accuracies, multiprocessing.managers.DictProxy):
        accuracies = dict(accuracies)
    with open(join(RESULTS_DIR, 'data', f'time--{file_name}.pckl'), 'wb') as f:
        dump(times, f, protocol=HIGHEST_PROTOCOL)
    with open(join(RESULTS_DIR, 'data', f'acc--{file_name}.pckl'), 'wb') as f:
        dump(accuracies, f, protocol=HIGHEST_PROTOCOL)

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
