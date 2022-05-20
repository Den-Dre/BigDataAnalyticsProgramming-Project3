# author: Laurens Devos
# author: Maaike Van Roy
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                          DO NOT MODIFY THIS FILE!                           #
#                                                                             #
###############################################################################

import os, sys, timeit
BASE_DIR = os.path.dirname(__file__)
VERBOSE = 0

import joblib
import numpy as np
from scipy.stats import mode
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

try:
    import prod_quan_nn
except ModuleNotFoundError as e:
    import glob
    lib_dir = glob.glob(os.path.join(BASE_DIR, "build", "lib*"))[0]

    #print(f"Module not found, adding {lib_dir} to path")
    sys.path.append(lib_dir)
    import prod_quan_nn

import functions



class BaseNN:
    def __init__(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain.astype(int)

    def get_neighbors(self, xtest, k=1):
        """
        Returns a pair of (N x k) matrices, the first containing the indices
        of the k nearest neighbors, the second containing the Euclidean
        distance to those neighbors.

        This should be implemented by the subclass.
        """
        raise NotImplementedError()

    def classify(self, xtest, k=1):
        """
        Returns the majority label out of the k nearest neighbors.
        """
        indices, distances = self.get_neighbors(xtest, k)
        labels = self.ytrain[indices]
        return mode(labels, axis=1).mode.ravel()

    def timed_classify(self, xtest, k=1):
        """
        Same as `classify`, but also returns time it took to classify the given
        examples.
        """
        t = timeit.default_timer()
        labels = self.classify(xtest, k)
        t = timeit.default_timer() - t
        return labels, t

class NumpyNN(BaseNN):

    def __init__(self, xtrain, ytrain):
        super().__init__(xtrain, ytrain)

    def get_neighbors(self, xtest, k=1):
        return functions.numpy_nn_get_neighbors(self.xtrain, xtest, k)

class SklearnNN(BaseNN):
    def __init__(self, xtrain, ytrain):
        super().__init__(xtrain, ytrain)
        self.nn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="euclidean")
        self.nn.fit(self.xtrain)

    def get_neighbors(self, xtest, k=1):
        #Returns nearest neighbors dist and indices
        dist, nbrs = self.nn.kneighbors(xtest, k, return_distance=True)
        return nbrs, dist

class ProdQuanNN(BaseNN):
    def _fit_kmeans(dataset, xtrain, nclusters):
        kmeans = MiniBatchKMeans(
                n_clusters=nclusters,
                random_state=0,
                batch_size=10_000,
                verbose=VERBOSE)
        kmeans.fit(xtrain)
        return kmeans

    def _build_quantization(dataset, xtrain, npartitions, nclusters):
        nexamples, nfeatures = xtrain.shape
        step = int(np.ceil(nfeatures/npartitions))
        # partition_boundaries = np.minimum(nfeatures,
        #         np.arange(0, nfeatures+step-1, step))
        # The TA suggests the following replacement to also test partitions containing only 1 feature for exmaple:
        partition_boundaries = np.round(np.linspace(0, nfeatures, npartitions+1).astype(int))

        partitions = []
        print("BUILDING PRODUCT QUANTIZATION PARTITIONS")
        for i in range(npartitions):
            begin, end = partition_boundaries[i:i+2]
            print(f"FITTING PARTITION {i} using features {begin}:{end}")
            kmeans = ProdQuanNN._fit_kmeans(dataset, xtrain[:, begin:end],
                    nclusters)

            partitions.append({
                "begin": begin,
                "end": end,
                "kmeans": kmeans,
            })
        return partitions

    def _get_pqnn(dataset, xtrain, npartitions, nclusters, cache_partitions):
        cache_path = os.path.join(BASE_DIR, "data",
                f"{dataset}_partitions_cache.gz")
        if cache_partitions and os.path.exists(cache_path):
            print(f"loading cached partitions for {dataset} from")
            print(f"    {cache_path}")
            partitions = joblib.load(cache_path)
        else:
            partitions = ProdQuanNN._build_quantization(dataset, xtrain,
                    npartitions, nclusters)
            if cache_partitions:
                print(f"caching partitions in")
                print(f"    {cache_path}")
                joblib.dump(partitions, cache_path, compress=9)
        return prod_quan_nn.ProdQuanNN(partitions)

    def _get_nn(dataset, xtrain):
        class fake_kmeans:
            def __init__(self, cs, ls):
                self.cluster_centers_ = cs
                self.labels_ = ls
        kmeans = fake_kmeans(xtrain, np.arange(xtrain.shape[0], dtype=np.int32))
        return prod_quan_nn.ProdQuanNN([{
            "begin": 0,
            "end": xtrain.shape[1],
            "kmeans": kmeans }])

    def __init__(self, dataset_name, xtrain, ytrain, npartitions, nclusters,
            cache_partitions=False):
        super().__init__(xtrain, ytrain)

        self.dataset_name = dataset_name
        self.npartitions = npartitions
        self.nclusters = nclusters
        self.cache_partitions = cache_partitions

        if nclusters is not None:
            self.pqnn = ProdQuanNN._get_pqnn(dataset_name, xtrain, npartitions,
                    nclusters, cache_partitions)
        else:
            self.pqnn = ProdQuanNN._get_nn(dataset_name, xtrain)

        self.pqnn.initialize_method()

    def get_neighbors(self, xtest, k=1):
        return self.pqnn.compute_nearest_neighbors(xtest, k)
