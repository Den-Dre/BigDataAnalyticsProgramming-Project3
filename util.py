import numpy as np
from sklearn.model_selection import train_test_split

from nn import *

from csv import writer

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = "data" # relative to BASE_DIR, dir of this file

DATASETS = {
    "covtype"    : "covtype.gz",
    "emnist"     : "emnist.gz",
    "emnist_orig": "emnist_original.gz",
    "higgs"      : "higgs.gz",
    "spambase"   : "spambase.gz",
}

PROD_QUAN_SETTINGS = {
    "covtype"   : {
        "nclusters"   : 255,
        "npartitions" : 10,
    },
    "emnist"    : {
        "nclusters"   : 255,
        "npartitions" : 47,
    },
    "emnist_orig"    : {
        "nclusters"   : 255,
        "npartitions" : 28,
    },
    "higgs"     : {
        "nclusters"   : 255,
        "npartitions" : 5,
    },
    "spambase"  : {
        "nclusters"   : 8,
        "npartitions" : 4,
    },
}

def load_dataset(dataset):
    try:
        data_path = os.path.join(BASE_DIR, DATA_DIR, DATASETS[dataset])
    except KeyError as e:
        print(f"Invalid dataset name {dataset}", file=sys.stderr)
        sys.exit(1)
    try:
        print(f"Loading dataset {dataset} from")
        print(f"    {data_path}")
        x, y = joblib.load(data_path)
    except IOError:
        print("Data not found", file=sys.stderr)
        print(data_path, file=sys.stderr)
        sys.exit(2)

    x, y = x.astype(np.float32), y.astype(int)
    save_to_csv(dataset, x, y)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,
            random_state=0)

    return xtrain, xtest, ytrain, ytest

def save_to_csv(filename, x, y):
    print('Saving dataset...')
    with open(f'data/{filename}.csv', 'w') as f:
        for features, classif in zip(x, y):
            f.write(np.array2string(features, separator=',', max_line_width=1000) + ', ' + str(classif) + '\n')
    print('Done saving')


def sample_xtest(xtest, ytest, n, seed):
    rng = np.random.default_rng(seed)
    js = rng.choice(range(xtest.shape[0]), min(n, xtest.shape[0]))

    xsample = np.array(xtest[js, :])
    ysample = np.array(ytest[js])

    return xsample, ysample

def get_nn_instances(dataset, xtrain, ytrain,
        npartitions=None,
        nclusters=None,
        cache_partitions=False):

    # Use default settings for ProdQuanNN if none are given as arguments
    if npartitions is None:
        npartitions=PROD_QUAN_SETTINGS[dataset]["npartitions"]
    if nclusters is None:
        nclusters = PROD_QUAN_SETTINGS[dataset]["nclusters"]

    pqnn = ProdQuanNN(dataset, xtrain, ytrain,
            npartitions,
            nclusters,
            cache_partitions)
    npnn = NumpyNN(xtrain, ytrain)
    sknn = SklearnNN(xtrain, ytrain)

    return pqnn, npnn, sknn

def decode_emnist_label(label):
    # not all lower case letter are distuinguished from upper case letters
    # https://arxiv.org/pdf/1702.05373v1.pdf
    lowercase = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't' ]

    if label <= 9:
        return str(label)
    elif label <= 9+26:
        return chr(ord('A')+label-10)
    else:
        return lowercase[label-10-26]
