#!/bin/env python

# import functools
import os
import numpy as np

from joblib import Memory
_memory = Memory('.', verbose=1)

DATA_DIR = os.path.expanduser('~/Desktop/datasets/nn-search')
join = os.path.join


# class Datasets:
#     RAND_UNIF = 0
#     RAND_GAUSS = 1
#     RAND_WALK = 2
#     GLOVE = 10
#     GLOVE_100 = 11
#     GLOVE_200 = 12
#     SIFT = 20
#     SIFT_100 = 21
#     SIFT_200 = 22
#     GIST = 30
#     GIST_100 = 31
#     GIST_200 = 32
#     # GIST_QUERIES = 33

#     RAND_DATASETS = [RAND_UNIF, RAND_GAUSS, RAND_WALK]
#     FILE_DATASETS = [GLOVE, GLOVE_100, GLOVE_200,
#                      SIFT, SIFT_100, SIFT_200,
#                      GIST, GIST_100, GIST_200]


# RAND_DATASETS = [RAND_UNIF, RAND_GAUSS, RAND_WALK]
# FILE_DATASETS = [GLOVE, GLOVE_100, GLOVE_200,
#                  SIFT, SIFT_100, SIFT_200,
#                  GIST, GIST_100, GIST_200]


class Random:
    UNIFORM = 'uniform'
    GAUSS = 'gauss'
    WALK = 'walk'


class Gist:
    DIR = join(DATA_DIR, 'gist')
    TRAIN    = join(DIR, 'gist_train.npy')     # noqa
    TEST     = join(DIR, 'gist.npy')           # noqa
    TEST_100 = join(DIR, 'gist_100k.npy')      # noqa
    TEST_200 = join(DIR, 'gist_200k.npy')      # noqa
    QUERIES  = join(DIR, 'gist_queries.npy')   # noqa
    TRUTH    = join(DIR, 'gist_truth.npy')     # noqa


class Sift1M:
    DIR = join(DATA_DIR, 'sift1m')
    TRAIN    = join(DIR, 'sift_learn.npy')          # noqa
    TEST     = join(DIR, 'sift_base.npy')           # noqa
    TEST_100 = join(DIR, 'sift_100k.txt')           # noqa
    TEST_200 = join(DIR, 'sift_200k.txt')           # noqa
    QUERIES  = join(DIR, 'sift_queries.npy')        # noqa
    TRUTH    = join(DIR, 'sift_groundtruth.npy')    # noqa


class Sift10M:
    DIR = join(DATA_DIR, 'sift1b')
    # TRAIN    = join(DIR, 'big_ann_learn_10M.npy') # noqa
    TRAIN    = join(DIR, 'big_ann_learn_1M.npy')    # noqa  # TODO use 10M?
    TRAIN_1M = join(DIR, 'big_ann_learn_1M.npy')    # noqa
    TEST     = join(DIR, 'sift_10M.npy')            # noqa
    QUERIES  = join(DIR, 'sift_queries.npy')        # noqa
    TRUTH    = join(DIR, 'true_nn_idxs_10M.npy')    # noqa


class LabelMe:
    DIR = join(DATA_DIR, 'labelme')
    # TODO pull out train and test, write out as files, and precompute truth


class Glove:
    DIR = join(DATA_DIR, 'glove')
    TEST     = join(DIR, 'glove.txt')           # noqa
    TEST_100 = join(DIR, 'glove_100k.txt')      # noqa
    TEST_200 = join(DIR, 'glove_200k.txt')      # noqa


# @_memory.cache  # cache this more efficiently than as text
def cached_load_txt(*args, **kwargs):
    return np.loadtxt(*args, **kwargs)


def load_file(fname, *args, **kwargs):
    if fname.split('.')[-1] == 'txt':
        return np.loadtxt(fname, *args, **kwargs)
    return np.load(fname, *args, **kwargs)


def extract_random_rows(X, how_many):
    which_rows = np.random.randint(len(X), size=how_many)
    rows = np.copy(X[which_rows])
    mask = np.ones(len(X), dtype=np.bool)
    mask[which_rows] = False
    return X[mask], rows


def _load_complete_dataset(which_dataset, num_queries=1):
    X_test = np.load(which_dataset.TEST)
    try:
        X_train = np.load(which_dataset.TRAIN)
    except AttributeError:
        X_train = X_test
    try:
        Q = np.load(which_dataset.QUERIES)
    except AttributeError:
        X_train, Q = extract_random_rows(X_train, how_many=num_queries)
    try:
        true_nn = np.load(which_dataset.TRUTH)
    except AttributeError:
        true_nn = None

    return X_train, Q, X_test, true_nn


def _ground_truth_for_dataset(which_dataset):
    return None  # TODO


def load_dataset(which_dataset, N=-1, D=-1, norm_mean=False, norm_len=False,
                 num_queries=1, Ntrain=-1):
    true_nn = None

    # randomly generated datasets
    if which_dataset == Random.UNIFORM:
        X_test = np.random.rand(N, D)
        X_train = np.random.rand(Ntrain, D) if Ntrain > 0 else X_test
        Q = np.random.rand(num_queries, D)
    elif which_dataset == Random.GAUSS:
        X_test = np.random.randn(N, D)
        X_train = np.random.randn(Ntrain, D) if Ntrain > 0 else X_test
        Q = np.random.randn(num_queries, D)
    elif which_dataset == Random.WALK:
        X_test = np.random.randn(N, D)
        X_test = np.cumsum(X_test, axis=1)
        X_train = X_test
        if Ntrain > 0:
            X_train = np.random.randn(Ntrain, D)
            X_train = np.cumsum(X_train)
        Q = np.random.randn(num_queries, D)
        Q = np.cumsum(Q, axis=-1)

    # datasets that are just one block of a "real" dataset
    elif isinstance(which_dataset, str):
        X_test = load_file(which_dataset)
        X_test, Q = extract_random_rows(X_test, how_many=num_queries)
        X_train = X_test
        true_nn = _ground_truth_for_dataset(which_dataset)

    elif which_dataset in (Glove, Gist, Sift1M, Sift10M):
        X_train, Q, X_test, true_nn = _load_complete_dataset(which_dataset)


    # return _load_complete_dataset(which_dataset, N, D)
        # X_train =

    # elif which_dataset == Glove.TEST:
    #     X = cached_load(Glove.TEST)
    # elif which_dataset == Datasets.GLOVE_100:
    #     X = cached_load(Glove.TEST_100)
    # elif which_dataset == Datasets.GLOVE_200:
    #     X = cached_load(Glove.TEST_200)

    # elif which_dataset == Datasets.SIFT:
    #     X = cached_load(Sift.TEST)
    # elif which_dataset == Datasets.SIFT_100:
    #     X = cached_load(Sift.TEST_100)
    # elif which_dataset == Datasets.SIFT_200:
    #     X = cached_load(Sift.TEST_200)

    # elif which_dataset == Datasets.GIST:
    #     X = np.load(Gist.TEST)
    # elif which_dataset == Datasets.GIST_100:
    #     X = np.load(Gist.TEST_100)
    # elif which_dataset == Datasets.GIST_200:
    #     X = np.load(Gist.TEST_200)

    else:
        raise ValueError("unrecognized dataset {}".format(which_dataset))

    N = X_test.shape[0] if N < 1 else N
    D = X_test.shape[1] if D < 1 else D
    X_test, X_train = X_test[:N, :D], X_train[:N, :D]
    Q = Q[:, :D] if len(Q.shape) > 1 else Q[:D]

    if norm_mean:
        means = np.mean(X_train, axis=0)
        X_train -= means
        if X_train is not X_test:
            X_test -= means
        Q -= means
    if norm_len:
        X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
        if X_train is not X_test:
            X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=-1, keepdims=True)

    # TODO don't convert datasets that are originally uint8s to floats
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Q = np.squeeze(Q.astype(np.float32))

    return X_train, Q, X_test, true_nn


def read_yael_vecs(path, c_contiguous=True, dtype=np.float32, limit=-1):
    """note that this probably won't work unless dtype is 4 bytes"""
    fv = np.fromfile(path, dtype=dtype, count=limit)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    print "vector length = {}".format(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + path)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


if __name__ == '__main__':
    pass

    # Clean up gist data
    #
    # path = Paths.DATA_DIR + '/gist/gist_base.fvecs'
    # path = Paths.DATA_DIR + '/gist/gist_learn.fvecs'
    # path = Paths.DATA_DIR + '/gist/gist_queries.fvecs'
    # path = Paths.DATA_DIR + '/gist/gist_groundtruth.ivecs'
    # out_path = Paths.GIST_100
    # out_path = Paths.GIST_200
    # out_path = Paths.GIST_TRAIN
    # out_path = Paths.GIST_QUERIES
    # out_path = Paths.GIST_TRUTH
    # X = read_yael_vecs(path)[:100000]
    # X = read_yael_vecs(path)[:200000]
    # X = read_yael_vecs(path)
    # X = read_yael_vecs(path, dtype=np.int32)
    # print X[:2]
    # print X.shape
    # np.save(out_path, X)

    # clean up LabelMe
    #
    # >>> for k, v in d.iteritems():
    # ...     try:
    # ...             print k, v.shape
    # ...     except:
    # ...             pass
    # ...
    # gist (22019, 512)
    # img (32, 32, 3, 22019)
    # nmat (1000, 1000, 20)
    # __header__ param (1, 1)
    # __globals__ seg (32, 32, 22019)
    # names (1, 3597)
    # DistLM (22019, 22019)
    # __version__ ndxtrain (1, 20019)
    # ndxtest (1, 2000)
    #
    # okay, no idea what most of these are even with the readme...
    #
    # >>> np.save('labelme_train_idxs', d['ndxtrain']) # training data idxs
    # >>> np.save('labelme_test_idxs', d['ndxtest'])   # test data idxs
    # >>> np.save('labelme_all_gists', d['gist'])     # actual gist descriptors

