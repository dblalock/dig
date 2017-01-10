#!/bin/env python

# import functools
import os
import numpy as np

from joblib import Memory
_memory = Memory('.', verbose=1)


class Datasets:
    RAND_UNIF = 0
    RAND_GAUSS = 1
    RAND_WALK = 2
    GLOVE = 10
    GLOVE_100 = 11
    GLOVE_200 = 12
    SIFT = 20
    SIFT_100 = 21
    SIFT_200 = 22

    RAND_DATASETS = [RAND_UNIF, RAND_GAUSS, RAND_WALK]
    FILE_DATASETS = [GLOVE, SIFT, GLOVE_100, SIFT_100, GLOVE_200, SIFT_200]


class Paths:
    DATA_DIR = os.path.expanduser('~/Desktop/datasets/nn-search')
    GLOVE = os.path.join(DATA_DIR, 'glove.txt')
    GLOVE_100 = os.path.join(DATA_DIR, 'glove_100k.txt')
    GLOVE_200 = os.path.join(DATA_DIR, 'glove_200k.txt')
    SIFT = os.path.join(DATA_DIR, 'sift.txt')
    SIFT_100 = os.path.join(DATA_DIR, 'sift_100k.txt')
    SIFT_200 = os.path.join(DATA_DIR, 'sift_200k.txt')


@_memory.cache  # cache this more efficiently than as text
def cached_load_txt(*args, **kwargs):
    return np.loadtxt(*args, **kwargs)


def extract_random_rows(X, how_many):
    which_rows = np.random.randint(len(X), size=how_many)
    rows = np.copy(X[which_rows])
    mask = np.ones(len(X), dtype=np.bool)
    mask[which_rows] = False
    return X[mask], rows


def load_dataset(which_dataset, N=-1, D=-1, norm_mean=False, norm_len=False,
                 num_queries=1):
    if which_dataset == Datasets.RAND_UNIF:
        X = np.random.rand(N, D)
        q = np.random.rand(num_queries, D)
    elif which_dataset == Datasets.RAND_GAUSS:
        X = np.random.randn(N, D)
        q = np.random.randn(num_queries, D)
    elif which_dataset == Datasets.RAND_WALK:
        X = np.random.randn(N, D)
        X = np.cumsum(X, axis=1)
        q = np.random.randn(num_queries, D)
        q = np.cumsum(q, axis=-1)
    elif which_dataset == Datasets.GLOVE:
        X = cached_load_txt(Paths.GLOVE)
    elif which_dataset == Datasets.SIFT:
        X = cached_load_txt(Paths.SIFT)
    elif which_dataset == Datasets.GLOVE_100:
        X = cached_load_txt(Paths.GLOVE_100)
    elif which_dataset == Datasets.SIFT_100:
        X = cached_load_txt(Paths.SIFT_100)
    elif which_dataset == Datasets.GLOVE_200:
        X = cached_load_txt(Paths.GLOVE_200)
    elif which_dataset == Datasets.SIFT_200:
        X = cached_load_txt(Paths.SIFT_200)
    else:
        raise ValueError("unrecognized dataset {}".format(which_dataset))

    if N > 0 and N < X.shape[0]:
        X = X[:N, :]

    if which_dataset in Datasets.FILE_DATASETS:
        X, q = extract_random_rows(X, how_many=num_queries)

    # if which_dataset in Datasets.FILE_DATASETS:
    #     if N > 0 and N < X.shape[0]:
    #         X = X[:N, :]
    #         X, q = extract_random_rows
    #     else:
    #         X = X[:-1, :]
    #         # q = np.copy(X[-1])
    #         # q = np.random.randn(X.shape[1])
    #         idx = np.random.randint(X.shape[1] - 1)
    #         q = np.copy(X[idx])
    #         q[64:] = X[idx + 1, 64:]  # mix 2 examples so differs from both
    #         print "start of q: ", q[:10]

    if norm_mean:
        means = np.mean(X, axis=0)
        X -= means
        q -= means
    if norm_len:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)

    return X.astype(np.float32), np.squeeze(q.astype(np.float32))
