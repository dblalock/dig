#!/usr/bin/env python

import numpy as np
from sklearn import cluster
import kmc2  # state-of-the-art kmeans initialization (as of NIPS 2016)

from joblib import Memory
_memory = Memory('.', verbose=0)


def dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=-1)


def dists_l1(X, q):
    diffs = np.abs(X - q)
    return np.sum(diffs, axis=-1)


def top_k_idxs(elements, k, smaller_better=True):
    if smaller_better:  # return indices of lowest elements
        which_nn = np.arange(k)
        return np.argpartition(elements, kth=which_nn)[:k]
    else:  # return indices of highest elements
        which_nn = len(elements) - 1 - np.arange(k)
        return np.argpartition(elements, kth=which_nn)[-k:][::-1]


def knn(X, q, k, dist_func=dists_sq):
    dists = dist_func(X, q)
    idxs = top_k_idxs(dists, k)
    return idxs, dists[idxs]


@_memory.cache
def kmeans(X, k, max_iter=16):
    seeds = kmc2.kmc2(X, k)
    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=max_iter).fit(X)
    return estimator.cluster_centers_, estimator.labels_


def orthonormalize_rows(A):
    Q, R = np.linalg.qr(A.T)
    return Q.T


def hamming_dist(v1, v2):
    return np.count_nonzero(v1 != v2)


def hamming_dists(X, q):
    return np.array([hamming_dist(row, q) for row in X])
