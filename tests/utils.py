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


def sq_dists_to_vectors(X, queries, rowNorms=None, queryNorms=None):
    Q = queries.shape[0]

    if rowNorms is None:
        rowNorms = np.sum(X * X, axis=1, keepdims=True)

    # t0 = time.clock()

    if queryNorms is None:
        queryNorms = np.sum(queries * queries, axis=1)

    dotProds = np.dot(X, queries.T)
    return (-2 * dotProds) + rowNorms + queryNorms

    # t_python = (time.clock() - t0) * 1000
    # print "-> python batch{} knn time\t= {:03} ({:.3}/query)".format(
    #     Q, t_python, t_python / Q)

    # idxs_sorted = np.argsort(dists, axis=0)
    # return dists, idxs_sorted, t_python


def all_eq(x, y):
    if len(x) != len(y):
        return False
    if len(x) == 0:
        return True
    return np.max(np.abs(x - y)) < .001


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
def kmeans(X, k, max_iter=16, init='kmc2'):
    X = X.astype(np.float32)

    # if k is huge, initialize centers with cartesian product of centroids
    # in two subspaces
    sqrt_k = int(np.sqrt(k) + .5)
    if k > 256 and sqrt_k ** 2 == k and init == 'subspaces':
        print "kmeans: clustering in subspaces first; k, sqrt(k) =" \
            " {}, {}".format(k, sqrt_k)
        _, D = X.shape
        centroids0, _ = kmeans(X[:, :D/2], sqrt_k, max_iter=1)
        centroids1, _ = kmeans(X[:, D/2:], sqrt_k, max_iter=1)
        seeds = np.empty((k, D), dtype=np.float32)
        for i in range(sqrt_k):
            for j in range(sqrt_k):
                row = i * sqrt_k + j
                seeds[row, :D/2] = centroids0[i]
                seeds[row, D/2:] = centroids1[j]
    elif init == 'kmc2':
        seeds = kmc2.kmc2(X, k).astype(np.float32)
    else:
        raise ValueError("init parameter must be one of {'kmc2', 'subspaces'}")

    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=max_iter).fit(X)
    return estimator.cluster_centers_, estimator.labels_


def orthonormalize_rows(A):
    Q, R = np.linalg.qr(A.T)
    return Q.T


def random_rotation(D):
    rows = np.random.randn(D, D)
    return orthonormalize_rows(rows)


def hamming_dist(v1, v2):
    return np.count_nonzero(v1 != v2)


def hamming_dists(X, q):
    return np.array([hamming_dist(row, q) for row in X])
