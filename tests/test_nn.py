#!/bin/env python

print("------------------------ running MatmulIndex tests")

import time
import numpy as np
import dig


def all_eq(x, y):
    if (len(x) == 0) and (len(y) == 0):
        return True
    return np.max(np.abs(x - y)) < .001


def test_index(idx_func=dig.MatmulIndex):
    N = 100 * 1000
    D = 100
    r2 = 1. * D

    # ------------------------ data and index construction

    X = np.random.randn(N, D)
    X = np.cumsum(X, axis=1)

    q = np.cumsum(np.random.randn(D))

    index = idx_func(X)
    t = index.getIndexConstructionTimeMs()
    print "index time: {}".format(t)

    # ------------------------ dists for ground truth

    t0 = time.clock()
    diffs = X - q
    trueDists = np.sum(diffs * diffs, axis=1)
    t_python = (time.clock() - t0) * 1000
    print "-> brute force time\t= {}".format(t_python)

    # ------------------------------------------------ single query

    # ------------------------ range query

    neighborIdxs = index.radius(q, np.sqrt(r2))
    t = index.getQueryTimeMs()
    neighborIdxs = np.sort(neighborIdxs)

    trueNeighborIdxs = np.where(trueDists <= r2)[0]

    if len(trueNeighborIdxs < 100):
        print "neighborIdxs: ", neighborIdxs
        print "trueNeighborIdxs: ", trueNeighborIdxs
    print "-> range query time\t= {}".format(t)

    assert(len(neighborIdxs) == len(trueNeighborIdxs))
    assert(all_eq(neighborIdxs, trueNeighborIdxs))

    # # ------------------------ knn query

    nn10 = index.knn(q, 10)
    t = index.getQueryTimeMs()
    trueNN10 = np.argsort(trueDists)[:10]
    # print "nn10, true nn10 = {}, {}".format(nn10, trueNN10)
    print "-> knn time\t\t= {}".format(t)
    assert(all_eq(nn10, trueNN10))

    # ------------------------------------------------ batch of queries

    # ------------------------ dists for ground truth

    Q = 4
    queries = np.random.randn(Q, D)
    queries = np.cumsum(queries, axis=1)

    # compute idxs in sorted order
    rowNorms = np.sum(X * X, axis=1, keepdims=True)
    t0 = time.clock()
    queryNorms = np.sum(queries * queries, axis=1)
    dotProds = np.dot(X, queries.T)
    dists = (-2 * dotProds) + rowNorms + queryNorms
    t_python = (time.clock() - t0) * 1000
    print "-> python batch brute force time ms\t= {:03} ({:.3}/query)".format(
        t_python, t_python / Q)

    # ------------------------ knn query

    k = 5
    idxs_sorted = np.argsort(dists, axis=0)
    true_knn = idxs_sorted[0:k, :].T

    nested_neighbors = index.knn_batch(queries, k)
    assert(all_eq(nested_neighbors, true_knn))
    t_cpp = index.getQueryTimeMs()
    print "-> cpp batch brute force time ms\t= {:03} ({:.3}/query)".format(
        t_cpp, t_cpp / Q)

    # ------------------------ radius

    # true_knn = np.where(dists < dists.mean())
    # print true_knn[:2]

    nested_neighbors = index.radius_batch(queries, r2)
    print nested_neighbors


if __name__ == '__main__':
    test_index()
