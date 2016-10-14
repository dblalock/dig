#!/bin/env python

print("------------------------ running MatmulIndex tests")

import time
import numpy as np
import dig


def all_eq(x, y):
    if (len(x) == 0) and (len(y) == 0):
        return True
    return np.max(np.abs(x - y)) < .001


def sq_dists_to_vectors(X, queries, rowNorms=None, queryNorms=None):
    Q = queries.shape[0]

    if rowNorms is None:
        rowNorms = np.sum(X * X, axis=1, keepdims=True)

    t0 = time.clock()

    if queryNorms is None:
        queryNorms = np.sum(queries * queries, axis=1)

    dotProds = np.dot(X, queries.T)
    dists = (-2 * dotProds) + rowNorms + queryNorms

    t_python = (time.clock() - t0) * 1000
    print "-> python batch{} knn time\t= {:03} ({:.3}/query)".format(
        Q, t_python, t_python / Q)

    idxs_sorted = np.argsort(dists, axis=0)
    return dists, idxs_sorted, t_python


def test_knn_batch_query(index, queries, idxs_sorted, k, name):
    Q = queries.shape[0]

    nested_neighbors = index.knn_batch(queries, k)
    true_knn = idxs_sorted[0:k, :].T
    assert(all_eq(nested_neighbors, true_knn))

    t_cpp = index.getQueryTimeMs()
    print "-> {} batch{} {}nn time \t= {:03} ({:.3}/query)".format(
        name, Q, k, t_cpp, t_cpp / Q)

    return t_cpp


def test_radius_batch_query(index, queries, dists, name, r2):
    nested_neighbors = index.radius_batch(queries, r2)

    for q in range(queries.shape[0]):
        # pull out idxs > 0 for this query in full dist mat
        query_dists = dists[:, q]
        true_idxs = np.where(query_dists < r2)[0]

        # find idxs > -1 in row for this query
        returned_idxs = nested_neighbors[q]
        where_valid = np.where(returned_idxs >= 0)[0]
        if len(where_valid):
            returned_idxs = returned_idxs[where_valid]
        else:
            returned_idxs = np.array([])

        # print "true_idxs:", true_idxs
        # print "returned_idxs:", returned_idxs
        assert(all_eq(returned_idxs, true_idxs))


def test_index(idx_func=dig.MatmulIndex, name="cpp", N=100, D=100, r2=-1,
               dtype=np.float64):
    N = 100 * 1000
    D = 100
    if r2 <= 0:
        r2 = 1. * D

    # ------------------------ data and index construction

    X = np.random.randn(N, D).astype(dtype)
    X = np.cumsum(X, axis=1)

    q = np.cumsum(np.random.randn(D).astype(dtype))

    index = idx_func(X)
    t = index.getIndexConstructionTimeMs()
    print "{} index time: {}".format(name, t)

    # ------------------------ dists for ground truth

    t0 = time.clock()
    diffs = X - q
    trueDists = np.sum(diffs * diffs, axis=1)
    t_python = (time.clock() - t0) * 1000
    print "-> python full dists time\t= {}".format(t_python)

    # ------------------------------------------------ single query

    # ------------------------ range query

    neighborIdxs = index.radius(q, np.sqrt(r2))
    t = index.getQueryTimeMs()
    neighborIdxs = np.sort(neighborIdxs)

    trueNeighborIdxs = np.where(trueDists <= r2)[0]

    if len(trueNeighborIdxs < 100):
        print "neighborIdxs: ", neighborIdxs
        print "trueNeighborIdxs: ", trueNeighborIdxs
    print "-> {} range query time\t= {}".format(name, t)

    assert(len(neighborIdxs) == len(trueNeighborIdxs))
    assert(all_eq(neighborIdxs, trueNeighborIdxs))

    # ------------------------ knn query

    nn10 = index.knn(q, 10)
    t = index.getQueryTimeMs()
    trueNN10 = np.argsort(trueDists)[:10]
    # print "nn10, true nn10 =\n{}\n{}".format(nn10, trueNN10)
    print "sorted nn10, true nn10 =\n{}\n{}".format(
        sorted(nn10), sorted(trueNN10))
    print "-> {} knn time\t\t= {}".format(name, t)
    assert(all_eq(nn10, trueNN10))

    # ------------------------------------------------ batch of queries

    # ------------------------ dists for ground truth

    Q = 128
    queries = np.random.randn(Q, D).astype(dtype)
    queries = np.cumsum(queries, axis=1)
    queries = np.copy(queries[:, ::-1])  # *should* help abandon a lot...

    dists, idxs_sorted, t_python = sq_dists_to_vectors(X, queries)

    # ------------------------ knn query

    test_knn_batch_query(index, queries, idxs_sorted, name=name, k=3)
    test_knn_batch_query(index, queries, idxs_sorted, name=name, k=10)

    # ------------------------ radius

    test_radius_batch_query(index, queries, dists, name=name, r2=r2)


def debug():
    N = 4
    D = 3

    dtype = np.float32

    # X = np.random.randn(N, D).astype(dtype)
    # X = np.cumsum(X, axis=1)
    X = np.arange(N * D, dtype=dtype).reshape((N, D))

    # q = np.cumsum(np.random.randn(D).astype(dtype))
    print "X py:\n", X
    index = dig.AbandonIndexF(X)

    q = np.arange(D, dtype=dtype)
    print "q py:\n", q
    index.radius(q, 99.)


if __name__ == '__main__':
    # debug()

    # test_index(dig.MatmulIndex, "matmul")
    # test_index(dig.MatmulIndexF, "matmulf", dtype=np.float32)

    # test_index(dig.AbandonIndex, "abandon")
    # test_index(dig.AbandonIndexF, "abandonf", dtype=np.float32)

    test_index(dig.SimpleIndex, "simple")
    test_index(dig.SimpleIndexF, "simplef", dtype=np.float32)
