#!/usr/bin/env python

import numpy as np
from sklearn import cluster
import kmc2  # state-of-the-art kmeans initialization (as of NIPS 2016)

from joblib import Memory
_memory = Memory('.', verbose=1)


# ================================================================ Utils

def dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=(len(X.shape) - 1))  # handle 1D X


def top_k_idxs(elements, k, smaller_better=True):
    if smaller_better:  # return indices of lowest elements
        which_nn = np.arange(k)
        return np.argpartition(elements, kth=which_nn)[:k]
    else:  # return indices of highest elements
        which_nn = len(elements) - 1 - np.arange(k)
        return np.argpartition(elements, kth=which_nn)[-k:][::-1]


def knn(X, q, k):
    dists = dists_sq(X, q)
    return top_k_idxs(dists, k)


@_memory.cache
def kmeans(X, k, max_iter=16):
    seeds = kmc2.kmc2(X, k)
    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=max_iter).fit(X)
    return estimator.cluster_centers_, estimator.labels_


# def orthogonalize_rows(A):
#     Q, R = np.linalg.qr(A.T)
#     return Q.T


# ================================================================ PQ

def learn_pq(X, ncentroids, nsubvects, subvect_len, max_kmeans_iters=16):
    codebooks = np.empty((ncentroids, nsubvects, subvect_len))
    assignments = np.empty((X.shape[0], nsubvects), dtype=np.int)

    print "codebooks shape: ", codebooks.shape

    for i in range(nsubvects):
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        centroids, labels = kmeans(X_in, ncentroids, max_iter=max_kmeans_iters)
        codebooks[:, i, :] = centroids
        assignments[:, i] = labels

    return codebooks, assignments  # [2**nbits x M x D/M], [N x M]


def reconstruct_X_pq(assignments, codebooks):
    """assignments: N x M ints; codebooks: 2**nbits x M x D/M floats"""
    _, M = assignments.shape
    subvect_len = codebooks.shape[2]

    assert assignments.shape[1] == codebooks.shape[1]

    D = M * subvect_len
    pointsCount = assignments.shape[0]
    points = np.zeros((pointsCount, D), dtype=np.float32)
    for i in xrange(M):
        subspace_start = subvect_len * i
        subspace_end = subspace_start + subvect_len
        subspace_codes = assignments[:, i]
        points[:, subspace_start:subspace_end] = codebooks[subspace_codes, i, :]
    return points


def _dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def _dists_elemwise_l1(x, q):
    return np.abs(x - q)


def _encode_X_pq(X, codebooks, elemwise_dist_func=_dists_elemwise_sq):
    ncentroids, nsubvects, subvect_len = codebooks.shape

    assert X.shape[1] == (nsubvects * subvect_len)

    idxs = np.empty((X.shape[0], nsubvects), dtype=np.int)
    X = X.reshape((X.shape[0], nsubvects, subvect_len))
    for i, row in enumerate(X):
        row = row.reshape((1, nsubvects, subvect_len))
        dists = elemwise_dist_func(codebooks, row)
        dists = np.sum(dists, axis=2)
        idxs[i, :] = np.argmin(dists, axis=0)

    # return idxs + self.offsets  # offsets let us index into raveled dists
    return idxs  # [N x nsubvects]


# ================================================================ OPQ

def _update_centroids_opq(X, assignments, ncentroids):  # [N x D], [N x M]
    nsubvects = assignments.shape[1]
    subvect_len = X.shape[1] // nsubvects

    assert X.shape[0] == assignments.shape[0]
    assert X.shape[1] % nsubvects == 0

    codebooks = np.zeros((ncentroids, nsubvects, subvect_len), dtype=np.float32)
    for i, row in enumerate(X):
        for m in range(nsubvects):
            start_col = m * subvect_len
            end_col = start_col + subvect_len
            codebooks[assignments[i, m], m, :] += row[start_col:end_col]

    for m in range(nsubvects):
        code_counts = np.bincount(assignments[:, m]).reshape((-1, 1))
        codebooks[:, m] /= np.maximum(code_counts, 1)  # preclude div by 0

    return codebooks


class NumericalException(Exception):
    pass


def _debug_rotation(R):
    D = np.max(R.shape)
    identity = np.identity(D, dtype=np.float32)
    RtR = np.dot(R.T, R)

    R_det = np.linalg.det(RtR)
    print "determinant of R*R: ", R_det
    R_trace = np.trace(RtR)
    print "trace of R*R, trace divided by D: {}, {}".format(R_trace, R_trace / D)
    off_diagonal_abs_mean = np.mean(np.abs(RtR - identity))
    print "mean(abs(off diagonals of R*R)): ", off_diagonal_abs_mean

    if R_det < .999 or R_det > 1.001:
        raise NumericalException("Bad determinant")
    if R_trace < .999 * D or R_trace > 1.001 * D:
        raise NumericalException("Bad trace")
    if off_diagonal_abs_mean > .001:
        raise NumericalException("Bad off-diagonals")


# based on https://github.com/arbabenko/Quantizations/blob/master/opqCoding.py
def learn_opq(X_train, ncodebooks, codebook_bits=8, niters=20,
              initial_kmeans_iters=1, debug=False):

    X = X_train.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks
    # codebook_indices = np.arange(ncodebooks, dtype=np.int)

    normalize_mse_by = np.var(X)

    assert D % subvect_len == 0  # equal number of dims for each codebook

    R = np.identity(D, dtype=np.float32)  # D x D
    # X_rotated = np.dot(X, R.T)  # (N x D) * (D x D) = N x D

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
    X_rotated = X
    codebooks, assignments = learn_pq(X_rotated, ncentroids=ncentroids,
                                      nsubvects=ncodebooks,
                                      subvect_len=subvect_len,
                                      max_kmeans_iters=1)
    # alternative: initialize codebooks by sampling randomly from the data
    # codebooks = np.zeros((ncodebooks, ncentroids, subvect_len))
    # all_idxs = np.arange(N, dtype=np.int)
    # for m in np.arange(ncodebooks):
    #     rand_idxs = np.random.choice(all_idxs, size=ncentroids, replace=False)
    #     start_col = subvect_len * i
    #     end_col = start_col + subvect_len
    #     codebooks[:, m, :] = X_rotated[rand_idxs, start_col:end_col]

    prev_err = np.inf
    prev_codebooks = None
    prev_assignments = None
    for it in np.arange(niters):
        # compute reconstruction errors
        X_hat = reconstruct_X_pq(assignments, codebooks)
        errors = X_rotated - X_hat
        err = np.mean(errors * errors) / normalize_mse_by
        print "---- OPQ iter {}: mse / variance = {}".format(it, err)

        if err > prev_err:
            print "WARNING: OPQ began to diverge"
            try:
                _debug_rotation(R)
            except NumericalException:
                pass
            codebooks = prev_codebooks
            assignments = prev_assignments
            break  # computation is diverging
        prev_err = err

        # update rotation matrix based on reconstruction errors
        U, s, V = np.linalg.svd(np.dot(X_hat.T, X), full_matrices=False)
        R = np.dot(U, V)

        if debug:
            try:
                _debug_rotation(R)
            except NumericalException as e:
                raise e

        # update centroids using new rotation matrix
        X_rotated = np.dot(X, R.T)
        prev_codebooks = codebooks
        prev_assignments = assignments
        codebooks = _update_centroids_opq(X, assignments, ncentroids)
        assignments = _encode_X_pq(X_rotated, codebooks)

    X_hat = reconstruct_X_pq(assignments, codebooks)
    errors = X_rotated - X_hat
    err = np.mean(errors * errors) / normalize_mse_by
    print "---- OPQ final mse / variance = {}".format(err)

    return codebooks, assignments, R


def main():
    import datasets
    # tmp = datasets.load_dataset(
    X_train, Q, X_test, truth = datasets.load_dataset(
        # datasets.Random.UNIFORM, N=1000, D=64)
        # datasets.Glove.TEST_100, N=10000)
        # datasets.Glove.TEST_100, N=100000)
        # datasets.Sift1M.TEST_100, N=100000)
        # datasets.Gist.TEST_100, N=50000)
        # datasets.Glove.TEST_100, D=96)
        datasets.Random.BLOBS, N=10000, D=96)

    # print X_train.shape

    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=0)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=2)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=20, debug=True)
    codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=20)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=8, niters=20)

    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=0,
    #                                       initial_kmeans_iters=16)



if __name__ == '__main__':
    main()
