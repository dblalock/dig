#!/usr/bin/env python

import numpy as np

from utils import kmeans, orthonormalize_rows

from joblib import Memory
_memory = Memory('.', verbose=0)


# ================================================================ PQ

def learn_pq(X, ncentroids, nsubvects, subvect_len, max_kmeans_iters=16):
    codebooks = np.empty((ncentroids, nsubvects, subvect_len))
    assignments = np.empty((X.shape[0], nsubvects), dtype=np.int)

    # print "codebooks shape: ", codebooks.shape

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


# ================================================================ Gaussian OPQ

# https://github.com/yahoo/lopq/blob/master/python/lopq/model.py; see
# https://github.com/yahoo/lopq/blob/master/LICENSE. For this function only:
#
# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0.
# See the LICENSE file associated with the project for terms.
#
def eigenvalue_allocation(num_buckets, eigenvalues):
    """
    Compute a permutation of eigenvalues to balance variance accross buckets
    of dimensions.
    Described in section 3.2.4 in http://research.microsoft.com/pubs/187499/cvpr13opq.pdf
    Note, the following slides indicate this function will break when fed eigenvalues < 1
    without the scaling trick implemented below:
        https://www.robots.ox.ac.uk/~vgg/rg/slides/ge__cvpr2013__optimizedpq.pdf
    :param int num_buckets:
        the number of dimension buckets over which to allocate eigenvalues
    :param ndarray eigenvalues:
        a vector of eigenvalues
    :returns ndarray:
        a vector of indices by which to permute the eigenvectors
    """
    D = len(eigenvalues)
    dims_per_bucket = D / num_buckets
    eigenvalue_product = np.zeros(num_buckets, dtype=float)
    bucket_size = np.zeros(num_buckets, dtype=int)
    permutation = np.zeros((num_buckets, dims_per_bucket), dtype=int)

    # We first must scale the eigenvalues by dividing by their
    # smallets non-zero value to avoid problems with the algorithm
    # when eigenvalues are less than 1.
    min_non_zero_eigenvalue = np.min(np.abs(eigenvalues[np.nonzero(eigenvalues)]))
    eigenvalues = eigenvalues / min_non_zero_eigenvalue

    # Iterate eigenvalues in descending order
    sorted_inds = np.argsort(eigenvalues)[::-1]
    log_eigs = np.log2(abs(eigenvalues))
    for ind in sorted_inds:

        # Find eligible (not full) buckets
        eligible = (bucket_size < dims_per_bucket).nonzero()

        # Find eligible bucket with least eigenvalue product
        i = eigenvalue_product[eligible].argmin(0)
        bucket = eligible[0][i]

        # Update eigenvalue product for this bucket
        eigenvalue_product[bucket] = eigenvalue_product[bucket] + log_eigs[ind]

        # Store bucket assignment and update size
        permutation[bucket, bucket_size[bucket]] = ind
        bucket_size[bucket] += 1

    return np.reshape(permutation, D)


def learn_opq_gaussian_rotation(X_train, ncodebooks, codebook_bits):
    means = np.mean(X_train, axis=0)
    # X = X_train - means
    cov = np.dot(X_train.T, X_train) - np.outer(means, means)
    eigenvals, eigenvects = np.linalg.eigh(cov)

    # tmp = np.dot(eigenvects.T, eigenvects)
    # I = np.identity(X_train.shape[1], dtype=np.float32)
    # assert np.max(np.abs(tmp - I)) < .001

    # return np.identity(X_train.shape[1], dtype=np.float32)
    # return np.identity(X_train.shape[1])
    # return eigenvects  # TODO rm after debug

    order_idxs = eigenvalue_allocation(ncodebooks, eigenvals)
    assert len(order_idxs) == X_train.shape[1]
    # print order_idxs
    return eigenvects[:, order_idxs].T  # rows are projections


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


def opq_rotate(X, R):  # so other code need not know what to transpose
    return np.dot(np.atleast_2d(X), R.T)


# loosely based on:
# https://github.com/arbabenko/Quantizations/blob/master/opqCoding.py
# @_memory.cache
def learn_opq(X_train, ncodebooks, codebook_bits=8, niters=20,
              initial_kmeans_iters=1, init='gauss', debug=False):
    """init in {'gauss', 'identity', 'random'}"""

    X = X_train.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks
    # codebook_indices = np.arange(ncodebooks, dtype=np.int)

    normalize_mse_by = np.var(X)

    assert D % subvect_len == 0  # equal number of dims for each codebook

    if init == 'gauss':
        R = learn_opq_gaussian_rotation(X_train, ncodebooks, codebook_bits)
        R = R.astype(np.float32)
        X_rotated = opq_rotate(X, R)
        # _debug_rotation(R)
        # assert X.shape[1] == R.shape[0]
        # assert X.shape[1] == R.shape[1]
        # assert X.shape == X_rotated.shape
        # norms = np.linalg.norm(X, axis=1)
        # norms_rot = np.linalg.norm(X_rotated, axis=1)
        # print "orig norms, rotated norms: ", norms, norms_rot
        # assert np.max(np.abs(norms - norms_rot)) < .01
    elif init == 'identity':
        R = np.identity(D, dtype=np.float32)  # D x D
        X_rotated = X
    elif init == 'random':
        R = np.random.randn(D, D).astype(np.float32)
        R = orthonormalize_rows(R)
        X_rotated = opq_rotate(X, R)
    else:
        raise ValueError("Unrecognized initialization method: ".format(init))
        # X_rotated = opq_rotate(X, R)

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
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

    # prev_err = np.inf
    # prev_codebooks = None
    # prev_assignments = None
    for it in np.arange(niters):
        # compute reconstruction errors
        X_hat = reconstruct_X_pq(assignments, codebooks)
        errors = X_rotated - X_hat
        err = np.mean(errors * errors) / normalize_mse_by
        print "---- OPQ iter {}: mse / variance = {}".format(it, err)

        # if err > prev_err:
        #     print "WARNING: OPQ began to diverge"
        #     try:
        #         _debug_rotation(R)
        #     except NumericalException:
        #         pass
        #     codebooks = prev_codebooks
        #     assignments = prev_assignments
        #     break  # computation is diverging
        # prev_err = err

        # update rotation matrix based on reconstruction errors
        U, s, V = np.linalg.svd(np.dot(X_hat.T, X), full_matrices=False)
        R = np.dot(U, V)

        # if debug:
        #     try:
        #         _debug_rotation(R)
        #     except NumericalException as e:
        #         raise e

        # update centroids using new rotation matrix
        # prev_codebooks = codebooks
        # prev_assignments = assignments
        X_rotated = opq_rotate(X, R)
        assignments = _encode_X_pq(X_rotated, codebooks)
        # X_hat = reconstruct_X_pq(assignments, codebooks)
        # errors = X_rotated - X_hat
        # err = np.mean(errors * errors) / normalize_mse_by
        # print "---- part b OPQ iter {}: mse / variance = {}".format(it, err)
        codebooks = _update_centroids_opq(X_rotated, assignments, ncentroids)

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
        # datasets.Glove.TEST_100, N=10000, D=32)
        datasets.Glove.TEST_100, N=50000, D=96)
        # datasets.Sift1M.TEST_100, N=10000, D=32)
        # datasets.Gist.TEST_100, N=50000)
        # datasets.Glove.TEST_100, D=96)
        # datasets.Random.BLOBS, N=10000, D=32)

    # print X_train.shape

    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=0)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=2)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=20, debug=True)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=20)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=8, niters=20)

    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=0,
    #                                       initial_kmeans_iters=16)

    # in terms of reconstruction err, gaussian < identity < random
    niters = 5
    codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='random')
    codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='identity')
    codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='gauss')


if __name__ == '__main__':
    main()
