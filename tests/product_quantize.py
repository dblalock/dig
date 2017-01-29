#!/usr/bin/env python

import time
import numpy as np

from utils import kmeans, orthonormalize_rows, random_rotation
from utils import sq_dists_to_vectors

from joblib import Memory
_memory = Memory('.', verbose=0)


# ================================================================ PQ

@_memory.cache
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
@_memory.cache
def eigenvalue_allocation(num_buckets, eigenvalues, shuffle=False):
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
    :param bool shuffle:
        whether to randomly shuffle the order of resulting buckets
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

    # this is not actually a requirement, but I'm curious about whether this
    # condition is ever violated
    assert np.all(eigenvalues > 0)

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

    if shuffle:
        shuffle_idxs = np.arange(num_buckets, dtype=np.int)
        np.random.shuffle(shuffle_idxs)
        permutation = permutation[shuffle_idxs]

    # wow, these are within <1% of each other
    # print "opq eigenvalue log prods: ", eigenvalue_product

    return np.reshape(permutation, D)


def learn_opq_gaussian_rotation(X_train, ncodebooks, shuffle=False):
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

    print "shuffling buckets?", shuffle

    order_idxs = eigenvalue_allocation(ncodebooks, eigenvals, shuffle=shuffle)
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


def opq_undo_rotate(X, R):  # so other code need not know what to transpose
    return np.dot(np.atleast_2d(X), R)


# @_memory.cache
def opq_initialize(X_train, ncodebooks, init='gauss'):
    X = X_train
    _, D = X.shape

    if init == 'gauss' or init == 'gauss_flat' or init == 'gauss_shuffle':
        permute = (init == 'gauss_shuffle')
        R = learn_opq_gaussian_rotation(X_train, ncodebooks, shuffle=permute)
        R = R.astype(np.float32)

        if init == 'gauss_flat':
            # assert R.shape[0] == R.shape[1]
            D = R.shape[1]
            d = D / ncodebooks
            assert int(d) == d  # need same number of dims in each subspace
            # d = int(d)
            local_r = random_rotation(int(d))
            tiled = np.zeros((D, D))
            for c in range(ncodebooks):
                start = c * d
                end = start + d
                tiled[start:end, start:end] = local_r

            R = np.dot(R, tiled)
        # elif init == 'gauss_shuffle':
        #     order_idxs = np.arange(ncodebooks)
        #     np.shuffle(order_idxs)
        #     R =

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

    return X_rotated, R


# loosely based on:
# https://github.com/arbabenko/Quantizations/blob/master/opqCoding.py
# @_memory.cache
def learn_opq(X_train, ncodebooks, codebook_bits=8, niters=20,
              initial_kmeans_iters=1, init='gauss', debug=False):
    """init in {'gauss', 'identity', 'random'}"""

    t0 = time.time()

    X = X_train.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks
    # codebook_indices = np.arange(ncodebooks, dtype=np.int)

    normalize_mse_by = np.var(X)

    assert D % subvect_len == 0  # equal number of dims for each codebook

    X_rotated, R = opq_initialize(X_train, ncodebooks=ncodebooks, init=init)

    # if init == 'gauss':
    #     R = learn_opq_gaussian_rotation(X_train, ncodebooks, codebook_bits)
    #     R = R.astype(np.float32)
    #     X_rotated = opq_rotate(X, R)
    #     # _debug_rotation(R)
    #     # assert X.shape[1] == R.shape[0]
    #     # assert X.shape[1] == R.shape[1]
    #     # assert X.shape == X_rotated.shape
    #     # norms = np.linalg.norm(X, axis=1)
    #     # norms_rot = np.linalg.norm(X_rotated, axis=1)
    #     # print "orig norms, rotated norms: ", norms, norms_rot
    #     # assert np.max(np.abs(norms - norms_rot)) < .01
    # elif init == 'identity':
    #     R = np.identity(D, dtype=np.float32)  # D x D
    #     X_rotated = X
    # elif init == 'random':
    #     R = np.random.randn(D, D).astype(np.float32)
    #     R = orthonormalize_rows(R)
    #     X_rotated = opq_rotate(X, R)
    # else:
    #     raise ValueError("Unrecognized initialization method: ".format(init))
    #     # X_rotated = opq_rotate(X, R)

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
        err = compute_reconstruction_error(X_rotated, X_hat)
        print "---- OPQ {}x{}b iter {}: mse / variance = {:.5f}".format(
            ncodebooks, codebook_bits, it, err)

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

    # X_hat = reconstruct_X_pq(assignments, codebooks)
    # errors = X_rotated - X_hat
    # err = np.mean(errors * errors) / normalize_mse_by
    err = compute_reconstruction_error(X_rotated, X_hat)
    t = time.time() - t0
    print "---- OPQ {}x{}b final mse / variance = {:.5f} ({:.3f}s)".format(
        ncodebooks, codebook_bits, err, t)

    return codebooks, assignments, R


# ================================================================ mine

# @_memory.cache
def learn_rpq(X_train, ncodebooks, observed_bits=4, latent_bits=2,
              initial_kmeans_iters=1, init='gauss', opq_iters=0):
    """init in {'gauss', 'identity', 'random'}"""

    t0 = time.time()

    X_train = X_train.astype(np.float32)
    N, D = X_train.shape
    total_bits = observed_bits + latent_bits
    ncentroids_total = int(2**total_bits)
    ncentroids_obs = int(2**observed_bits)
    ncentroids_lat = int(2**latent_bits)

    X_rotated, R = opq_initialize(X_train, ncodebooks=ncodebooks, init=init)

    # huh; R is a valid rotation, but variance of X is way different
    # _debug_rotation(R)
    # print "orig var, rotated var: ", np.var(X_train), np.var(X_rotated)
    # return

    remainder = D % (ncodebooks + 1)
    if remainder:
        pad_len = (ncodebooks + 1) - remainder
        padding = np.zeros((N, pad_len), dtype=np.float32)
        X_rotated = np.hstack((X_rotated, padding))
        D = X_rotated.shape[1]
        assert D % (ncodebooks + 1) == 0


    subspace_len = int(D / (ncodebooks + 1))
    centroid_len = 2 * subspace_len

    # ncentroids = ncentroids_total  # use total number of centroids
    # codebooks = np.empty((ncentroids, ncodebooks, centroid_len))
    # assignments = np.empty((N, ncodebooks), dtype=np.int)

    # # PQ learning loop, but with subspaces overlapping
    # for c in range(ncodebooks):
    #     start = c * subspace_len
    #     end = start + 2 * subspace_len
    #     data = X_rotated[:, start:end]
    #     print "data in subspace ", c
    #     print data
    #     centroids, labels = kmeans(
    #         data, ncentroids, max_iter=initial_kmeans_iters)
    #     codebooks[:, c, :] = centroids
    #     assignments[:, c] = labels

    # print "X_rotated.shape", X_rotated.shape
    # print "subspace_len", subspace_len
    X_overlap = construct_X_overlap(X_rotated, subspace_len)

    if opq_iters > 0:
        codebooks, assignments, R_ov = learn_opq(X_overlap,
                                                 ncodebooks=ncodebooks,
                                                 codebook_bits=total_bits,
                                                 niters=opq_iters,
                                                 init='identity')
        X_overlap_hat = reconstruct_X_pq(assignments, codebooks)
        X_overlap_hat = opq_undo_rotate(X_overlap_hat, R_ov)
    else:
        codebooks, assignments = learn_pq(X_overlap, ncentroids=ncentroids_total,
                                          nsubvects=ncodebooks,
                                          subvect_len=centroid_len)
        X_overlap_hat = reconstruct_X_pq(assignments, codebooks)

    # print "X_overlap[_hat] shapes: ", X_overlap.shape, X_overlap_hat.shape
    X_hat = reconstruct_X_from_X_overlap(X_overlap_hat, D, subspace_len)

    err_overlap = compute_reconstruction_error(X_overlap, X_overlap_hat)
    err_rot = compute_reconstruction_error(X_rotated, X_hat)
    t = time.time() - t0
    print "=== RPQ {},{}: overlap, rotated mse / variance = " \
        " {:.5f}, {:.5f} ({:.3f}s)".format(
            observed_bits, latent_bits, err_overlap, err_rot, t)



    # SELF: pick up here by learning the factors
    centroid_norms = np.sum(codebooks * codebooks, axis=2)
    for c in range(ncodebooks - 1):
        centroids0 = codebooks[:, c, :]
        centroids1 = codebooks[:, c + 1, :]
        row_norms0 = centroid_norms[:, c]
        row_norms1 = centroid_norms[:, c + 1]

        pairwise_dists = sq_dists_to_vectors(
            centroids0, centroids1, row_norms0, row_norms1)





    # compressed_assignments = np.floor(assignments / ncentroids_lat).astype(int)
    # return codebooks, assignments, compressed_assignments


def compute_reconstruction_error(X, X_hat):
    errors = X - X_hat
    errors = np.mean(errors * errors, axis=1)
    # variances = np.maximum(.0001, np.var(X, axis=1))
    variances = np.var(X, axis=1)
    return np.mean(errors) / np.mean(variances)


def construct_X_overlap(X, subspace_len):
    N, D = X.shape
    D_overlap = 2 * (D - subspace_len)
    X_overlap = np.zeros((N, D_overlap), dtype=np.float32) + -999

    # deal with edges
    X_overlap[:, :subspace_len] = X[:, :subspace_len]
    X_overlap[:, -subspace_len:] = X[:, -subspace_len:]

    # double up middle subspaces, then flatten to 2D and insert into X_overlap
    X_mid = X[:, subspace_len:-subspace_len].reshape((N, -1, subspace_len))
    # print "X_mid"
    # print X_mid
    X_middle = np.tile(np.atleast_3d(X_mid), (1, 1, 2))
    X_overlap[:, subspace_len:-subspace_len] = X_middle.reshape((N, -1))

    assert not np.any(X_overlap == -999)

    return X_overlap


def reconstruct_X_from_X_overlap(X_overlap, D, subspace_len):
    N = X_overlap.shape[0]

    X_hat = np.empty((N, D), dtype=np.float32)
    # X_overlap[:, subspace_len:-subspace_len] /= 2

    # fill in first and last subspaces, which are only encoded once
    X_hat[:, :subspace_len] = X_overlap[:, :subspace_len]
    X_hat[:, -subspace_len:] = X_overlap[:, -subspace_len:]

    # print "D, N, M", D, N, M
    # print "subspace_len", subspace_len
    # print "X_overlap.shape", X_overlap.shape
    # print "X_mid.shape", X_overlap[:, subspace_len:-subspace_len].shape

    X_mid = X_overlap[:, subspace_len:-subspace_len].reshape((N, -1, 2 * subspace_len))
    # X_mid = X_overlap[:, subspace_len:-subspace_len].reshape((N, -1, 2))
    # X_mid = X_overlap[:, subspace_len:-subspace_len].reshape((N, -1, 2))
    # X_mid = np.mean(X_mid, axis=2)
    X_mid = (X_mid[:, :, :subspace_len] + X_mid[:, :, subspace_len:]) / 2
    X_hat[:, subspace_len:-subspace_len] = X_mid.reshape((N, -1))
    # TODO uncomment above
    # X_hat[:, subspace_len:-subspace_len] = X_mid[:, :, 0].reshape((N, -1))
    # X_hat[:, subspace_len:-subspace_len] = X_mid[:, :, :subspace_len].reshape((N, -1))

    return X_hat


def reconstruct_X_rpq_oracle(assignments, codebooks):
    # subspace_len = int(codebooks.shape[2] / 2)
    ncentroids, ncodebooks, subspace_len = codebooks.shape
    N, M = assignments.shape
    subspace_len /= 2
    D = subspace_len * (ncodebooks + 1)
    # D_overlap = 2 * D
    assert M == ncodebooks  # codebooks and assignments must agree

    X_overlap = reconstruct_X_pq(assignments, codebooks)
    return reconstruct_X_from_X_overlap(X_overlap, D, subspace_len)


def reconstruct_X_rpq(compressed_assignments, codebooks):
    subspace_len = int(codebooks.shape[2] / 2)
    pass  # TODO


# ================================================================ Main

def test_rpq():  # TODO put in a real unit test
    D = 20
    ncodebooks = 4
    subspace_len = int(D / (ncodebooks + 1))
    X = np.tile(np.arange(D), (2, 1))
    X_overlap = construct_X_overlap(X, subspace_len)
    X_hat = reconstruct_X_from_X_overlap(X_overlap, D, subspace_len)
    assert np.array_equal(X, X_hat)

    # learn_rpq(X, ncodebooks=4, observed_bits=4, latent_bits=0)


def main():
    # test_rpq()
    # return

    import datasets
    # np.set_printoptions(formatter={'float':lambda x: '{:.3f}'.format(x)})
    np.set_printoptions(formatter={'float':lambda x: '{}'.format(int(x))})
    # tmp = datasets.load_dataset(
    X_train, Q, X_test, truth = datasets.load_dataset(
        # datasets.Random.UNIFORM, N=1000, D=64)
        # datasets.Glove.TEST_100, N=1000, D=16, norm_mean=True)
        # datasets.Glove.TEST_100, N=100000, D=96)
        datasets.Sift1M.TEST_100, N=10000, D=32, norm_mean=True)
        # datasets.Sift1M.TEST_100, N=100000, norm_mean=True)
        # datasets.Sift1M.TEST_100, N=50000)
        # datasets.Gist.TEST_100, N=20000, norm_mean=True)
        # datasets.Glove.TEST_100, D=96)
        # datasets.Random.BLOBS, N=10000, D=32)



    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0)
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=2)
    # print '------------------------ rpq with opq'
    # # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0, opq_iters=10)
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0, opq_iters=5)
    print '------------------------ rpq without opq'
    learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0)
    # print '------------------------ rpq with opq'
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=1, opq_iters=10)
    # print '------------------------ rpq without opq'
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=1)

    # print '--- gauss shuffle'
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0, init='gauss_shuffle')
    # print '--- gauss'
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=2, init='gauss')
    # print '--- gauss shuffle'
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=2, init='gauss_shuffle')

    print '------------------------ opq'
    # X_train[0, 0] = .05
    # return
    # X_train[-1, -1] += X_train[-1, -1] / 10 * np.random.randn()  # block joblib caching
    # learn_opq(X_train, ncodebooks=8, codebook_bits=4, niters=10, init='identity')
    learn_opq(X_train, ncodebooks=8, codebook_bits=4, niters=5, init='gauss')

    # ------------------------ gauss vs gauss_flat initialization
    # X_rotated, R = opq_initialize(X_train, ncodebooks=8, init='gauss')
    # print "---- gaussian init:"
    # _debug_rotation(R)
    # print "X_rotated col variances: ", np.var(X_rotated, axis=0)

    # X_rotated, R = opq_initialize(X_train, ncodebooks=8, init='gauss_flat')
    # print "---- flat gaussian init:"
    # _debug_rotation(R)
    # print "X_rotated col variances: ", np.var(X_rotated, axis=0)

    # print X_train.shape

    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=0)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=2)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=20, debug=True)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=20)
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=8, niters=20)

    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=0,
    #                                       initial_kmeans_iters=16)

    # in terms of reconstruction err, gaussian < identity < random = gauss_flat
    # niters = 5
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='random')
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='identity')
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='gauss')
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=4, niters=niters, init='gauss_flat')
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=8, niters=niters, init='gauss')
    # codebooks, assignments, R = learn_opq(X_train, ncodebooks=8, niters=niters, init='gauss_flat')

    # am I right that this will undo a rotation? EDIT: yes
    # R = random_rotation(4)
    # A = np.arange(16).reshape((4, 4))
    # tmp = np.dot(A, R.T)
    # print A
    # print np.dot(tmp, R).astype(np.int)
    # return


if __name__ == '__main__':
    main()
