#!/usr/bin/env python

import functools
import heapq
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

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


# ================================================================ Permutation

def greedy_balanced_partition(vals, num_buckets):
    N = len(vals)
    max_size = int(np.ceil(float(N) / num_buckets))
    # assert N / num_buckets == 0  # actually, no, this is fine
    sums = np.zeros(num_buckets, dtype=float)
    bucket_sizes = np.zeros(num_buckets, dtype=int)
    buckets = [[] for i in range(num_buckets)]

    idx_offset = 0
    # if len(vals) >= 2 * num_buckets:
    if False:
        vals_increasing = np.sort(vals)
        for i in range(num_buckets):
            end = N - 1 - i
            sums[i] = vals_increasing[i] + vals_increasing[end]
            buckets[i].append(i)
            buckets[i].append(end)
            bucket_sizes[i] = 2

        vals = vals[num_buckets:-num_buckets]
        idx_offset = num_buckets

    # sort vals in decreasing order of absolute value
    # val_order = np.argsort(np.abs(vals))[::-1]
    # vals = vals[val_order]

    # actually, just offset so that all values are nonnegative
    min_val = np.min(vals)
    vals = vals - min_val
    vals = np.sort(vals)[::-1]

    # print "N, num_buckets, max_size: ", N, num_buckets, max_size
    for i, val in enumerate(vals):
        eligible_idxs = np.where(bucket_sizes < max_size)[0]
        eligible_sums = sums[eligible_idxs]
        # print "eligible_sums:", eligible_sums
        which_sum = np.argmin(eligible_sums)
        # if val >= 0:
        #     which_sum = np.argmin(eligible_sums)
        # else:
        #     which_sum = np.argmax(eligible_sums)
        orig_idx = eligible_idxs[which_sum]
        bucket_sizes[orig_idx] += 1
        sums[orig_idx] += val + min_val
        buckets[orig_idx].append(i + idx_offset)

    return buckets, sums


def random_permutation(D, num_buckets):
    idxs = np.arange(D, dtype=np.int)
    np.random.shuffle(idxs)
    return idxs


def _compute_log_dets(cov, num_buckets):  # log determinants of diag blocks
    _, D = cov.shape
    d = D / num_buckets
    logdets = np.zeros(num_buckets)
    for i in range(num_buckets):
        start = i * d
        end = start + d
        sign, logdet = np.linalg.slogdet(cov[start:end, start:end])
        logdets[i] = logdet
    return logdets


# def argmax_2d(X):
#     idx = np.argmax(X.ravel())
#     row = idx // X.shape[1]
#     col = idx % X.shape[1]
#     return row, col


def learn_groups_kruskal(pairwise_losses, num_buckets):
    D = pairwise_losses.shape[0]
    d = int(D / num_buckets)
    assert D == pairwise_losses.shape[1]  # must be square matrix
    assert D % num_buckets == 0
    assert d >= 2  # "groups" of size 1; I didn't mean this, so fail fast

    edges = []
    all_idxs = np.arange(D, dtype=np.int)
    all_rows_cols = itertools.product(all_idxs, all_idxs)
    edges = zip(pairwise_losses.flatten(), all_rows_cols)  # (-|cov_ij|, i, j)
    # for i in range()
    heapq.heapify(edges)
    # colors = np.zeros(D, dtype=np.int)
    colors = np.arange(D, dtype=np.int)
    color_counts = np.ones(D, dtype=np.int)
    # max_color = D - 1
    max_count = d
    # color_counts = {i: 1 for i in range(D)}

    for _ in range(len(edges)):  # TODO don't need quite this many iters
        abs_cov, (i, j) = heapq.heappop(edges)
        color_i, color_j = colors[i], colors[j]

        if i == j:  # can't connect a node to itself
            continue

        if colors[i] == colors[j]:
            continue  # already part of the same tree -> can't add edge

        if color_counts[color_i] + color_counts[color_j] >= max_count:
            continue  # resulting tree would be too big -> can't add edge

        # add the edge by merging the two trees (giving both the same color)
        old_color, new_color = colors[j], colors[i]
        colors[old_color] = new_color
        color_counts[new_color] += color_counts[old_color]
        # print "replacing color {} with {}; new count = {}".format(
        #     old_color, new_color, color_counts[new_color])

    uniq_colors, indices, counts = np.unique(
        colors, return_inverse=True, return_counts=True)
    print "colors: ", colors
    print "D, d:", D, d
    print "uniq colors: ", uniq_colors
    print "uniq color counts: ", counts

    print "groups:"
    groups = []
    for color in uniq_colors:
        groups.append(np.where(colors == color)[0])
        print groups[-1]

    return np.hstack(groups).reshape((num_buckets, d))

    # at this point, we have *at least* num_buckets buckets, but not
    # necessarily exactly this number of buckets


    # assert np.array_equal(counts, np.full(num_buckets, d))

    # return indices.reshape((num_buckets, d))


def construct_edges_heap(pairwise_losses):
    D = pairwise_losses.shape[1]
    assert D == pairwise_losses.shape[0]
    edges = []
    all_idxs = np.arange(D, dtype=np.int)
    all_rows_cols = itertools.product(all_idxs, all_idxs)
    edges = zip(pairwise_losses.flatten(), all_rows_cols)  # (-|cov_ij|, i, j)
    heapq.heapify(edges)

    return edges


def form_groups_of_2(pairwise_losses):
    D = pairwise_losses.shape[1]
    taken = np.zeros(D, dtype=bool)
    edges = construct_edges_heap(pairwise_losses)
    best_pairs = []
    for i in range(len(edges)):
        edge = heapq.heappop(edges)
        loss, (i, j) = edge
        if i == j:
            continue
        if taken[i] or taken[j]:
            continue
        best_pairs.append(edge)
        taken[i] = taken[j] = True

    # print "best pairs: ", '\n'.join(["{:.1f}, {}".format(*pair) for pair in best_pairs])
    # print "taken: ", taken
    # print "D, len(best_pairs)", D, len(best_pairs)
    assert len(best_pairs) == D / 2
    return best_pairs  # list of (loss, (i, j))


def form_groups_of_4(corrs, covs):
    D = corrs.shape[1]
    # pairwise_losses_2 = -np.abs(corrs)
    pairwise_losses_2 = -np.abs(covs)
    pairs = form_groups_of_2(pairwise_losses_2)
    D2 = len(pairs)
    D4 = D2 // 2
    assert D2 == D / 2
    assert D2 % 2 == 0  # D4 must be an integer

    # pairwise losses are logs of determinants; we want groups
    # to have the smallest determinants possible
    pairwise_losses_4 = np.zeros((D2, D2), dtype=np.float32)
    mat = np.empty((4, 4), dtype=np.float32)
    for i in range(D2):
        _, (idx0, idx1) = pairs[i]
        for j in range(i + 1, D2):
            _, (idx2, idx3) = pairs[j]

            # pull out cov mat entries corresponding to the 4 indices
            # included in the two pairs; the loss for the group formed from
            # these two pairs is the log of the determinant of the resulting
            # matrix, which is monotonically related to its entropy
            idxs = np.array([idx0, idx1, idx2, idx3])
            for m, row in enumerate(idxs):
                for n, col in enumerate(idxs):
                    mat[m, n] = covs[row, col]
            _, loss = np.linalg.slogdet(mat)
            # pairwise_losses_4[i, j] = pairwise_losses_4[j, i] = -loss
            pairwise_losses_4[i, j] = pairwise_losses_4[j, i] = loss

    pairs_4 = form_groups_of_2(pairwise_losses_4)
    assert len(pairs_4) == D4

    # we have the pairs of pairs; now unravel these into flat groups of 4
    groups = np.empty((D4, 4), dtype=np.int)
    group_logdets = np.empty(D4, dtype=np.float32)
    for n, pair4 in enumerate(pairs_4):
        _, (i, j) = pair4
        _, (idx0, idx1) = pairs[i]
        _, (idx2, idx3) = pairs[j]
        groups[n, :] = np.array([idx0, idx1, idx2, idx3])
        group_logdets[n] = pairwise_losses_4[i, j]

    # print "pairwise_losses_4: ", pairwise_losses_4
    assert np.array_equal(group_logdets, sorted(group_logdets))

    return groups, group_logdets


def learn_permutation(X_train, num_buckets):
    N, D = X_train.shape

    # loss_exponent = float(d) / D

    means = np.mean(X_train, axis=0)
    covs = np.dot(X_train.T, X_train) - np.outer(means, means)
    corrs = np.corrcoef(X_train.T)

    groups, group_logdets = form_groups_of_4(corrs, covs)

    # print "groups: ", groups
    # print "group logdets: ", group_logdets

    bucket_idxs, bucket_logdets = greedy_balanced_partition(
        group_logdets, num_buckets)

    # print "bucket logdets: ", bucket_logdets

    buckets = [groups[idxs] for idxs in bucket_idxs]  # not always equal sizes
    permutation = np.vstack(buckets).ravel()

    # permutation = groups.ravel()

    # TODO try first creating pairs of maximally correlated vars,
    # then using negative abs joint determinants as the pairwise losses

    # TODO use corr, not cov

    # groups = learn_groups_kruskal(-np.abs(cov), num_buckets)
    # corrs = np.corrcoef(X_train.T)
    # groups = learn_groups_kruskal(-corrs, num_buckets)
    # print "groups: ", groups

    # permutation = groups.ravel()

    # X_shuf = X_train[:, permutation]
    # corrs_shuf = np.corrcoef(X_shuf.T)
    # means_shuf = means[permutation]
    # cov_shuf = np.dot(X_shuf.T, X_shuf) - np.outer(means_shuf, means_shuf)

    logdets = _compute_log_dets(covs, num_buckets)
    print "initial submatrix log determinants:", logdets
    print "total: ", np.sum(logdets)
    covs_shuf = covs[:, permutation]
    covs_shuf = covs_shuf[permutation, :]
    logdets_shuf = _compute_log_dets(covs_shuf, num_buckets)
    print "shuffled submatrix log determinants:", logdets_shuf
    print "total: ", np.sum(logdets_shuf)

    variances = np.var(X_train, axis=0)
    variances_shuf = variances[permutation]
    subspace_variances = variances.reshape((num_buckets, -1)).sum(axis=1)
    subspace_variances_shuf = variances_shuf.reshape((num_buckets, -1)).sum(axis=1)
    print "orig sums of variances in groups:", subspace_variances
    print "shuffled sums of variances in groups:", subspace_variances_shuf

    # _, axes = plt.subplots(1, 2, figsize=(10, 4))
    # sb.heatmap(covs, ax=axes[0])
    # sb.heatmap(covs_shuf, ax=axes[1])
    # plt.show()
    # # sb.heatmap(corrs, ax=axes[0])
    # # sb.heatmap(corrs_shuf, ax=axes[1])

    # _, axes = plt.subplots(1, 3, figsize=(12, 4))
    # corrs_shuf2 = corrs[:, permutation]
    # corrs_shuf2 = corrs_shuf2[permutation, :]
    # sb.heatmap(corrs_shuf2, ax=axes[2]) # ya, these are the same

    return permutation

    # return groups.ravel()

    # h = []
    # heapq.heapify
    # while np.any(colors == 0):



    # ------------------------
    # logdets = _compute_log_dets(cov, num_buckets)

    # # offset = np.min(logdets)
    # # offset_dets = logdets - offset
    # # dets = np.exp(offset_dets)
    # print "initial submatrix log determinants:", logdets
    # print "initial losses: ", np.exp(logdets * loss_exponent)

    # # hmm; quantization lower bound actually gets worse (at least on sift)
    # # when we permute randomly...suggests not actually equilizing variance
    # idxs = random_permutation(D, num_buckets)
    # cov = cov[:, idxs]
    # cov = cov[idxs, :]
    # logdets = _compute_log_dets(cov, num_buckets)
    # print "shuffled submatrix log determinants:", logdets
    # print "shuffled losses: ", np.exp(logdets * loss_exponent)


    # ------------------------
    # D = np.diag(np.sum(cov, axis=1))
    # laplacian = D - cov
    # print "D.shape", D.shape
    # print "cov.shape", cov.shape
    # print "L.shape", laplacian.shape
    # eigenvals, eigenvects = np.linalg.eigh(laplacian)
    # fiedler_vec = eigenvects[:, -2]
    # plt.stem(np.arange(len(fiedler_vec)), fiedler_vec.ravel())
    # print "fiedler_vec.shape", fiedler_vec.shape
    # plt.stem(fiedler_vec)

    # plt.figure()
    # sb.heatmap(cov)
    # plt.figure()
    # sb.heatmap(laplacian)
    # plt.show()


def permutation_to_rotation(permutation):
    """matrix R such that np.dot(X, R.T) = X[:, permutation]"""
    D = len(permutation)
    R = np.zeros((D, D), dtype=np.float32)

    for i, idx in enumerate(permutation):
        R[i, idx] = 1

    return R

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
    if not np.all(eigenvalues > 0):
        print "WARNING: some eigenvalues were nonpositive"

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
    cov = np.dot(X_train.T, X_train) - np.outer(means, means)
    eigenvals, eigenvects = np.linalg.eigh(cov)

    order_idxs = eigenvalue_allocation(ncodebooks, eigenvals, shuffle=shuffle)
    assert len(order_idxs) == X_train.shape[1]
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
        code_counts = np.bincount(assignments[:, m], minlength=ncentroids)
        codebooks[:, m] /= np.maximum(code_counts, 1).reshape((-1, 1))  # no div by 0

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
    elif init == 'permute':
        permutation = learn_permutation(X_train, num_buckets=ncodebooks)
        R = permutation_to_rotation(permutation)
        X_rotated = opq_rotate(X, R)
    else:
        raise ValueError("Unrecognized initialization method: ".format(init))
        # X_rotated = opq_rotate(X, R)

    return X_rotated, R


# loosely based on:
# https://github.com/arbabenko/Quantizations/blob/master/opqCoding.py
@_memory.cache
def learn_opq(X_train, ncodebooks, codebook_bits=8, niters=20,
              initial_kmeans_iters=1, init='gauss', max_nonzeros=-1,
              debug=False):
    """init in {'gauss', 'identity', 'random'}"""

    print "OPQ: Using init '{}'".format(init)

    t0 = time.time()

    X = X_train.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks
    # codebook_indices = np.arange(ncodebooks, dtype=np.int)

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
        # err = compute_reconstruction_error(X_rotated, X_hat, subvect_len=subvect_len)
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

        # plt.figure()
        # sb.heatmap(R)

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
    X_hat = reconstruct_X_pq(assignments, codebooks)
    err = compute_reconstruction_error(X_rotated, X_hat)
    t = time.time() - t0
    print "---- OPQ {}x{}b final mse / variance = {:.5f} ({:.3f}s)".format(
        ncodebooks, codebook_bits, err, t)

    if max_nonzeros > 0:  # This seems kinda bad; don't use it
        norms = np.linalg.norm(R, axis=1)
        print "old rotation mat norms mean, std", np.mean(norms), np.std(norms)

        norms = np.empty(D)
        for j in range(D):
            y = X_hat[:, j]
            v0 = R[j, :]  # initialize with corresponding projection from R
            R[j, :] = hard_threshold_pursuit(A=X, y=y, init=v0, s=max_nonzeros)
            norms[j] = np.linalg.norm(R[j, :])  # TODO rm

        norms = np.linalg.norm(R, axis=1)
        print "new rotation mat norms mean, std", np.mean(norms), np.std(norms)

        R /= norms.reshape((-1, 1))
        # print "new R norms after rescaling: ", np.linalg.norm(R, axis=1)

        X_rotated_orig = X_rotated
        X_rotated = opq_rotate(X, R)
        assignments = _encode_X_pq(X_rotated, codebooks)
        codebooks = _update_centroids_opq(X_rotated, assignments, ncentroids)

        X_hat = reconstruct_X_pq(assignments, codebooks)
        # err = compute_reconstruction_error(X_rotated, X_hat)
        err = compute_reconstruction_error(X_rotated_orig, X_hat)
        print " -- OPQ {}x{}b {}-sparse mse / variance = {:.5f} ({:.3f}s)" \
            .format(ncodebooks, codebook_bits, max_nonzeros, err, t)

    return codebooks, assignments, R


# ================================================================ thresh pq

def learn_htpq(X, ncodebooks, codebook_bits=8, niters=20,
               initial_kmeans_iters=1, max_nonzeros=4):

    t0 = time.time()

    X = X.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2 ** codebook_bits)
    subvect_len = D // ncodebooks

    assert D % subvect_len == 0  # equal number of dims for each codebook

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
    codebooks, assignments = learn_pq(X, ncentroids=ncentroids,
                                      nsubvects=ncodebooks,
                                      subvect_len=subvect_len,
                                      max_kmeans_iters=3)
                                      # max_kmeans_iters=1)

    R = np.zeros((D, D), dtype=np.float32)
    X_rotated = X

    for it in np.arange(niters):
        X_hat = reconstruct_X_pq(assignments, codebooks)
        residuals = X - X_hat
        # residuals = X_rotated - X_hat

        # compute reconstruction errors
        # err = compute_reconstruction_error(X_rotated, X_hat)
        err = compute_reconstruction_error(X, X_hat)
        print "---- HTPQ {}x{}b t={}: mse / variance, resids = {:.5f}, {:.2f}" \
            .format(ncodebooks, codebook_bits, it, err,
                    np.sum(residuals * residuals))

        # TODO think I have to do AQ decoding here
        #   -the problem as far as not being monotonic is that
        #   updating the centroids improves the encoding of X_rotated,
        #   but that doesn't necessarily reduce the residuals
        #   -should write out the math for this, approaching it as an AQ
        #   problem

        # print "max_nonzeros: ", max_nonzeros

        # update rotation matrix
        for j in range(D):
            targets = residuals[:, j]
            # v, _, _, _ = np.linalg.lstsq(X, targets)
            v = hard_threshold_pursuit(  # works with niters=0, which is lstsq
                A=X, y=targets, init='least_squares', s=max_nonzeros-1, niters=5)
                # A=X, y=targets, init='least_squares', s=max_nonzeros-1, niters=5, verbosity=1)
                # A=X, y=targets, init='least_squares', s=max_nonzeros-1, niters=0)

            assert np.sum(v != 0) == (max_nonzeros - 1)

            orig_err = np.sum(targets * targets)
            y_hat = np.dot(X, v)
            new_err = np.sum((targets - y_hat)**2)
            # print "orig err, new err (ratio: {:.4f}, {:.4f} ({:.4f})".format(
            #     orig_err, new_err, new_err / orig_err)
            assert new_err <= orig_err

            R[j, :] = v

            # v0 = R[j, :]
            # R[j, :] = hard_threshold_pursuit(
            #     A=X, y=targets, init='least_squares', s=max_nonzeros-1, niters=0)

            # if R[j, j] != 0:
            #     print "bad row {}:".format(j), R[j, :]
            # assert R[j, j] == 0

        # update assignments and codebooks
        X_rotated = X + np.dot(X, R.T)
        assignments = _encode_X_pq(X_rotated, codebooks)
        codebooks = _update_centroids_opq(X_rotated, assignments, ncentroids)

    X_hat = reconstruct_X_pq(assignments, codebooks)
    # err = compute_reconstruction_error(X, X_hat)
    err = compute_reconstruction_error(X, X_hat)
    t = time.time() - t0
    print " -- HTPQ {}x{}b {}-sparse mse / variance = {:.5f} ({:.3f}s)" \
        .format(ncodebooks, codebook_bits, max_nonzeros, err, t)

    # we have that X_hat is a reconstruction of X(R + I)
    # X_hat -=

    # # X_hat = reconstruct_X_pq(assignments, codebooks)
    # # err = compute_reconstruction_error(X, X_hat)
    # err = compute_reconstruction_error(X, X_hat)
    # t = time.time() - t0
    # print " -- HTPQ {}x{}b {}-sparse mse / variance = {:.5f} ({:.3f}s)" \
    #     .format(ncodebooks, codebook_bits, max_nonzeros, err, t)

    return codebooks, assignments, R + np.eye(D)


# ================================================================ perm pq

def _learn_permutation(X, X_hat, assignments, subvect_len):
    # N, D = X_perm.shape
    N, D = X.shape
    M = assignments.shape[1]

    # if perm is None:
        # perm = np.arange(D, dtype=np.int)
    perm = np.arange(D, dtype=np.int)

    assert D == X_hat.shape[1]
    assert D == len(perm)
    assert D == len(np.unique(perm))
    assert D % subvect_len == 0
    assert D / subvect_len == M
    perm = np.copy(perm)

    labels = np.unique(assignments)
    L = len(labels)

    assert labels[0] == 0
    assert labels[-1] == L - 1
    # print "labels: ", labels

    # X_reconstruct =
    means = np.empty((M, L, D))
    # err_upper_bounds = np.zeros((M, L, D))
    # current_errs = np.zeros((M, L, subvect_len))
    # compute feature means conditioned on centroid assignments
    for m in range(M):

        # _, counts = np.unique(assignments, return_counts=True)
        # print "code counts in codebook {}".format(m), counts

        for ll, lbl in enumerate(labels):
            which_rows = assignments[:, m] == lbl
            # rows = X_perm[which_rows, :]
            rows = X[which_rows, :]
            # print "rows.shape", rows.shape
            means[m, ll, :] = np.mean(rows, axis=0)

            # # compute err upper bound
            # which_rows = assignments == lbl
            # variances = np.variance(X[which_rows], axis=0)
            # err_upper_bounds[m, ll, :] = variances * len(which_rows)

            # # compute label-conditional error using current perm; the
            # # "upper bound" for cu
            # start = m * subvect_len
            # end = start + subvect_len
            # X_subspace = X[:, start:end]
            # variances = np.variance(X_subspace[which_rows], axis=0)
            # # current_errs[m, ll, :] = variances * len(which_rows)

    # diffs = X_perm - X_hat
    diffs = X - X_hat
    residuals = np.sum(diffs * diffs, axis=0)

    # figure out whether there are any pairs of cols we should swap
    nswaps = 0
    already_swapped = np.zeros(D, dtype=np.bool)
    # TODO iterate thru in decreasing order of residual so stuff with
    # bigger errors gets first crack at getting swapped

    # X_new = np.full(X.shape, 9999, dtype=np.float32)
    X_new = np.copy(X)

    for i in range(D):
        best_err_decrease = 0
        best_j = -1

        # if nswaps > -1:
        if nswaps > 0:
            break

        m_i = int(i / subvect_len)
        assigs_i = assignments[:, m_i]
        means_i = means[m_i, :, i]

        assert means_i.shape == (L,)

        # TODO rm
        best_col_i_reconstruction = None
        best_col_j_reconstruction = None

        if already_swapped[i]:
            continue  # only swap one time per iter

        for j in range(i + 1, D):
            m_j = int(j / subvect_len)
            means_j = means[m_i, :, j]
            assigs_j = assignments[:, m_j]

            assert means_j.shape == (L,)

            if m_i == m_j:
                continue  # switching dims in same subspace doesn't help

            if already_swapped[j]:
                continue  # only swap one time per iter

            # okay, so ya, centroids are the means conditioned on assignments
            # print "reconstructions: "
            # actual_i_recons = means_i[assigs_i]
            # print actual_i_recons[:10], X_hat[:10, i]
            # assert np.all(np.abs(actual_i_recons - X_hat[:, i]) - .0001)
            # actual_j_recons = means_j[assigs_j]
            # assert np.all(np.abs(actual_j_recons - X_hat[:, j]) - .0001)

            col_i_reconstruction = means_j[assigs_i]
            col_j_reconstruction = means_i[assigs_j]

            assert col_i_reconstruction.shape == (N,)
            assert col_j_reconstruction.shape == (N,)

            # diffs_i = X_perm[:, i] - col_i_reconstruction
            # diffs_j = X_perm[:, j] - col_j_reconstruction
            diffs_i = X[:, i] - col_i_reconstruction
            diffs_j = X[:, j] - col_j_reconstruction

            current_err = residuals[i] + residuals[j]
            new_err = np.sum(diffs_i * diffs_i) + np.sum(diffs_j * diffs_j)
            err_decrease = current_err - new_err

            # if i % 20 == 0 and j % 20 == 0:
            #     print "curr err, new err = {:.3g}, {:.3g}".format(current_err, new_err)

            # if False:
            if err_decrease > 0 and err_decrease > best_err_decrease:
                best_err_decrease = err_decrease
                best_j = j

                best_col_i_reconstruction = np.copy(col_i_reconstruction)
                best_col_j_reconstruction = np.copy(col_j_reconstruction)

        if best_err_decrease > 0:
            # if i % 5 == 0:
            if True:
                print "swapping cols {} and {}; err decrease = {:.3f}" \
                    .format(i, best_j, best_err_decrease)
            assert best_j > i

            nswaps += 1
            # perm[i] , perm[best_j] = perm[best_j], perm[i]
            tmp = perm[i]
            perm[i] = perm[best_j]
            perm[best_j] = tmp

            already_swapped[i] = True
            already_swapped[best_j] = True

            # TODO rm after debug
            X_new[:, i] = best_col_i_reconstruction
            X_new[:, best_j] = best_col_j_reconstruction

    # TODO rm
    if np.any(already_swapped):
        diffs_old = X[:, already_swapped] - X_hat[:, already_swapped]
        diffs_new = X[:, already_swapped] - X_new[:, already_swapped]
        residuals_old = np.sum(diffs_old * diffs_old, axis=0)
        residuals_new = np.sum(diffs_new * diffs_new, axis=0)
        old_err, new_err = np.sum(residuals_old), np.sum(residuals_new)
        print "old err, new err = ", old_err, new_err
        assert new_err <= old_err - .0001


    # # below is (seemingly) correct, but bound is too loose so it never
    # # swaps anything
    # pairwise_errs = np.zeros((D, D))
    # for i in range(D):
    #     for j in range(D):
    #         if i % subvect_len == j % subvect_len:
    #             continue  # switching dims in same subspace doesn't help
    #         diffs = X[:, i] - X_hat[:, j]
    #         pairwise_errs[i, j] = np.sum(diffs * diffs)

    # nswaps = 0
    # pairwise_errs += pairwise_errs.T
    # for i in range(D):
    #     for j in range(i + 1, D):
    #         current_err = pairwise_errs[i, i] + pairwise_errs[j, j]
    #         new_err = pairwise_errs[i, j]  # err [j, i] added by transpose
    #         if new_err < current_err:
    #             nswaps += 1
    #             perm[i], perm[j] = perm[j], perm[i]

    print "_learn_permutation: performed {} column swaps".format(nswaps)

    return perm


def _inverse_permutation(perm):
    idxs = np.arange(len(perm))
    return idxs[perm]


def learn_ppq(X, ncodebooks, codebook_bits=8, niters=20,
              initial_kmeans_iters=1, max_nonzeros=4):

    t0 = time.time()

    X = X.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2 ** codebook_bits)
    subvect_len = D // ncodebooks

    assert D % subvect_len == 0  # equal number of dims for each codebook

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
    codebooks, assignments = learn_pq(X, ncentroids=ncentroids,
                                      nsubvects=ncodebooks,
                                      subvect_len=subvect_len,
                                      # max_kmeans_iters=3)
                                      max_kmeans_iters=1)

    # ensure centroids actually means of their clusters (cuz using minibatch)
    codebooks = _update_centroids_opq(X, assignments, ncentroids)


    # R = np.zeros((D, D), dtype=np.float32)
    X_rotated = X
    perm = np.arange(D, dtype=np.int)

    for it in np.arange(niters):
        X_hat = reconstruct_X_pq(assignments, codebooks)
        # residuals = X - X_hat

        # compute reconstruction errors
        # err = compute_reconstruction_error(X_rotated, X_hat)
        residuals = X_rotated - X_hat
        err = compute_reconstruction_error(X_rotated, X_hat)
        print "---- PPQ {}x{}b t={}: mse / variance, resids = {:.5f}, {:.2f}" \
            .format(ncodebooks, codebook_bits, it, err,
                    np.sum(residuals * residuals))

        X_hat = X_hat[:, _inverse_permutation(perm)]
        perm = _learn_permutation(X, X_hat, assignments=assignments,
                                  subvect_len=subvect_len)
                                  # perm=None, subvect_len=subvect_len)

        # update assignments and codebooks
        X_rotated = X[:, perm]

        # TODO uncomment
        codebooks = _update_centroids_opq(X_rotated, assignments, ncentroids)
        assignments = _encode_X_pq(X_rotated, codebooks)

        # # did this actually reduce the error?
        # X_hat = reconstruct_X_pq(assignments, codebooks)
        # new_residuals = X_rotated - X_hat
        # new_resids = np.sum(new_residuals * new_residuals)
        # new_err = compute_reconstruction_error(X_rotated, X_hat)
        # print "Debug PPQ {}x{}b t={}: new mse / variance, resids = {:.5f}, {:.2f}".format(
        #     ncodebooks, codebook_bits, it, new_err, new_resids)
        # assert new_err <= err

    X_hat = reconstruct_X_pq(assignments, codebooks)
    # err = compute_reconstruction_error(X, X_hat)
    X_rotated = X[:, perm]
    err = compute_reconstruction_error(X_rotated, X_hat)
    t = time.time() - t0
    print " -- PPQ {}x{}b final mse / variance = {:.5f} ({:.3f}s)" \
        .format(ncodebooks, codebook_bits, err, t)

    return codebooks, assignments, perm


# ================================================================ redundant pq

# @_memory.cache
def learn_rpq(X_train, ncodebooks, observed_bits=4, latent_bits=2,
              initial_kmeans_iters=1, init='gauss', opq_iters=0):
    """init in {'gauss', 'identity', 'random'}"""

    t0 = time.time()

    X_train = X_train.astype(np.float32)
    N, D = X_train.shape
    total_bits = observed_bits + latent_bits
    ncentroids_total = int(2**total_bits)
    # ncentroids_obs = int(2**observed_bits)
    ncentroids_lat = int(2**latent_bits)

    # use all bits for initial clustering for now
    nbits = total_bits
    ncentroids = ncentroids_total

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
                                                 codebook_bits=nbits,
                                                 niters=opq_iters,
                                                 init='identity')
        X_overlap_hat = reconstruct_X_pq(assignments, codebooks)
        X_overlap_hat = opq_undo_rotate(X_overlap_hat, R_ov)
    else:
        codebooks, assignments = learn_pq(X_overlap, ncentroids=ncentroids,
                                          nsubvects=ncodebooks,
                                          subvect_len=centroid_len)
        X_overlap_hat = reconstruct_X_pq(assignments, codebooks)

    # print "X_overlap[_hat] shapes: ", X_overlap.shape, X_overlap_hat.shape
    X_hat = reconstruct_X_from_X_overlap(X_overlap_hat, D, subspace_len)

    # learn the factors that will let us infer the latent bits
    log_factors = np.empty((ncodebooks - 1, ncentroids, ncentroids), dtype=np.float32)
    # log_factors = np.empty((ncodebooks - 1, ncentroids ** 2), dtype=np.float32)
    for c in range(ncodebooks - 1):
        centroids0 = codebooks[:, c, :]
        centroids1 = codebooks[:, c + 1, :]
        assert np.array_equal(centroids0.shape, (ncentroids, 2 * subspace_len))
        assert np.array_equal(centroids1.shape, (ncentroids, 2 * subspace_len))
        centroids0 = centroids0[:, subspace_len:]
        centroids1 = centroids0[:, :subspace_len]
        # row_norms0 = centroid_norms[:, c]
        # row_norms1 = centroid_norms[:, c + 1]

        pairwise_dists = sq_dists_to_vectors(centroids0, centroids1)

        # sb.heatmap(pairwise_dists)
        # plt.show()

        log_factors[c, :, :] = pairwise_dists
        # log_factors[c, :] = pairwise_dists.ravel()

    # return

    # TODO external funcs need to take in opq rotation so stuff outside
    # this function body can compute reconstructions
    compressed_assignments = np.floor(assignments / ncentroids_lat).astype(int)
    X_overlap_hat2 = reconstruct_X_rpq(
        compressed_assignments, codebooks, log_factors, latent_bits=latent_bits)
    if opq_iters > 0:
        X_overlap_hat2 = opq_undo_rotate(X_overlap_hat2, R_ov)
    X_hat2 = reconstruct_X_from_X_overlap(X_overlap_hat2, D, subspace_len)

    print "X_rot, X_hat, X_hat2"
    print X_rotated[:5, :10]
    print X_hat[:5, :10]
    print X_hat2[:5, :10]

    err_overlap = compute_reconstruction_error(X_overlap, X_overlap_hat)
    err_rot = compute_reconstruction_error(X_rotated, X_hat)
    err_lat = compute_reconstruction_error(X_rotated, X_hat2)
    t = time.time() - t0
    print "=== RPQ {},{}: overlap, rotated, latent mse / variance = " \
        " {:.5f}, {:.5f}, {:.5f} ({:.3f}s)".format(
            observed_bits, latent_bits, err_overlap, err_rot, err_lat, t)

    return codebooks, assignments, compressed_assignments, log_factors


def compute_reconstruction_error(X, X_hat, subvect_len=-1):
    diffs = X - X_hat
    diffs_sq = diffs * diffs
    if subvect_len > 0:
        errs = []
        for i in range(0, diffs_sq.shape[1], subvect_len):
            # X_block = X[:, i:i+subvect_len]
            errs_block = diffs_sq[:, i:i+subvect_len]
            errs.append(np.mean(errs_block))
            # print "mse in block:", np.mean(errs_block) / np.sum(np.var(X_block, axis=1))
            # print "squared error in block:", np.mean(errs_block)
        print "   errors in each block: {} ({})".format(
            np.array(errs), np.sum(errs))

    errors = np.mean(diffs_sq, axis=1)
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

    # fill in first and last subspaces, which are only encoded once
    X_hat[:, :subspace_len] = X_overlap[:, :subspace_len]
    X_hat[:, -subspace_len:] = X_overlap[:, -subspace_len:]

    # fill in middle subspaces by averaging the overlapping sections
    X_mid = X_overlap[:, subspace_len:-subspace_len].reshape((N, -1, 2 * subspace_len))
    X_mid = (X_mid[:, :, :subspace_len] + X_mid[:, :, subspace_len:]) / 2
    X_hat[:, subspace_len:-subspace_len] = X_mid.reshape((N, -1))

    return X_hat


# def reconstruct_X_rpq_oracle(assignments, codebooks):  # uses true assignments
#     # subspace_len = int(codebooks.shape[2] / 2)
#     ncentroids, ncodebooks, subspace_len = codebooks.shape
#     N, M = assignments.shape
#     subspace_len /= 2
#     D = subspace_len * (ncodebooks + 1)
#     assert M == ncodebooks  # codebooks and assignments must agree

#     # X_overlap = reconstruct_X_pq(assignments, codebooks)
#     # return reconstruct_X_from_X_overlap(X_overlap, D, subspace_len)


def reconstruct_X_rpq(compressed_assignments, codebooks, log_factors,
                      latent_bits=2, random=False):
    ncentroids, ncodebooks, subspace_len = codebooks.shape
    N, M = compressed_assignments.shape
    subspace_len /= 2
    nstates = int(2**latent_bits)
    assert M == ncodebooks
    assert latent_bits >= 0
    # D = subspace_len * (ncodebooks + 1)

    if latent_bits == 0:
        return reconstruct_X_pq(compressed_assignments, codebooks)

    if random:
        assigs = compressed_assignments * nstates
        offsets = np.random.randint(nstates, size=assigs.shape)
        assigs += offsets
        return reconstruct_X_pq(assigs, codebooks)

    assigs = np.empty((N, M), dtype=np.int)
    offsets = np.arange(nstates, dtype=np.int).reshape((-1, 1))
    for n in range(N):
        # compute which centroid indices could be the true indices within
        # each codebook
        symbols = compressed_assignments[n]
        candidates = (symbols * nstates).reshape((1, -1))
        candidates = np.tile(candidates, (nstates, 1))
        candidates += offsets

        # print "candidates.shape", candidates.shape
        # latent_assigs_left

        # compute best path thru dist mat
        sums_so_far = np.zeros((M, nstates), dtype=np.float32)
        predecessors = np.zeros((M, nstates), dtype=np.int)

        # TODO rm
        relevant_logs = np.empty((M-1, nstates, nstates), dtype=np.float32)

        for m in range(1, M):
            left_candidates = candidates[:, m-1]
            right_candidates = candidates[:, m]
            for cur_state_idx, right_cand in enumerate(right_candidates):
                best_prev_idx = -1
                best_cost = 2. ** 31  # huge number
                for prev_state_idx, left_cand in enumerate(left_candidates):
                    path_so_far_cost = sums_so_far[m-1, prev_state_idx]
                    edge_cost = log_factors[m-1, left_cand, right_cand]

                    relevant_logs[m-1, prev_state_idx, cur_state_idx] = edge_cost

                    cost = path_so_far_cost + edge_cost
                    if cost < best_cost:
                        # print "{}, {}: new best cost: {} = {} + {}".format(
                        #     prev_state_idx, cur_state_idx, cost, path_so_far_cost, edge_cost)
                        best_cost = cost
                        best_prev_idx = prev_state_idx
                sums_so_far[m, cur_state_idx] = best_cost
                predecessors[m, cur_state_idx] = best_prev_idx
            # print "partial sums_so_far:\n", sums_so_far[:m+1].T

        states = np.empty(M, dtype=np.int)
        state = np.argmin(sums_so_far[-1, :])
        states[-1] = candidates[state, -1]
        state_idxs = np.empty(M, dtype=np.int)  # TODO rm
        state_idxs[-1] = state  # TODO rm
        for m in range(M - 1, 0, -1):  # M-2 through 1, inclusive
            state = predecessors[m, state]
            state_idxs[m-1] = state
            # print "state: ", state
            states[m-1] = candidates[state, m-1]
        assigs[n, :] = states

        # if n < 1:
        if False:
            print "states:\n", states
            print "state idxs:\n", state_idxs
            print "candidates:\n", candidates
            print "log_factors.shape:", log_factors.shape
            print "log_factors:"
            for m in range(M-1):
                print log_factors[m]
            print "relevant log_factors:"
            for m in range(M-1):
                print relevant_logs[m]

            # populate matrix of distances; best path will give the decoding
            # this probably works, but who knows
            dists_mat = np.zeros((nstates ** 2, M), dtype=np.float32)
            for m in range(1, M):
                left_candidates = candidates[:, m-1]
                right_candidates = candidates[:, m]
                combos = itertools.product(left_candidates, right_candidates)
                for i, (left, right) in enumerate(combos):
                    dists_mat[m, i] = log_factors[m-1, left, right]
            fig, axes = plt.subplots(3, figsize=(10, 10))
            sb.heatmap(dists_mat.T, annot=True, fmt="0.1f", ax=axes[0])
            sb.heatmap(sums_so_far.T, annot=True, fmt="0.1f", ax=axes[1])
            print "sums_so_far.T:\n", sums_so_far.T
            sb.heatmap(predecessors.T, annot=True, ax=axes[2])
            plt.show()

            import sys; sys.exit(0)

    return reconstruct_X_pq(assigs, codebooks)


# ================================================================ Sparse

# def hard_threshold_pursuit(A, y, s, init='zeros', niters=5,
@_memory.cache
def hard_threshold_pursuit(A, y, s, init='least_squares', niters=5,
                           normalized=True, eps=1e-6, verbosity=0):
    """Solves argmin_x ||y - Ax||_2^2, ||x||_0 <= s
    For details, see:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.4207&rep=rep1&type=pdf
    And for a matlab implementation, see:
        https://github.com/foucart/HTP/blob/master/HTPCode/HTP.m

    eps (float): elements with absolute values of eps times the largest
        absolute value in the intermediate solutions are rounded to 0
    """
    s = int(s)
    N, D = A.shape
    y = y.reshape((N, 1))  # column vector
    # x0 = x0.ravel() if x0 is not None else np.zeros(D, dtype=np.float32)
    try:
        if init is None or isinstance(init, str):
            raise Exception()  # jump to the except
        x0 = np.asarray(init)
    except:
        if init == 'least_squares':
            x_ideal, residual, _, _ = np.linalg.lstsq(A, y.ravel())
            x0 = x_ideal
        elif init == 'zeros':
            x0 = np.zeros(D)
        else:
            raise ValueError("init must be one of "
                             " {an array, 'least_squares', 'zeros'}")

    assert s > 0
    assert len(A.shape) == 2
    assert len(y) == N
    assert len(x0) == D
    assert len(np.squeeze(x0).shape) == 1

    # quantities to precompute
    B = np.dot(A.T, A)  # [D x N] * [N x D] = D x D
    Ay = np.dot(A.T, y)  # [D x N] * [N x 1] = [D x 1]

    # initialize the solution and nonzero indices
    idxs = np.argsort(np.abs(x0))[-s:].ravel()  # idxs is 'S' in the paper
    idxs = np.sort(idxs)
    x = x0.reshape((-1, 1))

    # print "idxs: ", idxs

    # print "s = ", s
    # print "A.shape", A.shape
    # print "B.shape", B.shape
    # print "Ay.shape", Ay.shape
    # print "B.X shape", np.dot(B, x).shape

    # diffs = y - np.dot(A, x)
    # print "HTP: initial idxs: ", idxs
    # print "HTP: initial (pre-thresholding) squared loss: ", np.sum(diffs * diffs)

    old_idxs = np.copy(idxs)
    for t in range(niters):
        update = Ay - np.dot(B, x)  # = A.T(y - Ax); [D x 1] - [D x 1]

        # print "update.shape", update.shape
        # print "A.shape", A.shape
        # print "A[:, idxs].shape", A[:, idxs].shape
        # print "x.shape", A.shape

        # if False:  # compute step size
        if normalized:  # compute step size
            numerator = np.linalg.norm(update[idxs])
            denominator = np.linalg.norm(np.dot(A[:, idxs], update[idxs]))
            mu = (numerator / denominator) ** 2
        else:
            mu = 1

        x += mu * update
        old_idxs = np.copy(idxs)
        idxs = np.argsort(np.abs(x.ravel()))[-s:]
        idxs = np.sort(idxs)

        x[:] = 0
        solution, _, _, _ = np.linalg.lstsq(A[:, idxs], y.ravel())
        x[idxs, 0] = solution  # specify 0 because x a col vector

        if verbosity > 1:
            diffs = y - np.dot(A, x)
            print "HTP: iter {} squared loss = {}".format(t, np.sum(diffs * diffs))

        if np.array_equal(old_idxs, idxs):
            if verbosity > 1:
                print "HTP: converged after {} iterations".format(t + 1)
            break

    if verbosity > 0:
        diffs = y - np.dot(A, x)
        loss_htp = np.sum(diffs * diffs)
        x_ideal, residual, _, _ = np.linalg.lstsq(A, y.ravel())
        diffs = y.ravel() - np.dot(A, x_ideal)
        loss_least_squares = np.sum(diffs * diffs)
        ratio = loss_htp / loss_least_squares
        print "HTP: loss via HTP, loss via least squares = " \
            "{:.3f},\t{:.3f}\t({:.3f}x)".format(
                loss_htp, loss_least_squares, ratio)

    return x.ravel()


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


def test_rpq2():  # TODO put in a real unit test
    N = 100
    D = 20
    ncodebooks = 4
    np.random.seed(123)
    X = np.tile(np.arange(D), (N, 1)) + np.random.randn(N, D)

    learn_rpq(X, ncodebooks=ncodebooks, observed_bits=4, latent_bits=0, init='identity')

    # learn_rpq(X, ncodebooks=ncodebooks, observed_bits=1, latent_bits=1, init='identity')
    # learn_rpq(X, ncodebooks=ncodebooks, observed_bits=4, latent_bits=2, init='identity')

    # SELF: pick up here by prolly having latent codes with same observed bits be similar


def test_permutation_to_rotation():
    perm = np.array([1, 3, 0, 2])
    R = permutation_to_rotation(perm)
    X = np.random.randn(4, 4)
    assert np.array_equal(X[:, perm], np.dot(X, R.T))
    # print X[:, perm]
    # print np.dot(X, R.T)


def test_hard_threshold_pursuit():
    np.random.seed(123)
    N = 100
    D = 60
    s = D // 2
    A = np.random.randn(N, D) + np.arange(D)
    y = np.random.randn(N) + np.arange(N) / 10 - N/2

    x = hard_threshold_pursuit(A=A, y=y, s=s)
    assert np.sum(np.abs(x) > .0001) == s

    # print np.linalg.norm(x)
    # print "x = ", x


def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})
    # # np.set_printoptions(formatter={'float':lambda x: '{}'.format(int(x))})

    # test_hard_threshold_pursuit()
    # return

    # test_permutation_to_rotation()
    # return

    # # test_rpq()
    # test_rpq2()
    # return

    # ncentroids_latent = 8
    # offsets = np.arange(ncentroids_latent, dtype=np.int).reshape((-1, 1))
    # symbols = np.arange(6)
    # candidates = (symbols * ncentroids_latent).reshape((1, -1))
    # candidates = np.tile(candidates, (ncentroids_latent, 1))
    # candidates += offsets
    # print candidates
    # return

    import datasets

    # tmp = datasets.load_dataset(
    X_train, Q, X_test, truth = datasets.load_dataset(
        # datasets.Random.GAUSS, N=5, D=64)
        # datasets.Random.UNIFORM, N=10, D=64)
        # datasets.Glove.TEST_100, N=1000, norm_mean=True)
        # datasets.Glove.TEST_100, N=10000, D=96, norm_mean=True)
        # datasets.Sift1M.TEST_100, N=10000, D=32, norm_mean=True)
        # datasets.Sift1M.TEST_100, N=50000, norm_mean=True)
        # datasets.Sift1M.TEST_100, N=10000, norm_mean=True)
        # datasets.Gist.TEST_100, N=1000, D=480, norm_mean=True)
        # datasets.Gist.TEST_100, N=10000, D=480, norm_mean=True)
        # datasets.Glove.TEST_100, D=96)
        # datasets.Glove, D=32)
        # datasets.Random.BLOBS, N=10000, D=32)
        # datasets.Mnist, N=10000, norm_mean=True)
        # datasets.Mnist, N=10000)
        # datasets.Convnet1M, N=10000, norm_mean=True)
        # datasets.Deep1M, N=10000, norm_mean=True)  # breaks gauss init
        # datasets.LabelMe, norm_mean=True, num_queries=500)
        datasets.LabelMe, norm_mean=True)

    # learn_ppq(X_train, ncodebooks=16, codebook_bits=4, niters=10)

    # learn_htpq(X_train, ncodebooks=8, max_nonzeros=32, niters=10)
    # learn_htpq(X_train, ncodebooks=8, max_nonzeros=4, niters=10)
    # learn_htpq(X_train, ncodebooks=8, max_nonzeros=3, niters=10)

    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0)
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=2)
    # print '------------------------ rpq with opq'
    # # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0, opq_iters=10)
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0, opq_iters=5)
    # print '------------------------ rpq without opq'
    # learn_rpq(X_train, ncodebooks=8, observed_bits=4, latent_bits=0)
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

    # print '------------------------ opq'
    # # X_train[0, 0] = .05
    # # return
    # # X_train[-1, -1] += X_train[-1, -1] / 10 * np.random.randn()  # block joblib caching
    # # learn_opq(X_train, ncodebooks=8, codebook_bits=4, niters=10, init='identity')
    # learn_opq(X_train, ncodebooks=8, codebook_bits=4, niters=5, init='gauss')

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

    # looks like permutation learning often does better than identity if
    # adjecent dims aren't already highly correlated
    # perm = random_permutation(X_train.shape[1], 8)
    # X_train = X_train[:, perm]

    # # yep, mnist looks right
    # sb.heatmap(  Q[3].reshape((28, 28)))  # noqa
    # print np.min(Q[3])
    # print np.max(Q[3])
    # plt.show()
    # return

    # print Q.shape
    # return

    # # in terms of reconstruction err, gaussian < identity < random = gauss_flat
    # niters = 10
    niters = 5
    # # niters = 0
    M = 16
    bits = 4
    learn_func = functools.partial(learn_opq, ncodebooks=M, codebook_bits=bits, niters=niters)
    # codebooks, assignments, R = learn_func(X_train, init='identity')
    # codebooks, assignments, R = learn_func(X_train, init='gauss_flat')
    # codebooks, assignments, R = learn_func(X_train, init='random')
    # codebooks, assignments, R = learn_func(X_train, init='gauss')
    # codebooks, assignments, R = learn_func(Q, init='gauss')
    # codebooks, assignments, R = learn_func(X_train, init='gauss', max_nonzeros=32)
    # codebooks, assignments, R = learn_func(X_train, init='identity', max_nonzeros=16)
    # codebooks, assignments, R = learn_func(X_train, init='identity', max_nonzeros=4)

    # # _, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plt.figure()
    # sb.heatmap(R)
    # # R_covs = np.cov(R.T)
    # # sb.heatmap(R_covs)
    # plt.show()

    # am I right that this will undo a rotation? EDIT: yes
    # R = random_rotation(4)
    # A = np.arange(16).reshape((4, 4))
    # tmp = np.dot(A, R.T)
    # print A
    # print np.dot(tmp, R).astype(np.int)
    # return

    # learn_permutation(X_train, num_buckets=4)
    # learn_permutation(X_train, num_buckets=8)
    # learn_permutation(X_train, num_buckets=16)

    # # ------------------------ sparsePCA appears to be mediocre; non-minibatch
    # # version is incredibly slow, and minibatch version yields not-great
    # # covariances even on SIFT, which has really large principal components
    # from sklearn import decomposition
    # D = X_train.shape[1]
    # # spca = decomposition.PCA(n_components=D)
    # # spca = decomposition.SparsePCA(n_components=D, alpha=0)
    # # spca = decomposition.SparsePCA(n_components=D, alpha=0.25)
    # # spca = decomposition.MiniBatchSparsePCA(n_components=D, alpha=0.25, n_iter=1000)
    # # spca = decomposition.MiniBatchSparsePCA(n_components=D, alpha=0.25, n_iter=1000, batch_size=10)
    # # spca = decomposition.MiniBatchSparsePCA(n_components=D, alpha=0.25, n_iter=100, batch_size=10)
    # spca = decomposition.MiniBatchSparsePCA(n_components=D, alpha=1.0, n_iter=100, batch_size=10)
    # X_out = spca.fit_transform(X_train) * D * D
    # components = spca.components_
    # print "nums nonzeros: ", np.sum(components > 0.001, axis=1)
    # covs = np.cov(X_out.T)
    # _, axes = plt.subplots(1, 2)
    # sb.heatmap(np.cov(X_train.T), ax=axes[0])
    # sb.heatmap(covs, ax=axes[1])
    # plt.show()


if __name__ == '__main__':
    main()
