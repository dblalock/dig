#!/usr/bin/env python

import numpy as np
from sklearn import cluster
import kmc2  # state-of-the-art kmeans initialization (as of NIPS 2016)


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


def kmeans(X, k):
    seeds = kmc2.kmc2(X, k)
    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=16).fit(X)
    return estimator.cluster_centers_, estimator.labels_


# def orthogonalize_rows(A):
#     Q, R = np.linalg.qr(A.T)
#     return Q.T


def learn_pq(X, ncentroids, nsubvects, subvect_len):
    codebooks = np.empty((ncentroids, nsubvects, subvect_len))
    assignments = np.empty((X.shape[0], nsubvects))
    for i in range(nsubvects):
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        centroids, labels = kmeans(X_in, ncentroids)
        codebooks[:, i, :] = centroids
        assignments[:, i] = labels

    return codebooks, assignments  # [2**nbits x M x D/M], [N x M]


def _update_centroids_opq(X, assignments, ncentroids):  # [N x D], [N x M]
    nsubvects = assignments.shape[1]
    subvect_len = X.shape[1] // nsubvects

    assert X.shape[0] == assignments.centroids[0]
    assert X.shape[1] % ncentroids == 0

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


# based on https://github.com/arbabenko/Quantizations/blob/master/opqCoding.py
def learn_opq(X, ncodebooks, codebook_bits=8, niters=20):

    X = X.astype(np.float32)
    N, D = X.shape
    ncentroids = int(2**codebook_bits)
    subvect_len = D // ncodebooks
    # codebook_indices = np.arange(ncodebooks, dtype=np.int)

    assert D % ncentroids == 0  # equal number of dims for each codebook

    R = np.identity(D, dtype=np.float32)  # D x D
    # X_rotated = np.dot(X, R.T)  # (N x D) * (D x D) = N x D

    # initialize codebooks by running kmeans on each rotated dim; this way,
    # setting niters=0 corresponds to normal PQ
    X_rotated = X
    codebooks, assignments = learn_pq(X_rotated, ncentroids=ncodebooks,
                                      nsubvects=ncentroids, subvect_len=subvect_len)
    # alternative: initialize codebooks by sampling randomly from the data
    # codebooks = np.zeros((ncodebooks, ncentroids, subvect_len))
    # all_idxs = np.arange(N, dtype=np.int)
    # for m in np.arange(ncodebooks):
    #     rand_idxs = np.random.choice(all_idxs, size=ncentroids, replace=False)
    #     start_col = subvect_len * i
    #     end_col = start_col + subvect_len
    #     codebooks[:, m, :] = X_rotated[rand_idxs, start_col:end_col]

    for it in np.arange(niters):
        # compute reconstruction errors
        X_hat = reconstruct_X_pq(assignments, codebooks)
        errors = X_rotated - X_hat
        print "OPQ iter {}: elementwise mse = {}".format(
            it, np.mean(errors * errors))

        # update rotation matrix based on reconstruction errors
        U, s, V = np.linalg.svd(np.dot(X_hat.T, X), full_matrices=False)
        R = np.dot(U, V)

        # update centroids using new rotation matrix
        X_rotated = np.dot(X, R.T)
        codebooks = _update_centroids_opq(X, assignments, ncentroids)
        assignments = _encode_X_pq(X_rotated, codebooks)

    X_hat = reconstruct_X_pq(assignments, codebooks)
    errors = X_rotated - X_hat
    print "OPQ iter {}: elementwise mse = {}".format(
        niters, np.mean(errors * errors))

    return codebooks, assignments, R

    # R = np.identity(dim)
    # rotatedPoints = np.dot(points, R.T).astype('float32')
    # codebookDim = dim / M
    # codebooks = np.zeros((M, K, codebookDim), dtype='float32')
    # # init vocabs
    # for i in xrange(M):
    #     perm = np.random.permutation(pointsCount)
    #     codebooks[i, :, :] = rotatedPoints[perm[:K], codebookDim*i:codebookDim*(i+1)].copy()
    # # init assignments
    # assigns = np.zeros((pointsCount, M), dtype='int32')
    # for i in xrange(M):
    #     (idx, dis) = ynumpy.knn(rotatedPoints[:,codebookDim*i:codebookDim*(i+1)].astype('float32'), codebooks[i,:,:], nt=30)
    #     assigns[:,i] = idx.flatten()
    # for it in xrange(ninit):
    #     approximations = reconstruct_X_pq(assigns, codebooks)
    #     errors = rotatedPoints - approximations
    #     error = 0
    #     for pid in xrange(pointsCount):
    #         error += np.dot(errors[pid,:], errors[pid,:].T)
    #     print 'Quantization error: ' + str(error / pointsCount)
    #     U, s, V = np.linalg.svd(np.dot(approximations.T, points), full_matrices=False)
    #     R = np.dot(U, V)
    #     rotatedPoints = np.dot(points, R.T).astype('float32')
    #     for m in xrange(M):
    #         counts = np.bincount(assigns[:,m])
    #         for k in xrange(K):
    #             codebooks[m,k,:] = np.sum(rotatedPoints[assigns[:,m]==k,codebookDim*m:codebookDim*(m+1)], axis=0) / counts[k]
    #     for m in xrange(M):
    #         subpoints = rotatedPoints[:,codebookDim*m:codebookDim*(m+1)].copy()
    #         (idx, dis) = ynumpy.knn(subpoints, codebooks[m,:,:], nt=30)
    #         assigns[:,m] = idx.flatten()
    # error = 0
    # for m in xrange(M):
    #     subpoints = rotatedPoints[:,m*codebookDim:(m+1)*codebookDim].copy()
    #     (idx, dis) = ynumpy.knn(subpoints, codebooks[m,:,:], nt=2)
    #     error += np.sum(dis.flatten())
    # print 'Quantization error: ' + str(error / pointsCount)
    # model = (codebooks, R)



def main():
    pass

if __name__ == '__main__':
    main()
