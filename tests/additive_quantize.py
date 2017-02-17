#!/usr/bin/env python

import numpy as np

import product_quantize as pq

from joblib import Memory
_memory = Memory('.', verbose=0)


def learn_lsq(X_train, ncodebooks, codebook_bits=8, nopq_iters=10,
              nlsq_icm_iters=10, niters=10):
    """
    Args:
        X_train (N x D array): training data; each row is a D-dim vector
        ncodebooks (int): number of codebooks to use; we select one code from
            each codebook to encode each vector
        codebook_bits (int): the number of bits used to specify the code
            used within each codebook. This implies that there are
            `2 ** codebook_bits` codes within the codebook
        nopq_iters (int): number of iterations of OPQ used to initialize the
            codebooks
        nlsq_icm_iters (int): number of iterations of Iterated Conditional
            Modes used to generate the encoding for each vector
        niters (int): number of iterations for which to optimize
            the codes and codebooks. `nlsq_icm_iters` is used within this
            outer loop.
    """

    codebooks, assignments, R = pq.learn_opq(X_train, ncodebooks=ncodebooks,
                                             codebook_bits=codebook_bits,
                                             niters=nopq_iters)

    for i in xrange(ncodebooks):
        centroids = codebooks[:, i, :]
        # we have: raw_centroids * R.T = centroids;
        #   therefore R * raw_centroids = centroids.T
        print R.shape
        print centroids.shape
        raw_centroids = np.linalg.solve(R, centroids.T)
        assert np.allclose(np.dot(raw_centroids, R.T), centroids)


        wait, how do I even compute this? because centroids have fewer dims
        than R. What do I even want?



def main():
    import datasets
    # tmp = datasets.load_dataset(
    X_train, Q, X_test, truth = datasets.load_dataset(
        # datasets.Random.UNIFORM, N=1000, D=64)
        datasets.Glove.TEST_100, N=10000, D=32)
        # datasets.Glove.TEST_100, N=50000, D=96)
        # datasets.Sift1M.TEST_100, N=10000, D=32)
        # datasets.Gist.TEST_100, N=50000)
        # datasets.Glove.TEST_100, D=96)
        # datasets.Random.BLOBS, N=10000, D=32)

    # codebooks, assignments, R = pq.learn_opq(X_train, ncodebooks=4, niters=0)
    learn_lsq(X_train, ncodebooks=4, niters=0, nopq_iters=0)


if __name__ == '__main__':
    main()
