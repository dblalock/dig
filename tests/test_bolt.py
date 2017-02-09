#!/usr/bin/env python

# this file is just a simple smoketest to ensure we can call the functions

import numpy as np
import dig


def main():
    # # check that our index classes are present in the module
    # idx = dig.KmeansIndex
    # idx = dig.MatmulIndex
    # idx = dig.SimpleIndex
    # idx = dig.AbandonIndex
    # enc = dig.BoltEncoder

    # try actually using bolt on some random data

    ncentroids = 16  # constant for bolt
    M = 8
    subvect_len = 4
    ncodebooks = 2 * M
    ncentroids_total = ncentroids * ncodebooks
    D = ncodebooks * subvect_len
    N = 20

    X = np.random.randn(N, D).astype(np.float32)
    centroids = np.random.randn(ncentroids_total, subvect_len).astype(np.float32)
    q = np.random.randn(D).astype(np.float32)

    enc = dig.BoltEncoder(M)
    enc.set_centroids(centroids)
    enc.set_data(X)
    dists = enc.dists_l2(q)
    print dists

    assert(len(dists) == N)


if __name__ == '__main__':
    main()
