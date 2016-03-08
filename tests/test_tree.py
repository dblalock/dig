#!/bin/env python

import numpy as np
import dig


def main():
	N = 100
	D = 10
	P = 8
	r2 = 1. * D

	X = np.random.randn(N, D)
	X = np.cumsum(X, axis=1)

	q = np.cumsum(np.random.randn(D))

	tree = dig.BinTree(X, P)

	# works up to this point

	# this breaks it
	# neighborIdxs = tree.rangeQuery(q, 1., -1)
	# neighborIdxs = tree.rangeQuery(q, np.sqrt(r2), -1)
	neighborIdxs = tree.rangeQuery(q, np.sqrt(r2))

	diffs = X - q
	trueDists = np.sum(diffs*diffs, axis=1)
	trueNeighborIdxs = np.where(trueDists <= r2)[0]

	neighborIdxs = np.sort(neighborIdxs)

	assert(len(neighborIdxs) == len(trueNeighborIdxs))
	assert(np.array_equal(neighborIdxs, trueNeighborIdxs))

	# nn1 = tree.knnQuery(q, 1)
	# nn10 = tree.knnQuery(q, 10)


if __name__ == '__main__':
	main()
