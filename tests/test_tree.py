#!/bin/env python

print("------------------------ running tree tests")

import time
import numpy as np
import dig

def all_eq(x, y):
	if (len(x) == 0) and (len(y) == 0):
		return True
	return np.max(np.abs(x - y)) < .001

def main():
	# N = 100
	N = 100 * 1000
	# D = 10
	D = 100
	P = 8
	r2 = 1. * D

	# ------------------------ data and index construction

	X = np.random.randn(N, D)
	X = np.cumsum(X, axis=1)

	q = np.cumsum(np.random.randn(D))

	tree = dig.BinTree(X, P)
	t = tree.getIndexConstructionTimeMs()
	print "index time: {}".format(t)

	# true dists; used for ground truth
	t0 = time.clock()
	diffs = X - q
	trueDists = np.sum(diffs*diffs, axis=1)
	t_python = (time.clock() - t0) * 1000
	print "-> brute force time\t= {}".format(t_python)

	# ------------------------ range query

	neighborIdxs = tree.rangeQuery(q, np.sqrt(r2))
	t = tree.getQueryTimeMs()
	neighborIdxs = np.sort(neighborIdxs)

	trueNeighborIdxs = np.where(trueDists <= r2)[0]

	if len(trueNeighborIdxs < 100):
		print "neighborIdxs: ", neighborIdxs
		print "trueNeighborIdxs: ", trueNeighborIdxs
	print "-> range query time\t= {}".format(t)

	assert(len(neighborIdxs) == len(trueNeighborIdxs))
	assert(all_eq(neighborIdxs, trueNeighborIdxs))

	# ------------------------ 1nn query

	nn = tree.knnQuery(q, 1)
	nn = nn[0] # unwrap from vector
	trueNN = np.argmin(trueDists)
	t = tree.getQueryTimeMs()

	# print "1nn, true 1nn = {}, {}".format(nn, trueNN)
	print "-> 1nn time\t\t= {}".format(t)
	assert(nn == trueNN)

	# ------------------------ knn query

	nn10 = tree.knnQuery(q, 10)
	t = tree.getQueryTimeMs()
	trueNN10 = np.argsort(trueDists)[:10]
	# print "nn10, true nn10 = {}, {}".format(nn10, trueNN10)
	print "-> knn time\t\t= {}".format(t)
	assert(all_eq(nn10, trueNN10))

if __name__ == '__main__':
	main()
