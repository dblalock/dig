#!/bin/env python

print("------------------------ running flock tests")
# TODO actual tests, rather than smoketest

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

import dig
import dig.datasets as ds
from dig import viz

import flock # python impl that works
import feature as feat
import viz_flock as flockviz

# np.random.seed(123)

def main():
	# tsList = ds.loadDataset(ds.SINES)
	# tsList = ds.loadDataset(ds.MSRC, whichExamples=range(1))
	# tsList = ds.loadDataset(ds.MSRC, whichExamples=range(2))
	# tsList = ds.loadDataset(ds.DISHWASHER, whichExamples=range(1))
	tsList = ds.loadDataset(ds.DISHWASHER, whichExamples=range(5))
	# tsList = ds.loadDataset(ds.TIDIGITS, whichExamples=range(2))
	# tsList = ds.loadDataset(ds.TIDIGITS, whichExamples=range(10))

	for ts in tsList:

		# ts.data = ts.data[:, :5]
		# ts.data = ts.data[::2, :]

		# make it have 3 signals with increasing amounts of noise
		# n = ts.data.shape[0]
		# noiseDim1 = ts.data + np.random.randn(n, 1) * .25
		# noiseDim2 = np.random.randn(n).reshape((-1, 1))
		# ts.data = np.hstack((ts.data, noiseDim1, noiseDim2))

		# seq = ts.data.ravel().reshape((-1, 1))
		# print seq.shape

		# print ts.data.shape
		# return

		# Lmin = .1
		# Lmax = .2
		Lmin = .05
		Lmax = .1
		Lmin_int = int(Lmin * len(ts.data))
		Lmax_int = int(Lmax * len(ts.data))

		# ts.data = ts.data[:, ::5] # downsample by factor of 5

		# plt.plot(ts.data[:, :5])
		# plt.show()
		# return

		ff = dig.FlockLearner()
		ff.learn(ts.data.T, Lmin, Lmax)

		T = ff.getTimeSeries()
		Phi = ff.getFeatureMat()
		PhiBlur = ff.getBlurredFeatureMat()
		W = ff.getPattern()
		Wsum = ff.getPatternSum()
		starts = ff.getInstanceStartIdxs()
		ends = ff.getInstanceEndIdxs()
		windowLen = ff.getWindowLen()

		print "windowLen: ", windowLen

		# scores = ff.getSeedScores()
		seeds = ff.getSeeds()


		# startIdxs, endIdxs, model, X, Xblur = flock.learnFFfromSeq(ts.data, Lmin,
		# 	Lmax)


		# ts.plot()

		printCandidates = False
		if printCandidates:
			FPhi = Phi
			FPhiBlur = PhiBlur
			for FPhi, FPhiBlur in [(Phi, PhiBlur), (X, Xblur)]:
				for seed in seeds:

					# Lmax = int(len(ts.data))
					# dotProds = dig.dotProdsForSeed(FPhi, FPhiBlur, seed, windowLen)
					dotProds = dig.dotProdsForSeed(FPhi, FPhiBlur, seed, windowLen)
					dotProds2 = flock.dotProdsForSeed(FPhi, FPhiBlur, seed, windowLen)

					# plt.figure()
					# plt.plot(dotProds)
					# plt.plot(dotProds2)

					candidates = dig.candidatesFromDotProds(dotProds, Lmin_int)
					candidates2 = flock.candidatesFromDotProds(dotProds, Lmin_int)

					# candidates = ff.getCandidatesForSeed(FPhi, FPhiBlur, long(seed))
					# candidates2 = flock.get
					print "{}) {} vs {}".format(seed, candidates, candidates2)

					assert np.allclose(dotProds, dotProds2)

					# cpp impl seems to be much more accurate than python
					# version, which has a known bug in the func that
					# finds spaced relative maxima
					# if len(candidates) != len(candidates2):
					# 	plt.plot(dotProds)
					# 	plt.plot(dotProds2)
					# 	plt.show()
					# 	assert(False)

		# okay, so it's receiving, storing, and returning mats correctly
		# print "T shape", T.shape
		# ts.plot()
		# plt.plot(T.T)
		# plt.title("Orig TS and TS from FF object")

		# alright, seed scores look awesome...
		# scores = ff.getSeedScores()
		# plt.figure()
		# plt.plot(scores)

		# are structure score computations the same?
		# -> little bit of noise, but basically yes
		# subseqLen = 40
		# walks = dig.createRandomWalks(seq.ravel(), subseqLen, 100)
		# structureScores = dig.structureScores1D(seq.ravel(), subseqLen, walks)
		# structureScores2 = feat.windowScoresRandWalk(ts.data, subseqLen)
		# plt.figure()
		# plt.plot(structureScores / np.max(structureScores))
		# plt.plot(structureScores2)

		# are seed scores about right?
		# scores = ff.getSeedScores()
		# scores2 = flock._seedScores(seq, Lmin_int)
		# plt.figure()
		# plt.plot(scores)
		# plt.plot(scores2)


		# okay, so python impl nails it; clearly we have a bug somewhere...
		# startIdxs, endIdxs, model, X, Xblur = flock.learnFFfromSeq(ts.data, Lmin,
		# 	Lmax)
		# plt.figure()
		# viz.imshowBetter(model)
		# plt.colorbar()
		# plt.figure()
		# viz.imshowBetter(X)
		# plt.colorbar()
		# plt.figure()
		# viz.imshowBetter(Xblur)
		# plt.colorbar()

		print "cpp startIdxs:", starts
		print "cpp endIdxs:", ends
		# print "python startIdxs:", startIdxs
		# print "python endIdxs:", endIdxs

		# random walks seem to be good
		# walks = dig.createRandomWalks(ts.data.ravel(), 80, 100)
		# plt.plot(walks.T)
		# for walk in walks:
		# 	if np.abs(np.mean(walk)) > .0001:
		# 		print np.mean(walk)
		# 		print "walk is not zero-meaned!"

		# phi has reasonable shape (a few rows, same num cols = ts len)
		# print "Phi shape", Phi.shape
		# print "PhiBlur shape", PhiBlur.shape

		# print "Phi:"
		# for row in Phi[:20]:
		# 	print row

		ploFeatureMats = 0
		if ploFeatureMats:
			# yep, phi and phi_blur look right
			# well, sort of...phi consistently has fewer 1s...
			_, axes = plt.subplots(2,2)
			viz.imshowBetter(Phi, ax=axes[0,0])
			viz.imshowBetter(PhiBlur, ax=axes[1,0])
			viz.imshowBetter(X, ax=axes[0,1])
			viz.imshowBetter(Xblur, ax=axes[1,1])

		# ya, phi_blur and blurring of phi using python code are identical
		# Lfilt = int(Lmin * len(ts.data))
		# print "Lfilt", Lfilt
		# # PhiBlur2 = _filterRows(Phi, Lfilt)
		# _, PhiBlur2 = preprocessFeatureMat(Phi, Lfilt)

		# plt.figure()
		# viz.imshowBetter(PhiBlur2)
		# plt.colorbar()

		# for row in W:
		# 	print row

		# print "True pattern sum", Wsum
		# print "sum of rx'd pattern", np.sum(W)
		# print "W nonzeros: ", np.nonzero(W)

		# plt.figure()
		# viz.imshowBetter(W)
		# plt.colorbar()

		# print "python Phi shape, Phi sum, PhiBlur sum", X.shape, np.sum(X), np.sum(Xblur)
		print "cpp    Phi shape, Phi sum, PhiBlur sum", Phi.shape, np.sum(Phi), np.sum(PhiBlur)

		plotOutput = False
		# plotOutput = True
		if plotOutput:
			axSeq, axSim, axPattern = flockviz.plotFFOutput(ts, starts, ends, Phi, W)
			# flockviz.plotFFOutput(ts, starts, ends, PhiBlur, W)
			axSeq.set_title("cpp output")

			# axes = flockviz.plotFFOutput(ts, startIdxs, endIdxs, X, W)
			# axes[0].set_title("python output")

	plt.show()


def _filterRows(X, filtLength):
	filt = np.hamming(filtLength)
	return filters.convolve1d(X, weights=filt, axis=1, mode='constant')


def preprocessFeatureMat(X, Lfilt):
	"""
	Binarizes and blurs the feature matrix

	Parameters
	----------
	X : 2D array
		The original feature matrix (presumably output by buildFeatureMat())
	Lfilt : int
		The width of the hamming filter used to blur the feature matrix.

	Returns
	-------
	X : 2D array
		The modified feature matrix without blur
	Xblur : 2D array
		The modified feature matrix with blur
	"""
	Xblur = _filterRows(X, Lfilt)

	# ensure that the maximum value in Xblur is 1; we do this by dividing
	# by the largets value within Lfilt / 2, rather than just clamping, so
	# that there's a smooth dropoff as you move away from dense groups of
	# 1s in X; otherwise it basically ends up max-pooled
	maxima = filters.maximum_filter1d(Xblur, Lfilt // 2, axis=1, mode='constant')
	Xblur[maxima > 0] /= maxima[maxima > 0]

	# have columns be adjacent in memory
	return np.asfortranarray(X), np.asfortranarray(Xblur)

# n = 100
# d = 2
# X = np.random.randn(d, n)
# Lmin = .1
# Lmax = .2

# ff = dig.FlockLearner()
# ff.learn(X, Lmin, Lmax)

# Phi = ff.getFeatureMat()
# PhiBlur = ff.getBlurredFeatureMat()
# W = ff.getPattern()
# starts = ff.getInstanceStartIdxs()
# ends = ff.getInstanceEndIdxs()

# print starts
# print ends

if __name__ == '__main__':
	main()

