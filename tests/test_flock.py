#!/bin/env python

print("------------------------ running flock tests")
# TODO actual tests, rather than smoketest

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

import dig
import dig.datasets as ds
from dig import viz


def main():
	tsList = ds.loadDataset(ds.SINES)

	for ts in tsList:

		print ts.data.shape

		Lmin = .1
		Lmax = .2

		ff = dig.FlockLearner()
		ff.learn(ts.data.T, Lmin, Lmax)

		T = ff.getTimeSeries()
		Phi = ff.getFeatureMat()
		PhiBlur = ff.getBlurredFeatureMat()
		W = ff.getPattern()
		starts = ff.getInstanceStartIdxs()
		ends = ff.getInstanceEndIdxs()

		# okay, so it's receiving, storing, and returning mats correctly
		# print "T shape", T.shape
		ts.plot()
		plt.plot(T.T)
		plt.title("Orig TS and TS from FF object")

		# random walks seem to be good
		# walks = dig.createRandomWalks(ts.data.ravel(), 80, 100)
		# plt.plot(walks.T)
		#
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

		# yep, phi and phi_blur look right
		# plt.figure()
		# viz.imshowBetter(Phi)
		# plt.colorbar()
		# plt.figure()
		# viz.imshowBetter(PhiBlur)
		# plt.colorbar()

		# ya, phi_blur and blurring of phi using python code are identical
		# Lfilt = int(Lmin * len(ts.data))
		# print "Lfilt", Lfilt
		# # PhiBlur2 = _filterRows(Phi, Lfilt)
		# _, PhiBlur2 = preprocessFeatureMat(Phi, Lfilt)

		# plt.figure()
		# viz.imshowBetter(PhiBlur2)
		# plt.colorbar()

		plt.figure()
		viz.imshowBetter(W)
		plt.colorbar()

		print starts
		print ends

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

