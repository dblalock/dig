#!/bin/env python

print("------------------------ running flock tests")

import numpy as np
import dig

n = 100
d = 2
X = np.random.randn(d, n)
Lmin = .1
Lmax = .2

ff = dig.FlockLearner(X, Lmin, Lmax)

Phi = ff.getFeatureMat()
PhiBlur = ff.getBlurredFeatureMat()
W = ff.getPattern()
starts = ff.getInstanceStartIdxs()
starts = ff.getInstanceEndIdxs()

