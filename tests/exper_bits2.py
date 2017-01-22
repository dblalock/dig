#!/usr/bin/env python

import functools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import time

import kmc2
from collections import namedtuple
from scipy import stats
from sklearn import cluster
from sklearn.decomposition import TruncatedSVD
# from numba import jit

import datasets
import quantize

from joblib import Memory
_memory = Memory('.', verbose=0)


# ================================================================ Distances

def dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=-1)


def dists_l1(X, q):
    diffs = np.abs(X - q)
    return np.sum(diffs, axis=-1)


def _learn_expected_dists_for_diffs(X_embed, X_quant, base_dist_func=dists_sq,
                                    samples_per_bin=1e3):
    # TODO try fitting dists based on orig data, not embedded

    assert np.array_equal(X_quant[:10], X_quant[:10].astype(np.int))
    assert X_embed.shape == X_quant.shape

    uniqs = np.unique(X_quant)
    cardinality = len(uniqs)
    dists = np.zeros(cardinality)
    counts = np.zeros(cardinality)

    assert np.max(uniqs) == (cardinality - 1)  # must be ints 0..b for some b

    nsamples = int(counts.size * samples_per_bin)
    for n in range(nsamples):
        row1, row2 = np.random.randint(X_embed.shape[0], size=2)
        col1, col2 = np.random.randint(X_embed.shape[1], size=2)
        diff = np.abs(X_quant[row1, col1] - X_quant[row2, col2]).astype(np.int)
        dist = base_dist_func(X_embed[row1, col1], X_embed[row2, col2])
        counts[diff] += 1
        dists[diff] += dist

    assert np.min(counts) > 0
    dists /= counts
    return dists - np.min(dists)
    # return dists / counts  # TODO uncomment
    # return np.array([base_dist_func(i, 0) for i in np.arange(cardinality)])


def learn_dists_func(X_embed, X_quant, base_dist_func=dists_sq,
                     samples_per_bin=1e3):
    """
    Args:
        X_embed (2D, array-like): the data just before quantization
        X_quant (2D, array-like): quantized version of `X_embed`
        base_dist_func (f(a, b) -> R+): function used to compute distances
            between pairs of scalars
        samples_per_bin (scalar > 0): the expected number of samples per bin

    Returns:
        f(X, q), a function with the same signature as `dists_sq` and `dists_l1`
    """

    expected_dists = _learn_expected_dists_for_diffs(
        X_embed, X_quant, base_dist_func, samples_per_bin)

    print "expected_dists: ", expected_dists

    def f(X, q, expected_dists=expected_dists):
        diffs = np.abs(X - q)
        orig_shape = diffs.shape
        # assert np.min(diffs)
        dists = expected_dists[diffs.ravel().astype(np.int)]
        return dists.reshape(orig_shape).sum(axis=-1)

    return f


def dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def dists_elemwise_l1(x, q):
    return np.abs(x - q)


LUT_QUANTIZE_FLOOR = 'floor'


def learn_query_lut(X_embed, X_quant, q_embed,
                    elemwise_dist_func=dists_elemwise_sq,
                    samples_per_bin=1e3,
                    quantize_algo=LUT_QUANTIZE_FLOOR):
    assert np.array_equal(X_embed.shape, X_quant.shape)
    assert np.equal(X_embed.shape[-1], q_embed.shape[-1])
    assert np.equal(X_embed.shape[-1], q_embed.ravel().shape[-1])

    ndims = q_embed.shape[-1]
    uniqs = np.unique(X_quant)
    cardinality = len(uniqs)
    distances = np.zeros((cardinality, ndims))
    counts = np.zeros((cardinality, ndims))

    assert cardinality == 4  # TODO rm
    assert np.min(uniqs) == 0
    assert np.max(uniqs) == (cardinality - 1)  # must be ints 0..b for some b

    nsamples = min(int(cardinality * samples_per_bin), X_embed.shape[0])
    all_cols = np.arange(ndims, dtype=np.int)
    for n in range(nsamples):
        bins = X_quant[n].astype(np.int)
        dists = elemwise_dist_func(X_embed[n], q_embed)
        counts[bins, all_cols] += 1
        distances[bins, all_cols] += dists.ravel()

    # TODO also learn avg dist and set scale factor such that avg point will
    # just barely integer overflow

    # TODO remove after debug
    # print counts.astype(np.int)
    # print dists
    # quantizer = Quantizer(X_embed, nbits=8, how=Quantizer.QUANTILE, shared_bins=True)
    # q_quant = quantizer.transform(q_embed).ravel()
    # print "q_quant.shape: ", q_quant.shape
    # for i in range(cardinality):
    #     for j in range(ndims):
    #         distances[i, j] = (i - q_quant[j]) ** 2
    # return distances

    # print distances / counts

    assert np.min(counts) > 0
    return np.asfortranarray(distances / counts)


# @jit
def _inner_dists_lut(X_quant, q_lut):
    # offset cols of X_quant so that col i has offset of i * `cardinality`;
    # this will allow us to directly index into q_lut all at once
    cardinality, ndims = q_lut.shape
    offsets = np.arange(ndims, dtype=np.int) * cardinality
    X_quant_offset = X_quant + offsets

    dists = q_lut.T.ravel()[X_quant_offset.ravel()]
    dists = dists.reshape(X_quant.shape)
    return np.sum(dists, axis=-1)


# @profile
def dists_lut(X_quant, q_lut):  # q_lut is [cardinality, ndims]
    """
    >>> X_quant = np.array([[0, 2], [1, 0]], dtype=np.int)
    >>> q_lut = np.array([[10, 11, 12], [20, 21, 22]]).T
    >>> dists_lut(X_quant, q_lut)
    array([32, 31])
    """
    assert X_quant.shape[-1] == q_lut.shape[-1]
    X_quant = np.atleast_2d(X_quant.astype(np.int))
    return _inner_dists_lut(X_quant, q_lut)


def dists_to_vects(X, q):
    row_norms = np.sum(X*X, axis=1, keepdims=True)
    q_norms = np.sum(q*q, axis=1)
    prods = np.dot(X, q.T)
    return -2 * prods + row_norms + q_norms


def hamming_dist(v1, v2):
    return np.count_nonzero(v1 != v2)


def hamming_dists(X, q):
    return np.array([hamming_dist(row, q) for row in X])


# ================================================================ Misc

def top_k_idxs(elements, k, smaller_better=False):
    if smaller_better:
        which_nn = np.arange(k)
        return np.argpartition(elements, kth=which_nn)[:k]
    else:
        which_nn = len(elements) - 1 - np.arange(k)
        return np.argpartition(elements, kth=which_nn)[-k:][::-1]


def find_knn(X, q, k):
    dists = dists_sq(X, q)
    idxs = top_k_idxs(dists, k, smaller_better=True)
    return idxs, dists[idxs]


def orthogonalize_rows(A):
    Q, R = np.linalg.qr(A.T)
    return Q.T


# ================================================================ Clustering

@_memory.cache
def kmeans(X, k):
    seeds = kmc2.kmc2(X, k)
    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=16).fit(X)
    return estimator.cluster_centers_, estimator.labels_


def groups_from_labels(X, labels, num_centroids):
    # form groups associated with each centroid
    groups = [[] for _ in range(num_centroids)]
    for i, lbl in enumerate(labels):
        groups[lbl].append(X[i])

    for i, g in enumerate(groups[:]):
        groups[i] = np.array(g, order='F')

    return groups


@_memory.cache  # TODO use X_train and X_test separately, as well as truth idxs
def load_dataset_and_groups(which_dataset, num_centroids=256,
                            **load_dataset_kwargs):
    X_train, q, X_test, true_nn = datasets.load_dataset(
        which_dataset, **load_dataset_kwargs)
    X = X_train  # TODO use train vs test
    assert q.shape[-1] == X.shape[-1]
    centroids, labels = kmeans(X, num_centroids)
    groups = groups_from_labels(X, labels, num_centroids)

    return Dataset(X, q, X, X_test, true_nn, centroids, groups)


Dataset = namedtuple('Dataset', [
    'X', 'q', 'X_train', 'X_test', 'true_nn', 'centroids', 'groups'])


# ================================================================ Preproc

# ------------------------------------------------ Z-Normalization

class Normalizer(object):
    def __init__(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.std = np.max(self.stds)

    def znormalize(self, A):
        return (A - self.means) / self.std


# ------------------------------------------------ Quantization (incl DBQ)

def cutoff_quantize(A, thresholds):
    out = np.empty(A.shape, dtype=np.int)
    if len(thresholds.shape) == 1:
        return np.digitize(A, thresholds)

    for i, col in enumerate(A.T):
        threshs = thresholds[:, i]  # use col i of threshs for col i of A
        out[:, i] = np.digitize(col, threshs)

    return out


def gauss_quantize_old(A, means, std, nbits=8, max_sigma=-1, normalize=True):
    nbins = int(2 ** nbits)
    if max_sigma <= 0:
        # set this such that end bins each have 1/nbins of the distro
        max_sigma = -stats.norm.ppf(1. / nbins)
        max_sigma *= (nbins / 2) / ((nbins / 2) - 1)

    # print "gauss_quantize: nbits = ", nbits
    # print "gauss_quantize: nbins = ", nbins
    # assert nbits == 2

    A_z = (A - means) / std if normalize else A
    max_val = 2 ** int(nbits - 1) - 1
    # max_val = 2 ** int(nbits) - 1  # TODO remove after debug
    min_val = -(max_val + 1)

    print "gauss_quantize: minval, maxval = ", min_val, max_val
    # print "gauss_quantize: nbins = ", nbins
    # assert nbits == 2

    scale_by = max_val / float(max_sigma)
    quantized = np.floor(A_z * scale_by)
    return np.clip(quantized, min_val, max_val).astype(np.int)


def fit_gauss_thresholds(A, nbits, shared=True, max_sigma=-1):
    nbins = int(2 ** nbits)
    quantiles = np.arange(1, nbins) / float(nbins)
    threshs = stats.norm.ppf(quantiles)

    thresholds = np.empty((nbins - 1, A.shape[1]))
    means = np.mean(A, axis=0)
    stds = np.std(A, axis=0)

    if shared:
        std = np.mean(stds)
        # XXX assumes means subtracted off
        # return threshs * std
        for i, std in enumerate(stds):
            thresholds[:, i] = threshs * std + means[i]
    else:
        # thresholds = np.empty(nbins - 1, A.shape[1])
        for i, std in enumerate(stds):
            thresholds[:, i] = threshs * std + means[i]

    return thresholds

    # if max_sigma <= 0:
    #     # set this such that end bins each have 1/nbins of the distro
    #     max_sigma = -stats.norm.ppf(1. / nbins)
    #     max_sigma *= (nbins / 2) / ((nbins / 2) - 1)


def fit_quantile_thresholds(X, nbits=-1, shared=True, nbins=-1):
    if nbins < 1:
        nbins = int(2 ** nbits)
    quantiles = np.arange(1, nbins) / float(nbins)
    percentiles = quantiles * 100
    if shared:
        return np.percentile(X, q=percentiles, interpolation='midpoint')
    return np.percentile(X, q=percentiles, axis=0, interpolation='midpoint')


def fit_kmeans_thresholds(X, nbits, shared=True):  # for manhattan hashing
    nbins = int(2 ** nbits)

    if shared or X.shape[1] == 1:  # one set of thresholds shared by all dims
        centroids, _ = kmeans(X.reshape((-1, 1)), nbins)
        centroids = np.sort(centroids.ravel())
        return (centroids[:-1] + centroids[1:]) / 2.

    # uniq set of thresholds for each dim
    thresholds = np.empty((nbins - 1, X.shape[1]))
    for i, col in enumerate(X.T):
        thresholds[:, i] = fit_kmeans_thresholds(col, nbits, shared=True)

    return thresholds


def dbq_quantize(A, lower_threshs, upper_threshs):
    return (A > lower_threshs).astype(np.float) + (A > upper_threshs)


def fit_dbq_thresholds(A, shared=True):
    return fit_quantile_thresholds(A, nbins=3, shared=shared)


class Quantizer(object):
    GAUSS = 'gauss'
    DBQ = 'dbq'
    KMEANS = 'kmeans'
    QUANTILE = 'quantile'

    def __init__(self, X, nbits=2, how=QUANTILE, shared_bins=True):
        self.X = X
        self.nbits = nbits
        self.how = how
        self.normalizer = Normalizer(X)  # just to store means and std
        if how == Quantizer.DBQ:
            self.dbq_thresholds = fit_dbq_thresholds(X, shared=shared_bins)
        elif how == Quantizer.KMEANS:
            self.kmeans_thresholds = fit_kmeans_thresholds(
                X, nbits=nbits, shared=shared_bins)
        elif how == Quantizer.QUANTILE:
            self.quantile_thresholds = fit_quantile_thresholds(
                X, nbits=nbits, shared=shared_bins)
        elif how == Quantizer.GAUSS:
            self.gauss_thresholds = fit_gauss_thresholds(
                X, nbits=nbits, shared=shared_bins)
        else:
            raise ValueError("Unrecognized quantization method: {}".format(how))

    def transform(self, A):
        if self.how == Quantizer.DBQ:
            return dbq_quantize(A, self.dbq_thresholds[0], self.dbq_thresholds[1])
        elif self.how == Quantizer.KMEANS:
            return cutoff_quantize(A, self.kmeans_thresholds)
        elif self.how == Quantizer.QUANTILE:
            return cutoff_quantize(A, self.quantile_thresholds)
        elif self.how == Quantizer.GAUSS:
            return cutoff_quantize(A, self.gauss_thresholds)
        else:
            raise ValueError("Unrecognized quantization method: {}".format(
                self.how))


# ================================================================ Encoder API

class EncoderMixin(object):

    def encode_X(self, X, **sink):  # needs to be overridden if no transform()
        return self.transform(X)

    def encode_q(self, q, **kwargs):
        return self.encode_X(q, **kwargs).ravel()

    def dists_true(self, X, q):
        # return np.sum(self.elemwise_dist_func(X, q), axis=-1)
        # return np.sum(dists_sq(X, q), axis=-1)
        return dists_sq(X, q)

    def fit_query(self, q, **sink):
        pass

    def dists_enc(self, X, q):
        return self.dists_true(X, q)


# class EmbedEncoder(EncoderMixin):

#     def __init__(self, dataset, embed_func_factory):
#         self.embed_func = embed_func_factory(dataset.X)

#     def encode_X(self, X, **kwargs):
#         return self.embed_func(np.atleast_2d(X), **kwargs)


# class EmbedAndQuantizeEncoder(EncoderMixin):

#     def __init__(self, dataset, embedder_factory, quantizer_factory):
#         self.embed_encoder = embedder_factory(dataset)
#         X = self.embed_encoder.encode_X(dataset.X)
#         self.quantizer = quantizer_factory(X)

#     def encode_X(self, X, **kwargs):
#         X_embed = self.embed_encoder.encode_X(X, **kwargs)
#         return self.quantizer.encode_X(X_embed)


# def create_cbe_encoder(dataset, nbits):
#     embed_func_factory = functools.partial(CirculantBinaryEmbedder, nbits=nbits)
#     return EmbedEncoder(dataset.X, embed_func_factory)


# ================================================================ Embedding

# ------------------------------------------------ CBE
class CirculantBinaryEmbedder(EncoderMixin):
    def __init__(self, X, nbits):
        D = X.shape[-1]
        self.nbits = nbits
        self.r = np.random.randn(D)
        self.R = np.fft.fft(self.r)
        self.signs = (np.random.randn(D) > 0) * 2 - 1

    def transform(self, X):
        # X_preproc = (X - centroids[idx]) * signs
        X_preproc = X * self.signs  # this yeilds corrs of like .9 on randwalks
        # X_preproc = (X - np.ones(D)) * signs  # alright, this is fine
        X_fft = np.fft.fft(X_preproc, axis=-1)
        X_rotated = np.fft.ifft(X_fft * self.R, axis=-1)
        X_rotated = X_rotated[..., :self.nbits]
        return np.real(X_rotated) > 0


# ------------------------------------------------ Random hyperplanes
class SignedRandomProjections(EncoderMixin):
    def __init__(self, X, nbits, orthogonal=False):
        self.hyperplanes = np.random.randn(nbits, X.shape[-1])
        if orthogonal:
            self.hyperplanes = orthogonalize_rows(self.hyperplanes)

    def transform(self, X):
        return np.dot(X, self.hyperplanes.T) > 0


# ------------------------------------------------ Striped rand projections
class StripedRandomProjections(EncoderMixin):
    def __init__(self, X, nbits, orthogonal=False):
        self.hyperplanes = np.random.randn(nbits, X.shape[-1])
        if orthogonal:
            self.hyperplanes = orthogonalize_rows(self.hyperplanes)

    def transform(self, X):
        prod = np.dot(X, self.hyperplanes.T)
        interval = np.max(prod) - np.min(prod)
        stripe_width = interval / 3.
        bins = np.floor(prod / stripe_width).astype(np.int)
        return np.mod(bins, 2).astype(np.bool)


# ------------------------------------------------ Partially Orthogonal SRP
class SuperbitLSH(EncoderMixin):
    def __init__(self, X, nbits, subvect_len=64):
        D = X.shape[-1]
        self.nbits = nbits
        self.subvect_len = subvect_len
        num_subvects = D / subvect_len
        assert D % subvect_len == 0
        self.projections = np.random.randn(nbits / num_subvects, subvect_len)
        # orthagonalize groups of subvect_len projections
        for i in range(0, len(self.projections), subvect_len):
            self.projections[i:i+subvect_len] = orthogonalize_rows(
                self.projections[i:i+subvect_len])

    def transform(self, X):
        new_shape = list(X.shape)
        new_shape[-1] = self.nbits
        X = X.reshape((-1, self.subvect_len))
        prods = np.dot(X, self.projections.T)
        prods = prods.reshape(new_shape)
        return prods > 0


# ------------------------------------------------ Sample subset of dims
class SampleDimsSketch(EncoderMixin):
    def __init__(self, X, ndims=64):
        self.keep_idxs = np.random.randint(X.shape[-1], size=ndims)

    def transform(self, X):
        return X[:, self.keep_idxs] if len(X.shape) == 2 else X[self.keep_idxs]


class QuantizedSampleDimsSketch(EncoderMixin):
    def __init__(self, X, ndims=64, **quantize_kwargs):
        self.inner_sketch = SampleDimsSketch(X, ndims=ndims)
        X = self.inner_sketch.transform(X)
        self.quantizer = Quantizer(X, **quantize_kwargs)

    def transform(self, X):
        ret = self.inner_sketch.transform(X)
        return self.quantizer.transform(ret)


# ------------------------------------------------ PCA / IsoHash

# @_memory.cache
def _fit_svd(X_train, n_components):
    return TruncatedSVD(n_components=n_components).fit(X_train)


# @_memory.cache
def _pca(svd, X, ndims):
    # return svd.transform(X)[:, 1:(ndims+1)]
    return svd.transform(X)[:, :ndims]


class Pca(object):
    DEFAULT_MAX_NUM_DIMS = 64

    def __init__(self, X, ndims=DEFAULT_MAX_NUM_DIMS):
        self.ndims = ndims
        self.pre_normalizer = Normalizer(X)
        X_train = self.pre_normalizer.znormalize(X)
        # self.svd = _fit_svd(X_train, ndims + 1)
        self.svd = _fit_svd(X_train, ndims)

        X_pca = self.transform(X_train, postnormalize=False)
        self.post_normalizer = Normalizer(X_pca)

    def transform(self, A, postnormalize=True):
        A_in = self.pre_normalizer.znormalize(A)
        A_pca = _pca(self.svd, A_in, ndims=self.ndims)
        # A_pca = self.svd.transform(A_in)[:, 1:(ndims+1)]
        if postnormalize:
            return self.post_normalizer.znormalize(A_pca)
        return A_pca


class PcaSketch(EncoderMixin):
    def __init__(self, X, ndims=64):
        self.ndims = ndims
        self.pca = Pca(X, ndims=ndims)

    def transform(self, X):
        return self.pca.transform(np.atleast_2d(X))


class QuantizedPcaSketch(EncoderMixin):
    def __init__(self, X, ndims=64, **quantize_kwargs):
        self.inner_sketch = RandomIsoHash(X, ndims=ndims)
        X = self.inner_sketch.transform(X)
        self.quantizer = Quantizer(X, **quantize_kwargs)

    def transform(self, X):
        ret = self.inner_sketch.transform(X)
        return self.quantizer.transform(ret)


class RandomIsoHash(EncoderMixin):
    def __init__(self, X, ndims=64):
        self.inner_sketch = PcaSketch(X, ndims=ndims)
        hyperplanes = np.random.randn(ndims, ndims)
        self.rotation = orthogonalize_rows(hyperplanes)

    def rotate(self, A):
        return np.dot(A, self.rotation.T)

    def transform(self, X):
        ret = self.inner_sketch.transform(X)
        return self.rotate(ret)


class QuantizedRandomIsoHash(EncoderMixin):
    def __init__(self, X, ndims=64, **quantize_kwargs):
        self.inner_sketch = RandomIsoHash(X, ndims=ndims)
        X = self.inner_sketch.transform(X)
        self.quantizer = Quantizer(X, **quantize_kwargs)

    def transform(self, X):
        ret = self.inner_sketch.transform(X)
        return self.quantizer.transform(ret)


# ------------------------------------------------ Random rotation
class RandomProjections(EncoderMixin):
    def __init__(self, X, ndims=64, orthogonal=False):
        self.ndims = ndims
        self.hyperplanes = np.random.randn(ndims, X.shape[-1])
        if orthogonal:
            self.hyperplanes = orthogonalize_rows(self.hyperplanes)

    def transform(self, X):
        return np.dot(X, self.hyperplanes.T)


class QuantizedRandomProjections(EncoderMixin):
    def __init__(self, X, ndims=64, orthogonal=False,  **quantize_kwargs):
        self.inner_sketch = RandomProjections(
            X, ndims=ndims, orthogonal=orthogonal)
        X = self.inner_sketch.transform(X)
        self.quantizer = Quantizer(X, **quantize_kwargs)

    def transform(self, X):
        ret = self.inner_sketch.transform(X)
        return self.quantizer.transform(ret)


# ------------------------------------------------ Dimension-specific LUT

# "Build sOme Lookup Tables"; or maybe
# BOLT distance = "Based On Lookup Tables" distance
class BoltEncoder(EncoderMixin):

    def __init__(self, dataset, inner_sketch,
                 elemwise_dist_func=dists_elemwise_sq,
                 **quantize_kwargs):
        self.inner_sketch = inner_sketch
        self.elemwise_dist_func = elemwise_dist_func
        self.X_embed = self.inner_sketch.transform(dataset.X)
        self.quantizer = Quantizer(self.X_embed, **quantize_kwargs)
        self.X_enc = self.quantizer.transform(self.X_embed)

    def transform(self, X):
        ret = self.inner_sketch.transform(X)
        return self.quantizer.transform(ret)

    def encode_q(self, q, **sink):
        return None  # fail fast if we try to use this

    def fit_query(self, q, **sink):
        q_embed = self.inner_sketch.transform(q)
        self.q_lut_ = learn_query_lut(self.X_embed, self.X_enc, q_embed,
                                      elemwise_dist_func=self.elemwise_dist_func)
        return self

    def dists_enc(self, X_enc, q_unused):
        return _inner_dists_lut(X_enc, self.q_lut_)


# ------------------------------------------------ Product Quantization

def _learn_centroids(X, ncentroids, nsubvects, subvect_len):
    ret = np.empty((ncentroids, nsubvects, subvect_len))
    for i in range(nsubvects):
        start_col = i * subvect_len
        end_col = start_col + subvect_len
        X_in = X[:, start_col:end_col]
        centroids, labels = kmeans(X_in, ncentroids)
        ret[:, i, :] = centroids

    return ret


def _parse_codebook_params(D, code_bits=-1, bits_per_subvect=-1, nsubvects=-1):
    if nsubvects < 0:
        nsubvects = code_bits // bits_per_subvect
    elif code_bits < 1:
        code_bits = bits_per_subvect * nsubvects
    elif bits_per_subvect < 1:
        bits_per_subvect = code_bits // nsubvects

    ncentroids = int(2 ** bits_per_subvect)
    subvect_len = D / nsubvects

    assert code_bits % bits_per_subvect == 0
    assert D % subvect_len == 0  # TODO rm this constraint

    return nsubvects, ncentroids, subvect_len


def _fit_opq_lut(q, centroids, elemwise_dist_func):
    _, nsubvects, subvect_len = centroids.shape
    assert len(q) == nsubvects * subvect_len

    q = q.reshape((1, nsubvects, subvect_len))
    q_dists_ = elemwise_dist_func(centroids, q)
    q_dists_ = np.sum(q_dists_, axis=-1)
    return np.asfortranarray(q_dists_)


class PQEncoder(object):

    def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
                 nsubvects=-1, elemwise_dist_func=dists_elemwise_sq):
        X = dataset.X
        self.elemwise_dist_func = elemwise_dist_func

        tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
                                     bits_per_subvect=bits_per_subvect,
                                     nsubvects=nsubvects)
        self.nsubvects, self.ncentroids, self.subvect_len = tmp

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

        # self.centroids, _, _ = quantize.learn_opq(
        #     X, ncodebooks=nsubvects, niters=0)

        self.centroids = _learn_centroids(X, self.ncentroids, self.nsubvects,
                                          self.subvect_len)

    def encode_X(self, X, **sink):
        idxs = quantize._encode_X_pq(X, codebooks=self.centroids,
                                     elemwise_dist_func=self.elemwise_dist_func)
        return idxs + self.offsets  # offsets let us index into raveled dists
        # return idxs

    def encode_q(self, q, **sink):
        return None  # we use fit_query() instead, so fail fast

    def dists_true(self, X, q):
        return np.sum(self.elemwise_dist_func(X, q), axis=-1)

    def fit_query(self, q, **sink):
        self.q_dists_ = _fit_opq_lut(q, centroids=self.centroids,
                                     elemwise_dist_func=self.elemwise_dist_func)

    # def dists_enc(self, X_enc, q_unused):
    #     dists = np.zeros(X_enc.shape[0])
    #     for i, row in enumerate(X_enc):
    #         for j, idx in enumerate(row):
    #             dists[i] += self.q_dists_[idx, j]
    #     return dists

    def dists_enc(self, X_enc, q_unused):
        # this line has each element of X_enc index into the flattened
        # version of q's distances to the centroids; we had to add
        # offsets to each col of X_enc above for this to work
        centroid_dists = self.q_dists_.T.ravel()[X_enc.ravel()]
        return np.sum(centroid_dists.reshape(X_enc.shape), axis=-1)


class OPQEncoder(PQEncoder):

    def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
                 nsubvects=-1, elemwise_dist_func=dists_elemwise_sq,
                 opq_iters=20):
        X = dataset.X
        self.elemwise_dist_func = elemwise_dist_func

        tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
                                     bits_per_subvect=bits_per_subvect,
                                     nsubvects=nsubvects)
        self.nsubvects, self.ncentroids, self.subvect_len = tmp

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

        self.centroids, _, self.R = quantize.learn_opq(
            X, ncodebooks=nsubvects, codebook_bits=bits_per_subvect,
            niters=opq_iters)

    def encode_X(self, X, **sink):
        X = quantize.opq_rotate(X, self.R)
        idxs = quantize._encode_X_pq(X, codebooks=self.centroids,
                                     elemwise_dist_func=self.elemwise_dist_func)

        # hmm...really small numbers just like in opq func
        # X_hat = quantize.reconstruct_X_pq(idxs, self.centroids)
        # errors = X - X_hat
        # err = np.mean(errors * errors) / np.var(X)
        # print "OPQ reconstruction mse / variance = {}".format(err)

        return idxs + self.offsets  # offsets let us index into raveled dists
        # return idxs

    def fit_query(self, q, **sink):
        # orig_norm = np.linalg.norm(q)
        q = quantize.opq_rotate(q, self.R).ravel()
        # print "norm before and after: ", orig_norm, np.linalg.norm(q)  # same

        # PQEncoder.fit_query(self, q)
        self.q_dists_ = _fit_opq_lut(q, centroids=self.centroids,
                                     elemwise_dist_func=self.elemwise_dist_func)

    # def dists_enc(self, X_enc, q_unused):
    #     dists = np.zeros(X_enc.shape[0])
    #     for i, row in enumerate(X_enc):
    #         for j, idx in enumerate(row):
    #             dists[i] += self.q_dists_[idx, j]
    #     return dists
        # return PQEncoder.dists_enc(self, X_enc, q_unused)

    # # ------------------------ DEBUG: just reconstruct points  # TODO rm
    # # yes, this works...so what's broken?
    # def encode_q(self, q, **sink):
    #     return q

    # def dists_enc(self, X_enc, q_unused):
    #     X_enc -= self.offsets
    #     X_hat = quantize.reconstruct_X_pq(X_enc, self.centroids)
    #     return dists_sq(X_hat, q_unused)


# ================================================================ Main

def eval_encoder(dataset, encoder, dist_func_true=None, dist_func_enc=None,
                 verbosity=1, plot=False):
    # X, queries, centroids, groups = dataset
    X = dataset.X
    queries = dataset.q
    centroids = dataset.centroids
    groups = dataset.groups

    if len(queries.shape) == 1:
        queries = [queries]

    # TODO: just encode dataset once, since we end up hitting each group
    # more than one time when using many queries

    if dist_func_true is None:
        dist_func_true = encoder.dists_true
    if dist_func_enc is None:
        dist_func_enc = encoder.dists_enc

    t0 = time.time()

    search_k = 20
    fracs = []
    for i, q in enumerate(queries):

        q_enc = encoder.encode_q(q)
        encoder.fit_query(q)

        all_true_dists = []
        all_enc_dists = []

        dists_to_centroids = dist_func_true(centroids, q)
        idxs = top_k_idxs(dists_to_centroids, search_k, smaller_better=True)
        for idx in idxs:
            X = groups[idx]
            if len(X) == 0:
                print "Warning: group at idx {} had no elements!".format(idx)
                continue

            true_dists = dist_func_true(X, q)
            all_true_dists.append(true_dists)

            X_enc = encoder.encode_X(X, idx=idx)

            enc_dists = dist_func_enc(X_enc, q_enc)
            all_enc_dists.append(enc_dists)

        all_true_dists = np.hstack(all_true_dists)
        all_enc_dists = np.hstack(all_enc_dists)

        # ------------------------ begin analysis / reporting code

        knn_idxs = top_k_idxs(all_true_dists, 10, smaller_better=True)
        cutoff = all_true_dists[knn_idxs[-1]]
        knn_bit_dists = all_enc_dists[knn_idxs]
        max_bit_dist = np.max(knn_bit_dists)
        num_below_max = np.sum(all_enc_dists <= max_bit_dist)
        frac_below_max = float(num_below_max) / len(all_enc_dists)
        fracs.append(frac_below_max)

        if plot and i < 3:  # at most 3 plots
            num_nn = min(10000, len(all_true_dists) - 1)
            xlim = [0, np.partition(all_true_dists, num_nn)[num_nn]]
            ylim = [0, np.partition(all_enc_dists, num_nn)[num_nn]]

            grid = sb.jointplot(x=all_true_dists, y=all_enc_dists,
                                xlim=xlim, ylim=ylim, joint_kws=dict(s=10))

            # hack to bully the sb JointGrid into plotting a vert line
            grid.x = [cutoff, cutoff]
            grid.y = ylim
            grid.plot_joint(plt.plot, color='r', linestyle='--')

            # also make it plot cutoff in terms of quantized dist
            grid.x = xlim
            grid.y = [max_bit_dist, max_bit_dist]
            grid.plot_joint(plt.plot, color='k', linestyle='--')

    if plot:
        plt.show()

    stats = np.array(fracs)
    t = time.time() - t0
    if verbosity > 0:
        print "mean, 90th pctile, std of fracs to search: " \
            "{:.3f}, {:.3f}, {:.3f} ({:.1f}s)".format(
                np.mean(stats), np.percentile(stats, q=90), np.std(stats), t)

    return fracs


def main():
    import doctest
    doctest.testmod()  # TODO uncomment after debug
    np.set_printoptions(precision=3)

    N, D = -1, -1

    # N = -1  # set this to not limit real datasets to first N entries
    # N = 10 * 1000
    # N = 50 * 1000
    N = 100 * 1000
    # N = 1000 * 1000
    # D = 96  # NOTE: this should be uncommented if using GLOVE + PQ
    # D = 32
    num_centroids = 256
    num_queries = 128
    # num_queries = 2

    dataset_func = functools.partial(load_dataset_and_groups,
                                     num_centroids=num_centroids, N=N, D=D,
                                     num_queries=num_queries)

    # dataset = dataset_func(datasets.Random.WALK, norm_len=True)
    # dataset = dataset_func(datasets.Random.UNIF, norm_len=True)
    # dataset = dataset_func(datasets.Random.GAUSS, norm_len=True)
    # dataset = dataset_func(datasets.Random.BLOBS, norm_len=True)
    # dataset = dataset_func(datasets.Glove.TEST_100, norm_mean=True)
    dataset = dataset_func(datasets.Sift1M.TEST_100, norm_mean=True)
    # dataset = dataset_func(datasets.Gist.TEST_100, norm_mean=True)

    # ncols = dataset.X.shape[1]
    # if ncols < D:
    #     padding = np.zeros((dataset.X.shape[0], D - ncols))
    #     dataset = dataset._replace(X=np.hstack((dataset.X, padding)))
    #     padding = np.zeros((dataset.X.shape[0], D - ncols))

    # print "------------------------ dbq l1"
    # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2, how=Quantizer.DBQ)
    # eval_encoder(dataset, encoder, dist_func_enc=dists_l1)
    # print "------------------------ dbq l2"
    # eval_encoder(dataset, encoder, dist_func_enc=dists_sq)

    # print "------------------------ manhattan l1"
    # # # note that we need shared_bins=False to mimic the paper
    # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2,
    #                                  how=Quantizer.KMEANS, shared_bins=False)
    # eval_encoder(dataset, encoder, dist_func_enc=dists_l1)
    # print "------------------------ manhattan l2"
    # eval_encoder(dataset, encoder, dist_func_enc=dists_sq)

    # print "------------------------ gauss l1"
    # # # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2,
    # #                                  how=Quantizer.GAUSS, shared_bins=False)
    # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2, how=Quantizer.GAUSS)
    # eval_encoder(dataset, encoder, dist_func_enc=dists_l1)
    # print "------------------------ gauss l2"
    # eval_encoder(dataset, encoder, dist_func_enc=dists_sq)

    # print "------------------------ pq l2, 8x10 bit centroids idxs"
    # # encoder = PQEncoder(dataset, code_bits=128, bits_per_subvect=8)
    # encoder = PQEncoder(dataset, nsubvects=8, bits_per_subvect=10)
    # eval_encoder(dataset, encoder)

    # print "------------------------ pq l2, 8x8 bit centroids idxs"
    # # encoder = PQEncoder(dataset, code_bits=128, bits_per_subvect=8)
    # encoder = PQEncoder(dataset, nsubvects=8, bits_per_subvect=8)
    # eval_encoder(dataset, encoder)

    # print "------------------------ pq l2, 16x6 bit centroid idxs"
    # encoder = PQEncoder(dataset, nsubvects=16, bits_per_subvect=6)
    # eval_encoder(dataset, encoder)

    print "------------------------ pq l2, 16x4 bit centroid idxs"
    encoder = PQEncoder(dataset, nsubvects=16, bits_per_subvect=4)
    eval_encoder(dataset, encoder)

    print "------------------------ opq l2, 16x4 bit centroid idxs"
    encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=5)
    # encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=0)
    # eval_encoder(dataset, encoder, plot=True)
    eval_encoder(dataset, encoder)

    # # print "------------------------ pca l1"
    # # encoder = PcaSketch(dataset.X, 32)    # mu, 90th on gist100? 0.023 0.040
    # encoder = PcaSketch(dataset.X, 64)      # mu, 90th on gist100? .011, .017
    # # encoder = PcaSketch(dataset.X, 128)   # mu, 90th on gist100? .007 .001
    # # encoder = PcaSketch(dataset.X, 256)   # mu, 90th on gist100? .005 .007
    # # encoder = PcaSketch(dataset.X, 512)   # mu, 90th on gist100? .004, .006
    # # eval_encoder(dataset, encoder, dist_func_enc=dists_l1)
    # print "------------------------ pca l2"  # much better than quantized
    # eval_encoder(dataset, encoder, dist_func_enc=dists_sq)

    # print "------------------------ quantile l1"  # yep, same performance as gauss
    # # # # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2,
    # # # #     how=Quantizer.QUANTILE, shared_bins=False)
    # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2,
    #                                  how=Quantizer.QUANTILE)
    # # # eval_encoder(dataset, encoder, dist_func_enc=dists_l1)
    # print "------------------------ quantile l2"
    # eval_encoder(dataset, encoder, dist_func_enc=dists_sq)

    # # print "------------------------ q lut l1"
    # inner_sketch = RandomIsoHash(dataset.X, 64)
    # encoder = BoltEncoder(dataset, inner_sketch=inner_sketch,
    #                       elemwise_dist_func=dists_elemwise_l1,
    #                       how=Quantizer.QUANTILE, shared_bins=True)
    # # eval_encoder(dataset, encoder)
    # print "------------------------ q lut l2"
    # encoder.elemwise_dist_func = dists_elemwise_sq
    # encoder = BoltEncoder(dataset, inner_sketch=inner_sketch,
    #                       elemwise_dist_func=dists_elemwise_sq,
    #                       how=Quantizer.QUANTILE, shared_bins=False)
    # # encoder = BoltEncoder(dataset, inner_sketch=pca_encoder,
    # #                       elemwise_dist_func=dists_elemwise_sq)
    # eval_encoder(dataset, encoder, dist_func_true=dists_sq)

    # ^ NOTE: seems to get much worse when we add top principal component
    # if inner_sketch is just raw pca
    #   -but is fine if we add a random rotation
    #   -this makes perfect sense; more quantization err in x if not isotropic

    # # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=8,  # 8 -> same as pca
    # # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=4,  # 4 -> little worse
    # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2,  # 2 ->
    #                                  how=Quantizer.QUANTILE, shared_bins=False)
    # q_encoding_func = create_q_encoding_func(dataset.X, encoder,
    #                                          dists_elemwise_l1)
    # t1 = time.time()
    # eval_embedding(dataset, encoder.transform, dist_func=dists_sq,
    #                q_encoding_func=q_encoding_func, bits_dist_func=dists_lut)
    # print "time to compute dists with lut: ", time.time() - t1

    # print "------------------------ q lut l2"
    # # t0 = time.time()
    # q_encoding_func = create_q_encoding_func(dataset.X, encoder,
    #                                          dists_elemwise_sq)
    # t1 = time.time()
    # # print "time to learn lut: ", t1 - t0

    # eval_embedding(dataset, encoder.transform, dist_func=dists_sq,
    #                q_encoding_func=q_encoding_func, bits_dist_func=dists_lut)

    # print "time to compute dists with lut: ", time.time() - t1

    # # print "===="
    # # TODO this should be encapsulated in the encoder and/or in a func
    # # that knows how to reach into encoders and just takes in X and the encoder
    # X_embed = encoder.inner_sketch.transform(dataset.X)
    # X_quant = encoder.quantizer.transform(X_embed)
    # learned_sq = learn_dists_func(X_embed, X_quant, base_dist_func=dists_sq)
    # learned_l1 = learn_dists_func(X_embed, X_quant, base_dist_func=dists_l1)

    # print "------------------------ quantile l1, learned dists"
    # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2,
    #                                  how=Quantizer.GAUSS, shared_bins=False)
    # # encoder = QuantizedRandomIsoHash(dataset.X, 64, nbits=2, how=Quantizer.QUANTILE)
    # eval_embedding(dataset, encoder.transform, dist_func=learned_l1)

    # print "------------------------ quantile l2, learned dists"
    # eval_embedding(dataset, encoder.transform, dist_func=learned_sq)


if __name__ == '__main__':
    main()
