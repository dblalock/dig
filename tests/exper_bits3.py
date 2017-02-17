#!/usr/bin/env python

import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats.stats import pearsonr
import seaborn as sb
import time

from collections import namedtuple
from sklearn.decomposition import TruncatedSVD

import datasets
import files
import product_quantize as pq
import pyience as pyn

from utils import dists_sq, kmeans, top_k_idxs
# from utils import kmeans, top_k_idxs

from joblib import Memory
_memory = Memory('.', verbose=0)

np.set_printoptions(precision=3)

SAVE_DIR = '../results'


# ================================================================ Distances

def dists_elemwise_sq(x, q):
    diffs = x - q
    return diffs * diffs


def dists_elemwise_l1(x, q):
    return np.abs(x - q)


def dists_elemwise_dot(x, q):
    return x * q


# ================================================================ Clustering

# @_memory.cache  # TODO use X_train and X_test separately, as well as truth idxs
def load_dataset_object(which_dataset, **load_dataset_kwargs):
    X_train, Q, X_test, true_nn = datasets.load_dataset(
        which_dataset, **load_dataset_kwargs)
    X = X_train  # TODO use train vs test
    assert Q.shape[-1] == X.shape[-1]

    # print "which_dataset: ", which_dataset
    # print "type(which_dataset): ", type(which_dataset)
    # print "type(which_dataset).__name__: ", which_dataset.__name__
    # print "which_dataset basename: ", files.basename(str(which_dataset), noext=True)

    if isinstance(which_dataset, str):
        name = files.basename(which_dataset, noext=True)
    else:
        name = which_dataset.__name__  # assumes which_dataset is a class

    return Dataset(Q, X, X_test, true_nn, name)


Dataset = namedtuple('Dataset', [
    'Q', 'X_train', 'X_test', 'true_nn', 'name'])


# ================================================================ Preproc

# ------------------------------------------------ Z-Normalization

class Normalizer(object):
    def __init__(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.std = np.max(self.stds)

    def znormalize(self, A):
        return (A - self.means) / self.std


# ================================================================ Encoder API

class EncoderMixin(object):

    def encode_X(self, X, **sink):  # needs to be overridden if no transform()
        return self.transform(X)

    def encode_q(self, q, **kwargs):
        return self.encode_X(q, **kwargs).ravel()

    def dists_true(self, X, q):
        return np.sum(self.elemwise_dist_func(X, q), axis=-1)
        # return np.sum(dists_sq(X, q), axis=-1)
        # return dists_sq(X, q)

    def fit_query(self, q, **sink):
        pass

    def dists_enc(self, X, q):
        return self.dists_true(X, q)


# ================================================================ Embedding

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
    if D % subvect_len:
        print "D, nsubvects, subvect_len = ", D, nsubvects, subvect_len
        assert D % subvect_len == 0  # TODO rm this constraint

    return nsubvects, ncentroids, subvect_len


def _fit_pq_lut(q, centroids, elemwise_dist_func):
    _, nsubvects, subvect_len = centroids.shape
    assert len(q) == nsubvects * subvect_len

    q = q.reshape((1, nsubvects, subvect_len))
    q_dists_ = elemwise_dist_func(centroids, q)
    q_dists_ = np.sum(q_dists_, axis=-1)

    # assert elemwise_dist_func == dists_elemwise_dot
    # assert np.array_equal(q_dists_, np.sum(centroids * q, axis=-1)) # TODO rm

    return np.asfortranarray(q_dists_)  # ncentroids, nsubvects, col-major


class PQEncoder(object):

    def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
                 nsubvects=-1, elemwise_dist_func=dists_elemwise_sq):
        X = dataset.X_train
        self.elemwise_dist_func = elemwise_dist_func

        tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
                                     bits_per_subvect=bits_per_subvect,
                                     nsubvects=nsubvects)
        self.nsubvects, self.ncentroids, self.subvect_len = tmp
        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

        self.centroids = _learn_centroids(X, self.ncentroids, self.nsubvects,
                                          self.subvect_len)

    def name(self):
        return "PQ_{}x{}b".format(self.nsubvects, self.code_bits)

    def params(self):
        return {'_algo': 'PQ', '_ncodebooks': self.nsubvects,
                '_code_bits': self.code_bits}

    def encode_X(self, X, **sink):
        idxs = pq._encode_X_pq(X, codebooks=self.centroids)
        return idxs + self.offsets  # offsets let us index into raveled dists

    def encode_q(self, q, **sink):
        return None  # we use fit_query() instead, so fail fast

    def dists_true(self, X, q):
        return np.sum(self.elemwise_dist_func(X, q), axis=-1)

    def fit_query(self, q, **sink):
        self.q_dists_ = _fit_pq_lut(q, centroids=self.centroids,
                                    elemwise_dist_func=self.elemwise_dist_func)

    # def dists_enc(self, X_enc, q_unused):
    #     dists = np.zeros(X_enc.shape[0])
    #     for i, row in enumerate(X_enc):
    #         for j, idx in enumerate(row):
    #             dists[i] += self.q_dists_[idx, j]
    #     return dists

    def dists_enc(self, X_enc, q_unused=None):
        # this line has each element of X_enc index into the flattened
        # version of q's distances to the centroids; we had to add
        # offsets to each col of X_enc above for this to work
        centroid_dists = self.q_dists_.T.ravel()[X_enc.ravel()]
        return np.sum(centroid_dists.reshape(X_enc.shape), axis=-1)


def _learn_best_quantization(luts):  # luts can be a bunch of vstacked luts
    best_loss = np.inf
    best_alpha = None
    best_floors = None
    best_scale_by = None
    for alpha in [.001, .002, .005, .01, .02, .05, .1]:
        alpha_pct = int(100 * alpha)
        # compute quantized luts this alpha would yield
        floors = np.percentile(luts, alpha_pct, axis=0)
        luts_offset = np.maximum(0, luts - floors)

        ceil = np.percentile(luts_offset, 100 - alpha_pct)
        scale_by = 255. / ceil
        luts_quantized = np.floor(luts_offset * scale_by).astype(np.int)
        luts_quantized = np.minimum(255, luts_quantized)

        # compute err
        luts_ideal = (luts - luts_offset) * scale_by
        diffs = luts_ideal - luts_quantized
        # diffs = (luts_offset * scale_by) - luts_quantized
        # print "diffs dtype: ", diffs.dtype  # float64
        loss = np.sum(diffs * diffs)

        # print "alpha = {}\t-> loss = {}".format(alpha, loss)
        # # yep, almost exactly alpha saturate in either direction
        # print "fraction of 0s, 255s = {}, {}".format(
        #     np.mean(luts_offset == 0), np.mean(luts_quantized == 255))

        if loss <= best_loss:
            best_loss = loss
            best_alpha = alpha
            best_floors = floors
            best_scale_by = scale_by

    return best_floors, best_scale_by, best_alpha


class OPQEncoder(PQEncoder):

    def __init__(self, dataset, code_bits=-1, bits_per_subvect=-1,
                 nsubvects=-1, elemwise_dist_func=dists_elemwise_sq,
                 opq_iters=20, quantize_lut=False, algo='OPQ', **opq_kwargs):
        X = dataset.X_train
        self.elemwise_dist_func = elemwise_dist_func
        self.quantize_lut = quantize_lut
        self.opq_iters = opq_iters
        self.algo = algo

        tmp = _parse_codebook_params(X.shape[1], code_bits=code_bits,
                                     bits_per_subvect=bits_per_subvect,
                                     nsubvects=nsubvects)
        self.nsubvects, self.ncentroids, self.subvect_len = tmp
        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.nsubvects, dtype=np.int) * self.ncentroids

        if self.algo == 'Bolt':
            self.centroids, _, self.rotations = pq.learn_bopq(
                X, ncodebooks=nsubvects, codebook_bits=bits_per_subvect,
                niters=opq_iters, **opq_kwargs)
        elif self.algo == 'OPQ':
            self.centroids, _, self.R = pq.learn_opq(
                X, ncodebooks=nsubvects, codebook_bits=bits_per_subvect,
                niters=opq_iters, **opq_kwargs)
        else:
            raise ValueError("argument algo must be one of {OPQ, Bolt}")

        # learn appropriate offsets and shared scale factor for quantization
        self.lut_offsets = np.zeros(self.nsubvects)
        self.order_idxs = np.arange(self.nsubvects, dtype=np.int)

        if self.quantize_lut:  # TODO put this logic in separate function
            num_rows = min(10*1000, len(X) / 2)
            _, queries = datasets.extract_random_rows(
                X[num_rows:], how_many=1000, remove_from_X=False)
            # print "sampled q:", np.mean(queries), np.std(queries), queries.shape, queries.dtype
            X = X[:num_rows]  # limit to first 10k rows of X for now

            if True:  # learn distros of entries in each lut
                print "learning quantization..."

                # compute luts for all the queries
                luts = [self._fit_query(q, quantize=False) for q in queries]
                luts = np.vstack(luts)
                assert luts.shape == (self.ncentroids * len(queries), self.nsubvects)

                self.lut_offsets, self.scale_by, _ = _learn_best_quantization(luts)

            if False:  # learn overall true dists
                # compute approximated nn distances between the queries
                # and the training set
                X_enc = self.encode_X(X)
                # num_neighbors = 10  # TODO accept as param
                num_neighbors = 10  # TODO accept as param
                all_enc_dists = np.empty((len(queries), num_neighbors))
                # all_true_dists = np.empty((len(queries), num_neighbors))
                for i, q in enumerate(queries):

                    self.fit_query(q, quantize=False)
                    dists_true = self.dists_true(X, q)
                    dists_enc = self.dists_enc(X_enc)
                    # if i == 0:
                    #     print "dists_true [:100]", dists_true[:100]
                    knn_idxs = top_k_idxs(dists_true, num_neighbors, smaller_better=True)
                    all_enc_dists[i] = dists_enc[knn_idxs]
                    # all_true_dists[i] = dists_true[knn_idxs]  # TODO rm

                max_enc_dists_for_queries = np.max(all_enc_dists, axis=1)
                min_enc_dists_for_queries = np.min(all_enc_dists, axis=1)

                lower_cutoff = np.percentile(min_enc_dists_for_queries, 5)
                upper_cutoff = np.percentile(max_enc_dists_for_queries, 95)

                self.lut_offsets += (lower_cutoff / float(self.nsubvects))
                self.scale_by = (1. / (upper_cutoff - lower_cutoff)) * 255

    def name(self):
        return "{}_{}x{}b_iters={}_quantize={}".format(
            self.algo, self.nsubvects, self.code_bits, self.opq_iters,
            int(self.quantize_lut))

    def params(self):
        return {'_algo': self.algo, '_ncodebooks': self.nsubvects,
                '_code_bits': self.code_bits, 'opq_iters': self.opq_iters,
                '_quantize': self.quantize_lut}

    def _fit_query(self, q, quantize=False):
        if self.algo == 'OPQ':
            qR = pq.opq_rotate(q, self.R).ravel()
        elif self.algo == 'Bolt':
            qR = pq.bopq_rotate(q, self.rotations).ravel()
        lut = _fit_pq_lut(qR, centroids=self.centroids,
                          elemwise_dist_func=self.elemwise_dist_func)

        if quantize:
            # print "quantizing query!"
            if False:  # roughly laplace distro, reaching all the way to 0
                ax = sb.distplot(lut.ravel(), hist=False, rug=True)
                ax.set_xlabel('Query dist to centroids (lut dist histogram)')
                ax.set_ylabel('Fraction of queries')
                plt.show()

            # return np.floor(lut * self.scale_by)
            # print "max lut value: np.max(lut * self.scale_by)"
            lut = np.maximum(0, lut - self.lut_offsets)
            lut = np.floor(lut * self.scale_by).astype(np.int)
            return np.minimum(lut, 255)
            # lut = np.floor((lut - self.dist_offset) * self.scale_by)
            # return np.maximum(0, lut)

        # print "not quantizing query!"
        return lut

    def encode_X(self, X, **sink):
        if self.algo == 'OPQ':
            X = pq.opq_rotate(X, self.R)
        elif self.algo == 'Bolt':
            X = pq.bopq_rotate(X, self.rotations)

        idxs = pq._encode_X_pq(X, codebooks=self.centroids)
        # idxs = pq._encode_X_pq(X, codebooks=self.centroids,
                               # elemwise_dist_func=self.elemwise_dist_func)

        return idxs + self.offsets  # offsets let us index into raveled dists

    def fit_query(self, q, quantize=True, **sink):
        quantize = quantize and self.quantize_lut
        self.q_dists_ = self._fit_query(q, quantize=quantize)

        if quantize:
            # print "min, max lut values: {}, {}".format(np.min(self.q_dists_),
            #     np.max(self.q_dists_))
            assert np.min(self.q_dists_) >= 0
            assert np.max(self.q_dists_) <= 255

        if False:
            _, axes = plt.subplots(3, figsize=(9, 11))
            sb.violinplot(data=self.q_dists_, inner="box", cut=0, ax=axes[0])
            axes[0].set_xlabel('Codebook')
            axes[0].set_ylabel('Distance to query')
            axes[0].set_ylim([0, np.max(self.q_dists_)])

            sb.heatmap(data=self.q_dists_, ax=axes[1], cbar=False, vmin=0)
            axes[1].set_xlabel('Codebook')
            axes[1].set_ylabel('Centroid')

            sb.distplot(self.q_dists_.ravel(), hist=False, rug=True, vertical=False, ax=axes[2])
            axes[2].set_xlabel('Centroid dist to query')
            axes[2].set_ylabel('Fraction of centroids')
            axes[2].set_xlim([0, np.max(self.q_dists_) + .5])

            # plot where the mean is
            mean_dist = np.mean(self.q_dists_)
            ylim = axes[2].get_ylim()
            axes[2].plot([mean_dist, mean_dist], ylim, 'r--')
            axes[2].set_ylim(ylim)

            plt.show()


# ================================================================ Main

def eval_encoder(dataset, encoder, dist_func_true=None, dist_func_enc=None,
                 eval_dists=True, verbosity=1, plot=False, smaller_better=True):

    X = dataset.X_test
    queries = dataset.Q
    true_nn = dataset.true_nn

    if true_nn is not None:
        print "eval encoder(): got true_nn with shape: ", true_nn.shape

    queries = queries[:200] # TODO rm for tables; fine for plots

    # print "queries.shape", queries.shape
    # queries = queries[:20, :] # TODO rm after debug
    print "queries.shape", queries.shape
    # assert smaller_better # TODO rm
    # assert true_nn is not None # TODO rm

    need_true_dists = eval_dists or plot or true_nn is None

    if len(queries.shape) == 1:
        queries = [queries]

    if dist_func_true is None:
        dist_func_true = encoder.dists_true
    if dist_func_enc is None:
        dist_func_enc = encoder.dists_enc

    t0 = time.time()

    # performance metrics
    RECALL_Rs = [1, 5, 10, 50, 100, 500, 1000]
    recall_counts = np.zeros(len(RECALL_Rs))
    # count_in_top1 = 0.
    # count_in_top10 = 0.
    # count_in_top100 = 0.
    # count_in_top1000 = 0.
    fracs_below_max = []
    if eval_dists:
        all_corrs = []
        all_rel_errs = []
        all_errs = []
        total_dist = 0.

    if need_true_dists:
        X = X[:10000]  # limit to 10k points because otherwise it takes forever
        queries = queries[:256, :]
        # queries = queries[:2, :]  # TODO rm

    assert not need_true_dists # TODO rm

    print "encoding X..."
    X_enc = encoder.encode_X(X)
    print "trying queries..."
    for i, q in enumerate(queries):

        if i % 50 == 0:
            print "trying query {}...".format(i)

        q_enc = encoder.encode_q(q)
        encoder.fit_query(q)
        if need_true_dists:
            all_true_dists = dist_func_true(X, q)

        #     # TODO rm
        #     dotprods = np.sum(X * q, axis=-1)
        #     assert np.array_equal(all_true_dists, dotprods)
        # assert need_true_dists  # TODO rm

        # # TODO rm
        # sq_dists = dists_sq(X, q)
        # assert np.array_equal(all_true_dists, sq_dists)

        all_enc_dists = dist_func_enc(X_enc, q_enc)

        # print "enc dists, true dists:"
        # print all_enc_dists[:10]
        # print all_true_dists[:10]
        # continue

        # ------------------------ begin analysis / reporting code

        # find true knn
        if need_true_dists:
            knn_idxs = top_k_idxs(all_true_dists, 10, smaller_better=smaller_better)
        else:
            knn_idxs = true_nn[i, :10]

        # compute fraction of points with enc dists as close as 10th nn
        knn_enc_dists = all_enc_dists[knn_idxs]
        if smaller_better:
            max_enc_dist = np.max(knn_enc_dists)
            num_below_max = np.sum(all_enc_dists <= max_enc_dist)
        else:
            max_enc_dist = np.min(knn_enc_dists)
            num_below_max = np.sum(all_enc_dists >= max_enc_dist)

        frac_below_max = float(num_below_max) / len(all_enc_dists)
        fracs_below_max.append(frac_below_max)

        # compute recall@R stats
        top_1000 = top_k_idxs(all_enc_dists, 1000, smaller_better=smaller_better)
        nn_idx = knn_idxs[0]
        for i, r in enumerate(RECALL_Rs):
            recall_counts[i] += nn_idx in top_1000[:r]
        # count_in_top1 += nn_idx == top_1000[0]
        # count_in_top10 += nn_idx in top_1000[:10]
        # count_in_top100 += nn_idx in top_1000[:100]
        # count_in_top1000 += nn_idx in top_1000

        # compute distortion in distances, quantified by corr and rel err
        if eval_dists:
            total_dist += np.sum(all_true_dists)
            corr, _ = pearsonr(all_enc_dists, all_true_dists)
            # corr = np.corrcoef(all_enc_dists.reshape((-1, 1)), all_true_dists.reshape((-1, 1)))
            # print "corr.shape", corr.shape
            # assert corr.shape == (,)
            all_corrs.append(corr)
            rel_errs = (all_enc_dists - all_true_dists) / all_true_dists
            all_rel_errs.append(rel_errs)
            all_errs.append(all_enc_dists - all_true_dists)
            # assert np.all(all_enc_dists >= 0)
            # assert np.all(all_true_dists >= 0)
            assert not np.any(np.isinf(all_enc_dists))
            assert not np.any(np.isnan(all_enc_dists))
            assert not np.any(np.isinf(all_true_dists))
            assert not np.any(np.isnan(all_true_dists))

            # mean_rel_errs.append(np.mean(rel_errs))
            # mean_rel_errs_sq.append(np.mean(rel_errs * rel_errs))

        if plot and i < 3:  # at most 3 plots
            num_nn = min(10000, len(all_true_dists) - 1)
            xlim = [0, np.partition(all_true_dists, num_nn)[num_nn]]
            ylim = [0, np.partition(all_enc_dists, num_nn)[num_nn]]

            grid = sb.jointplot(x=all_true_dists, y=all_enc_dists,
                                xlim=xlim, ylim=ylim, joint_kws=dict(s=10))

            # hack to bully the sb JointGrid into plotting a vert line
            cutoff = all_true_dists[knn_idxs[-1]]
            grid.x = [cutoff, cutoff]
            grid.y = ylim
            grid.plot_joint(plt.plot, color='r', linestyle='--')

            # also make it plot cutoff in terms of quantized dist
            grid.x = xlim
            grid.y = [max_enc_dist, max_enc_dist]
            grid.plot_joint(plt.plot, color='k', linestyle='--')

        # if i < 10:
        #     # print "true nn dists: ", all_true_dists[knn_idxs]
        #     print "approx nn dists: ", np.sort(knn_bit_dists)

    if plot:
        plt.show()

    t = time.time() - t0

    detailed_stats = []  # list of dicts
    stats = {}
    # stats['encoder'] = name_for_encoder(encoder)
    stats['X_rows'] = X.shape[0]
    stats['X_cols'] = X.shape[1]
    stats['nqueries'] = len(queries)
    stats['eval_time_secs'] = t
    # print "fracs_below_max", fracs_below_max
    stats['fracs_below_max_mean'] = np.mean(fracs_below_max)
    stats['fracs_below_max_std'] = np.std(fracs_below_max)
    stats['fracs_below_max_50th'] = np.median(fracs_below_max)
    stats['fracs_below_max_90th'] = np.percentile(fracs_below_max, q=90)
    for i, r in enumerate(RECALL_Rs):
        key = 'recall@{}'.format(r)
        val = float(recall_counts[i]) / len(queries)
        stats[key] = val
    # stats['recall@1'] = count_in_top1 / len(queries)
    # stats['recall@10'] = count_in_top10 / len(queries)
    # stats['recall@100'] = count_in_top100 / len(queries)
    # stats['recall@1000'] = count_in_top1000 / len(queries)
    if eval_dists:
        corrs = np.hstack(all_corrs)
        rel_errs = np.hstack(all_rel_errs)
        # assert np.all(rel_errs >= -1)
        # print "rel_errs[:20]", rel_errs[:20]
        # print "rel_errs inf idxs: ", np.where(np.isinf(rel_errs))[0]
        # print "rel_errs nan idxs: ", np.where(np.isnan(rel_errs))[0]
        rel_errs = rel_errs[~(np.isnan(rel_errs) + np.isinf(rel_errs))]
        errs = np.hstack(all_errs)
        # mean_dist = total_dist / (X.shape[1] * len(queries))
        # errs = errs[~(np.isnan(errs) + np.isinf(rel_errs))]
        # print "rel errs shape", rel_errs.shape
        # print "np.where"

        stats['corr_mean'] = np.mean(all_corrs)
        stats['corr_std'] = np.std(all_corrs)
        stats['mse_mean'] = np.mean(errs * errs)
        stats['mse_std'] = np.std(errs * errs)
        stats['rel_err_mean'] = np.mean(rel_errs)
        stats['rel_err_std'] = np.std(rel_errs)
        stats['rel_err_sq_mean'] = np.mean(rel_errs * rel_errs)
        stats['rel_err_sq_std'] = np.std(rel_errs * rel_errs)

        # sample some relative errs cuz all we need them for is plotting
        # confidence intervals
        np.random.shuffle(rel_errs)
        np.random.shuffle(errs)
        detailed_stats = [{'corr': all_corrs[i], 'rel_err': rel_errs[i],
                           'err': errs[i]} for i in range(len(corrs))]
                          #  'err': errs[i], 'trial': i}  # trial for sb.tsplot
                          # for i in range(len(corrs))]

        for d in detailed_stats:
            d.update(encoder_params(encoder))

    if verbosity > 0:
        print "------------------------ {}".format(name_for_encoder(encoder))
        keys = sorted(stats.keys())
        lines = ["{}: {}".format(k, stats[k]) for k in keys if isinstance(stats[k], str)]
        lines += ["{}: {:.4g}".format(k, stats[k]) for k in keys if not isinstance(stats[k], str)]
        print "\n".join(lines)

    stats.update(encoder_params(encoder))

    # if eval_dists:
    #     return stats, detailed_stats
    # return stats
    return stats, detailed_stats  # detailed_stats empty unless `eval_dists`


def name_for_encoder(encoder):
    # return encoder.name()
    try:
        return encoder.name()
    except AttributeError:
        return str(type(encoder))


def encoder_params(encoder):
    try:
        return encoder.params()
    except AttributeError:
        return {'algo': name_for_encoder(encoder)}


# @_memory.cache
def _experiment_one_dataset(which_dataset, eval_dists=False, dotprods=False,
                            save_dir=None):
    SAVE_DIR = save_dir if save_dir else '../results/acc/'

    elemwise_dist_func = dists_elemwise_dot if dotprods else dists_elemwise_sq
    smaller_better = not dotprods

    N, D = -1, -1
    # N = 10 * 1000 # TODO rm after debug

    num_queries = -1  # no effect for "real" datasets
    if isinstance(which_dataset, str):
        assert False # TODO rm; want to be certain using real queries for now
        print "WARNING: sampling queries from data file"
        num_queries = 128  # if just loading one file, need to sample queries
        # assert False
    # num_queries = 1
    # num_queries = 3
    # num_queries = 8
    norm_len = False
    norm_mean = True

    # max_ncodebooks = 32
    max_ncodebooks = 64

    dataset_func = functools.partial(load_dataset_object, N=N, D=D,
                                     num_queries=num_queries,
                                     norm_len=norm_len, norm_mean=norm_mean,
                                     D_multiple_of=max_ncodebooks)

    dataset = dataset_func(which_dataset)
    print "=== Using Dataset: {} ({}x{})".format(dataset.name, N, D)

    # print "dataset.Q.shape", dataset.Q.shape
    # print "dataset.X_train.shape", dataset.X_train.shape
    # import sys; sys.exit()

    dicts = []
    detailed_dicts = []
    nbytes_list = [8, 16, 32]
    # nbytes_list = [8, 32]
    # max_opq_iters = 5 # TODO uncomment below
    max_opq_iters = 20

    # ------------------------------------------------ Bolt
    # rotation_sizes = [8, 16, 32]
    rotation_sizes = [32]
    # rotation_sizes = [16]
    for nbytes in nbytes_list:
        # for opq_iters in (0, max_opq_iters):  # see how much rotations help
        for opq_iters in [0]:  # nope, no rotations
            rot_sizes = rotation_sizes if opq_iters > 0 else [16]
            for rot_sz in rot_sizes:
                nsubvects = nbytes * 2
                encoder = OPQEncoder(dataset, nsubvects=nsubvects,
                                     bits_per_subvect=4,
                                     opq_iters=opq_iters,
                                     R_sz=rot_sz,
                                     elemwise_dist_func=elemwise_dist_func,
                                     algo='Bolt', quantize_lut=True)
                stats, detailed_stats = eval_encoder(
                    dataset, encoder, eval_dists=eval_dists,
                    smaller_better=smaller_better)
                stats['rot_sz'] = rot_sz
                for d in detailed_dicts:
                    d['rot_sz'] = rot_sz
                dicts.append(stats)
                detailed_dicts += detailed_stats

    # ------------------------------------------------ PQ
    # for codebook_bits in [4]:
    # for codebook_bits in [8]:
    for codebook_bits in [4, 8]:
        for nbytes in nbytes_list:
            nsubvects = nbytes * (8 / codebook_bits)
            encoder = PQEncoder(dataset, nsubvects=nsubvects,
                                bits_per_subvect=codebook_bits,
                                elemwise_dist_func=elemwise_dist_func)
            stats, detailed_stats = eval_encoder(
                dataset, encoder, eval_dists=eval_dists,
                smaller_better=smaller_better)
            dicts.append(stats)
            detailed_dicts += detailed_stats

    # ------------------------------------------------ OPQ
    init = 'identity'
    opq_iters = max_opq_iters
    # opq_iters = 5
    # for codebook_bits in [4, 8]:
    # for codebook_bits in [4]:
    for codebook_bits in [8]:
        for nbytes in nbytes_list:
            nsubvects = nbytes * (8 / codebook_bits)
            encoder = OPQEncoder(dataset, nsubvects=nsubvects,
                                 bits_per_subvect=codebook_bits,
                                 opq_iters=opq_iters, init=init,
                                 elemwise_dist_func=elemwise_dist_func)
            stats, detailed_stats = eval_encoder(
                dataset, encoder, eval_dists=eval_dists,
                smaller_better=smaller_better)
            dicts.append(stats)
            detailed_dicts += detailed_stats

    for d in dicts:
        d['dataset'] = dataset.name
        d['norm_mean'] = norm_mean
    for d in detailed_dicts:
        d['dataset'] = dataset.name
        d['norm_mean'] = norm_mean

    savedir = os.path.join(SAVE_DIR, dataset.name)

    pyn.save_dicts_as_data_frame(dicts, savedir, name='summary')
    # also just save versions with timestamps to recover from clobbering
    pyn.save_dicts_as_data_frame(dicts, savedir, name='summary',
                                 timestamp=True)
    if eval_dists:
        pyn.save_dicts_as_data_frame(detailed_dicts, savedir, name='all_results')
        pyn.save_dicts_as_data_frame(detailed_dicts, savedir, name='all_results',
                                     timestamp=True)

    return dicts, detailed_dicts


def experiment(eval_dists=False, dotprods=False):

    # which_datasets = [datasets.Mnist]
    # which_datasets = [datasets.Sift1M.TEST_100]
    # which_datasets = [datasets.Sift1M]
    # which_datasets = [datasets.Convnet1M.TEST_100]
    # which_datasets = [datasets.Convnet1M]
    # which_datasets = [datasets.LabelMe]
    # which_datasets = [datasets.Convnet1M, datasets.Mnist]
    # which_datasets = [datasets.LabelMe, datasets.Sift1M,
    which_datasets = [datasets.Sift1M,
                      datasets.Convnet1M, datasets.Mnist]

    save_dir = '../results/acc_dotprods/' if dotprods else '../results/acc_l2_3'

    for which_dataset in which_datasets:
        _dicts, _details = _experiment_one_dataset(
            which_dataset, eval_dists=eval_dists, dotprods=dotprods,
            save_dir=save_dir)
            # save_dir='../results/tmp/')
            # save_dir='../results/acc/')
            # which_dataset, eval_dists=eval_dists, save_dir='../results/acc_l2/')


def main():
    import doctest
    doctest.testmod()
    np.set_printoptions(precision=3)

    # dataset = load_dataset_object(datasets.Convnet1M)
    # print dataset.name
    # return

    # print ">>>>>>>> eval-ing l2 dists"
    # experiment(eval_dists=True, dotprods=False)
    # print ">>>>>>>> eval-ing dot prods"
    # experiment(eval_dists=True, dotprods=True)
    print ">>>>>>>> eval-ing recall@R"
    experiment(eval_dists=False, dotprods=False)
    return

    N, D = -1, -1

    # N = -1  # set this to not limit real datasets to first N entries
    # N = 10 * 1000
    # N = 20 * 1000
    # N = 50 * 1000
    # N = 100 * 1000
    # N = 1000 * 1000
    # D = 954    # for 6 and 9 subvects on Gist
    # D = 125  # for 5 subvects on SIFT
    # D = 126  # for 6 (possibly with 9) subvects on SIFT
    # D = 120  # for 6 and 8 subvects on SIFT
    # D = 120  # for 6 and 9 subvects on SIFT
    # D = 96  # NOTE: this should be uncommented if using GLOVE + PQ
    # D = 480
    # D = 90
    # D = 80
    # D = 32
    num_queries = 128
    # num_queries = 1
    # num_queries = 3
    # num_queries = 8
    norm_len = False
    norm_mean = True

    dataset_func = functools.partial(load_dataset_object,
                                     N=N, D=D,
                                     num_queries=num_queries,
                                     norm_len=norm_len, norm_mean=norm_mean)

    # dataset = dataset_func(datasets.Random.WALK)
    # dataset = dataset_func(datasets.Random.UNIF)
    # dataset = dataset_func(datasets.Random.GAUSS)
    # dataset = dataset_func(datasets.Random.BLOBS)
    # dataset = dataset_func(datasets.Glove.TEST_100)
    # dataset = dataset_func(datasets.Sift1M.TEST_100)
    # dataset = dataset_func(datasets.Gist.TEST_100)
    dataset = dataset_func(datasets.Mnist)
    # dataset = dataset_func(datasets.Convnet1M.TEST_100)
    # dataset = dataset_func(datasets.Deep1M.TEST_100)
    # dataset = dataset_func(datasets.LabelMe)
    # dataset = dataset_func(datasets.Convnet1M)
    # dataset = dataset_func(datasets.Sift1M)
    print "=== Using Dataset: {} ({}x{})".format(dataset.name, N, D)

    # print "dataset.X_train.mean():", dataset.X_train.mean()
    # return

    # if False:
    #     nrows = 5
    #     X_test = dataset.X_test
    #     Q = dataset.Q

    #     np.set_printoptions(precision=6)
    #     print X_test[:5, :5]
    #     print Q[:5, :5]
    #     # return

    #     true_nn = dataset.true_nn
    #     for i in range(nrows):
    #         dists = dists_sq(X_test, Q[i, :])
    #         print np.argsort(dists, axis=-1)[:10]
    #         print true_nn[i, :10]

    #     import utils
    #     truth2 = utils.compute_true_knn(X_test, Q[:nrows])
    #     print
    #     print truth2[:, :10]

    #     # print utils.sq_dists_to_vectors(X_test[:8, :10], Q[:nrows, :10])
    #     # for i in range(nrows):
    #     #     print dists_sq(X_test[:8, :10], Q[i, :10])

    #     return

    dicts = []

    # print dataset.X_train.shape
    # print dataset.X_test.shape
    # print dataset.Q.shape
    # return

    # for quantize in (True, False):
    #     # encoder = OPQEncoder(dataset, nsubvects=8, bits_per_subvect=8,
    #     #                      opq_iters=5, quantize_lut=quantize)
    #     # eval_encoder(dataset, encoder)

    #     encoder = OPQEncoder(dataset, nsubvects=8, bits_per_subvect=8,
    #                          opq_iters=5, quantize_lut=quantize, algo='Bolt')
    #     eval_encoder(dataset, encoder)

    for nsubvects in (32, 64):
        for niters in (0, 5):  # how much to the rotations help?
            encoder = OPQEncoder(dataset, nsubvects=nsubvects, bits_per_subvect=4,
                                 opq_iters=niters, quantize_lut=True, algo='Bolt')
            eval_encoder(dataset, encoder)
    return

    # ------------------------------------------------ old stuff

    # ncols = dataset.X.shape[1]
    # if ncols < D:
    #     padding = np.zeros((dataset.X.shape[0], D - ncols))
    #     dataset = dataset._replace(X=np.hstack((dataset.X, padding)))
    #     padding = np.zeros((dataset.X.shape[0], D - ncols))

    # print "------------------------ pq l2, 16x4 bit centroid idxs"
    # encoder = PQEncoder(dataset, nsubvects=16, bits_per_subvect=4)
    # dicts.append(eval_encoder(dataset, encoder))

    # print "------------------------ opq l2, 16x4 bit centroid idxs"
    # encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=5)
    # dicts.append(eval_encoder(dataset, encoder))

    # print "------------------------ pq l2, 32x4 bit centroid idxs"
    # encoder = PQEncoder(dataset, nsubvects=32, bits_per_subvect=4)
    # dicts.append(eval_encoder(dataset, encoder))

    # print "------------------------ opq l2, 24x4 bit centroid idxs"
    # encoder = OPQEncoder(dataset, nsubvects=24, bits_per_subvect=4, opq_iters=5)
    # eval_encoder(dataset, encoder)

    # print "------------------------ opq l2, 16x4 bit centroid idxs"
    # encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=5)
    # # encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=0)
    # # eval_encoder(dataset, encoder, plot=True)
    # eval_encoder(dataset, encoder)

    # print "------------------------ opq l2, 16x4 bit centroid idxs, 8bit dists"
    # encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=5, quantize_lut=True)
    # # encoder = OPQEncoder(dataset, nsubvects=16, bits_per_subvect=4, opq_iters=0)
    # # eval_encoder(dataset, encoder, plot=True)
    # eval_encoder(dataset, encoder)

    # print "------------------------ pq l2, 8x8 bit centroids idxs"
    # encoder = PQEncoder(dataset, nsubvects=8, bits_per_subvect=8)
    # dicts.append(eval_encoder(dataset, encoder))

    print "------------------------ opq l2, 8x8 bit centroid idxs"
    encoder = OPQEncoder(dataset, nsubvects=8, bits_per_subvect=8, opq_iters=5)
    # eval_encoder(dataset, encoder, plot=True)
    dicts.append(eval_encoder(dataset, encoder))

    # print "------------------------ opq l2, 8x8 bit centroid idxs, 8bit dists"
    # encoder = OPQEncoder(dataset, nsubvects=8, bits_per_subvect=8, opq_iters=5, quantize_lut=True)
    # eval_encoder(dataset, encoder)

    # print "------------------------ pca l2"  # much better than quantized
    # encoder = PcaSketch(dataset.X, 64)      # mu, 90th on gist100? .011, .017
    # eval_encoder(dataset, encoder, dist_func_enc=dists_sq)

    for d in dicts:
        d['dataset'] = dataset.name
        d['norm_mean'] = norm_mean
        # d['num_queries'] = num_queries  # nah...ignored for real datasets
    pyn.save_dicts_as_data_frame(dicts, SAVE_DIR)


if __name__ == '__main__':
    main()
