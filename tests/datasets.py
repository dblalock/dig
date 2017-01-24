#!/bin/env python

# import functools
import os
import numpy as np

from joblib import Memory
_memory = Memory('.', verbose=1)

DATA_DIR = os.path.expanduser('~/Desktop/datasets/nn-search')
join = os.path.join


class Datasets:
    RAND_UNIF = 0
    RAND_GAUSS = 1
    RAND_WALK = 2
    GLOVE = 10
    GLOVE_100 = 11
    GLOVE_200 = 12
    SIFT = 20
    SIFT_100 = 21
    SIFT_200 = 22
    GIST = 30
    GIST_100 = 31
    GIST_200 = 32
    # GIST_QUERIES = 33

    RAND_DATASETS = [RAND_UNIF, RAND_GAUSS, RAND_WALK]
    FILE_DATASETS = [GLOVE, GLOVE_100, GLOVE_200,
                     SIFT, SIFT_100, SIFT_200,
                     GIST, GIST_100, GIST_200]


class Gist:
    TRAIN    = join(DATA_DIR, 'gist_train.npy')     # noqa
    TEST     = join(DATA_DIR, 'gist.npy')           # noqa
    TEST_100 = join(DATA_DIR, 'gist_100k.npy')      # noqa
    TEST_200 = join(DATA_DIR, 'gist_200k.npy')      # noqa
    queries  = join(DATA_DIR, 'gist_queries.npy')   # noqa
    truth    = join(DATA_DIR, 'gist_truth.npy')     # noqa


class Sift:
    TEST     = join(DATA_DIR, 'sift.txt')           # noqa
    TEST_100 = join(DATA_DIR, 'sift_100k.txt')      # noqa
    TEST_200 = join(DATA_DIR, 'sift_200k.txt')      # noqa


class Glove:
    TEST     = join(DATA_DIR, 'glove.txt')          # noqa
    TEST_100 = join(DATA_DIR, 'glove_100k.txt')     # noqa
    TEST_200 = join(DATA_DIR, 'glove_200k.txt')     # noqa

# class Paths:
#     GLOVE = join(DATA_DIR, 'glove.txt')
#     GLOVE_100 = join(DATA_DIR, 'glove_100k.txt')
#     GLOVE_200 = join(DATA_DIR, 'glove_200k.txt')
#     SIFT = join(DATA_DIR, 'sift.txt')
#     SIFT_100 = join(DATA_DIR, 'sift_100k.txt')
#     SIFT_200 = join(DATA_DIR, 'sift_200k.txt')

#     GIST = _Gist
#     # GIST = join(DATA_DIR, 'gist.npy')
#     # GIST_100     = join(DATA_DIR, 'gist_100k.npy')
#     # GIST_200     = join(DATA_DIR, 'gist_200k.npy')
#     # GIST_TRAIN   = join(DATA_DIR, 'gist_train.npy')
#     # GIST_QUERIES = join(DATA_DIR, 'gist_queries.npy')
#     # GIST_TRUTH   = join(DATA_DIR, 'gist_truth.npy')


@_memory.cache  # cache this more efficiently than as text
def cached_load_txt(*args, **kwargs):
    return np.loadtxt(*args, **kwargs)


# def cached_load_npy(*args, **kwargs):
#     return read_yael_vecs(*args, **kwargs)


def extract_random_rows(X, how_many):
    which_rows = np.random.randint(len(X), size=how_many)
    rows = np.copy(X[which_rows])
    mask = np.ones(len(X), dtype=np.bool)
    mask[which_rows] = False
    return X[mask], rows


def load_dataset(which_dataset, N=-1, D=-1, norm_mean=False, norm_len=False,
                 num_queries=1):
    if which_dataset == Datasets.RAND_UNIF:
        X = np.random.rand(N, D)
        q = np.random.rand(num_queries, D)
    elif which_dataset == Datasets.RAND_GAUSS:
        X = np.random.randn(N, D)
        q = np.random.randn(num_queries, D)
    elif which_dataset == Datasets.RAND_WALK:
        X = np.random.randn(N, D)
        X = np.cumsum(X, axis=1)
        q = np.random.randn(num_queries, D)
        q = np.cumsum(q, axis=-1)

    elif which_dataset == Datasets.GLOVE:
        X = cached_load_txt(Glove.TEST)
    elif which_dataset == Datasets.GLOVE_100:
        X = cached_load_txt(Glove.TEST_100)
    elif which_dataset == Datasets.GLOVE_200:
        X = cached_load_txt(Glove.TEST_200)

    elif which_dataset == Datasets.SIFT:
        X = cached_load_txt(Sift.TEST)
    elif which_dataset == Datasets.SIFT_100:
        X = cached_load_txt(Sift.TEST_100)
    elif which_dataset == Datasets.SIFT_200:
        X = cached_load_txt(Sift.TEST_200)

    elif which_dataset == Datasets.GIST:
        X = np.load(Gist.TEST)
    elif which_dataset == Datasets.GIST_100:
        X = np.load(Gist.TEST_100)
    elif which_dataset == Datasets.GIST_200:
        X = np.load(Gist.TEST_200)

    else:
        raise ValueError("unrecognized dataset {}".format(which_dataset))

    # if N > 0 and N < X.shape[0]:
    #     X = X[:N, :]

    if which_dataset in Datasets.FILE_DATASETS:
        X, q = extract_random_rows(X, how_many=num_queries)

        N = X.shape[0] if N < 1 else N
        D = X.shape[1] if D < 1 else D
        X = X[:N, :D]
        if len(q.shape) > 1:
            q = q[:, :D]
        else:
            q = q[:D]

    # if which_dataset in Datasets.FILE_DATASETS:
    #     if N > 0 and N < X.shape[0]:
    #         X = X[:N, :]
    #         X, q = extract_random_rows
    #     else:
    #         X = X[:-1, :]
    #         # q = np.copy(X[-1])
    #         # q = np.random.randn(X.shape[1])
    #         idx = np.random.randint(X.shape[1] - 1)
    #         q = np.copy(X[idx])
    #         q[64:] = X[idx + 1, 64:]  # mix 2 examples so differs from both
    #         print "start of q: ", q[:10]

    if norm_mean:
        means = np.mean(X, axis=0)
        X -= means
        q -= means
    if norm_len:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)

    return X.astype(np.float32), np.squeeze(q.astype(np.float32))


def read_yael_vecs(path, c_contiguous=True, limit_rows=-1, dtype=None):
    dim = np.fromfile(path, dtype=np.int32, count=2)[0]
    print "vector length = {}".format(dim)

    if dtype is None:
        if 'fvecs' in path:
            dtype = np.float32
        elif 'ivecs' in path:
            dtype = np.int32
        elif 'bvecs' in path:
            dtype = np.uint8
        else:
            raise ValueError("couldn't infer dtype from path {}".format(path))
    itemsize = np.dtype(dtype).itemsize

    assert dim > 0
    assert itemsize in (1, 2, 4)

    cols_for_dim = 4 // itemsize
    row_size_bytes = 4 + dim * itemsize
    row_size_elems = row_size_bytes // itemsize
    limit = int(limit_rows) * row_size_elems if limit_rows > 0 else -1

    fv = np.fromfile(path, dtype=dtype, count=limit)
    fv = fv.reshape((-1, row_size_elems))

    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + path)

    fv = fv[:, cols_for_dim:]

    if c_contiguous:
        fv = fv.copy()
    return fv


if __name__ == '__main__':
    pass

    # c------------------------ lean up gist
    #
    # path = Paths.DATA_DIR + '/gist/gist_base.fvecs'
    # path = Paths.DATA_DIR + '/gist/gist_learn.fvecs'
    # path = Paths.DATA_DIR + '/gist/gist_queries.fvecs'
    # path = Paths.DATA_DIR + '/gist/gist_groundtruth.ivecs'
    # out_path = Paths.GIST_100
    # out_path = Paths.GIST_200
    # out_path = Paths.GIST_TRAIN
    # out_path = Paths.GIST_QUERIES
    # out_path = Paths.GIST_TRUTH
    # X = read_yael_vecs(path)[:100000]
    # X = read_yael_vecs(path)[:200000]
    # X = read_yael_vecs(path)
    # X = read_yael_vecs(path, dtype=np.int32)
    # print X[:2]
    # print X.shape
    # np.save(out_path, X)

    # ------------------------ clean up sift1b (bigann)
    # data_dir = '/Volumes/MacHDD/datasets/sift1b/'
    # out_dir = '/Volumes/MacSSD_OS/Users/davis/Desktop/datasets/sift1b/'
    # path = data_dir + 'bigann_learn.bvecs'
    # path = data_dir + 'bigann_base.bvecs'
    # path = data_dir + 'queries.bvecs'
    # out_path = out_dir + 'big_ann_learn_1M.npy'
    # out_path = out_dir + 'big_ann_learn_10M.npy'
    # out_path = out_dir + 'sift_10M.npy'
    # out_path = out_dir + 'sift_queries.npy'
    # limit_rows = int(1e6)
    # limit_rows = int(10e6)
    # X = read_yael_vecs(path, limit_rows=limit_rows)
    # X = read_yael_vecs(path)
    # print X.shape
    # np.save(out_path, X)

    # truth_dir = data_dir + 'gnd/'
    # # truth_idxs_files = ['idx_1M', 'idx_10M', 'idx_100M']
    # truth_idxs_files = ['idx_1000M']
    # for f in truth_idxs_files:
    #     path = truth_dir + f + '.ivecs'
    #     out_path = out_dir + f + '.npy'
    #     print "unpacking {} to {}".format(path, out_path)
    #     X = read_yael_vecs(path)
    #     print X.shape
    #     np.save(out_path, X)

    # ------------------------ clean up sift1m
    data_dir = '/Volumes/MacHDD/datasets/sift1m/'
    out_dir = '/Volumes/MacSSD_OS/Users/davis/Desktop/datasets/sift1m/'
    for fname in os.listdir(data_dir):
        in_path = data_dir + fname
        out_path = out_dir + fname.split('.')[0] + '.npy'
        print "unpacking {} to {}".format(in_path, out_path)
        X = read_yael_vecs(in_path)
        print X.shape
        np.save(out_path, X)

    # ------------------------ clean up deep1b
    # data_dir = '/Volumes/MacHDD/datasets/deep1b/'
    # out_dir = '/Volumes/MacSSD_OS/Users/davis/Desktop/datasets/deep1b/'

    # # expected_cols = 96
    # # equivalent_elements_in_first_1M = int(1e6) * (1 + expected_cols)
    # arrays = []
    # # arrays.append(('deep1B_queries.fvecs', 'deep_queries.npy', -1))
    # # arrays.append(('deep1B_groundtruth.ivecs', 'deep_true_nn_idxs.npy', -1))
    # # arrays.append(('deep10M.fvecs', 'deep_1M.npy', 1e6))
    # arrays.append(('deep10M.fvecs', 'deep_10M.npy', -1))
    # for in_file, out_file, limit in arrays:
    #     in_path = data_dir + in_file
    #     out_path = out_dir + out_file
    #     X = read_yael_vecs(in_path, limit_rows=limit)
    #     print "unpacking {} to {}".format(in_path, out_path)
    #     print X.shape
    #     np.save(out_path, X)

    # ------------------------ clean up LabelMe
    #
    # >>> for k, v in d.iteritems():
    # ...     try:
    # ...             print k, v.shape
    # ...     except:
    # ...             pass
    # ...
    # gist (22019, 512)
    # img (32, 32, 3, 22019)
    # nmat (1000, 1000, 20)
    # __header__ param (1, 1)
    # __globals__ seg (32, 32, 22019)
    # names (1, 3597)
    # DistLM (22019, 22019)
    # __version__ ndxtrain (1, 20019)
    # ndxtest (1, 2000)
    #
    # okay, no idea what most of these are even with the readme...
    #
    # >>> np.save('labelme_train_idxs', d['ndxtrain']) # training data idxs
    # >>> np.save('labelme_test_idxs', d['ndxtest'])   # test data idxs
    # >>> np.save('labelme_all_gists', d['gist'])     # actual gist descriptors

