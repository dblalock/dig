//
//  nn_utils.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __NN_UTILS_HPP
#define __NN_UTILS_HPP

#include <vector>

#include "Dense"
#include "redsvd.hpp"

#include "array_utils.hpp"
#include "neighbors.hpp"

#include "Euclidean.hpp" // just for max_dist<>

// TODO just code brute force NeighborDB_L2 here using Eigen and give no
// craps about integrating with array_utils or anything

using std::vector;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::MatrixXi;
using Eigen::ArrayXi;

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;

// using ar::argsort;
using ar::map;
// using ar::at_idxs;

typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;

static const int16_t kInvalidIdx = -1;

static constexpr int kDefaultAlignBytes = EIGEN_DEFAULT_ALIGN_BYTES;
static_assert(kDefaultAlignBytes == 32, "EIGEN_DEFAULT_ALIGN_BYTES is not 32!");

namespace nn {

namespace {
using dist_t = typename Neighbor::dist_t;
static constexpr dist_t kMaxDist = dist::max_dist<dist_t>();
}
// wrapper subclasses to allow only specifying type
// template<class T, int Dim1=Dynamic, int Dim2=Dynamic>
// class MatrixX: public Matrix<T, Dynamic, Dynamic> {};

// template<class T, int Dim1=Dynamic, int Dim2=Dynamic>
// class RowMatrixX: public Matrix<T, Dim1, Dim2, RowMajor> {};

// ------------------------------------------------ Preprocessing

template<class T, class IdxT>
inline void reorder_query(const T* RESTRICT q, const IdxT* order, idx_t len,
	T* RESTRICT out)
{
    for(idx_t i = 0; i < len; i++) {
        out[i] = q[order[i]];
    }
}
template<class VectorT1, class VectorT2>
inline void reorder_query(const VectorT1& q, const VectorT2& order_idxs,
    typename VectorT1::Scalar* out)
{
    reorder_query(q.data(), order_idxs.data(), order_idxs.size(), out);
}
template<class RowMatrixT, class VectorT, class RowMatrixT2>
inline void reorder_query_batch(const RowMatrixT& queries, const VectorT& order_idxs,
    const RowMatrixT2& out)
{
    for (idx_t i = 0; i < queries.rows(); i++) {
        reorder_query(queries.row(i), order_idxs, out.row(i).data());
    }
}

template<class IdxT=int32_t, class MatrixT=char>
inline std::vector<IdxT> order_col_variance(const MatrixT& X) {
	using Scalar = typename MatrixT::Scalar;
	// using RowMatrixT = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::Rowmajor>;

    auto means = X.colwise().mean();
    auto Xnorm = (X.rowwise() - means).eval();
	Scalar nrows = static_cast<Scalar>(X.rows());
    MatrixT variances = Xnorm.colwise().squaredNorm() / nrows;

    std::vector<IdxT> ret(X.cols());
    ar::argsort(variances.data(), X.cols(), ret.data(), false); // descending
    return ret;
}

// ------------------------------------------------ Alignment

template<class T, int AlignBytes, class IntT>
inline constexpr IntT aligned_length(IntT ncols) {
	static_assert(AlignBytes >= 0, "AlignBytes must be nonnegative");
	assert(ncols > 0);
	if (AlignBytes == 0) { return ncols; }

    int16_t align_elements = AlignBytes / sizeof(T);
    int16_t remaindr = ncols % align_elements;
    if (remaindr > 0) {
        ncols += align_elements - remaindr;
    }
    return ncols;
}

// helper struct to get Eigen::Map<> MapOptions based on alignment in bytes
template<int AlignBytes> struct _AlignHelper {};
template<> struct _AlignHelper<0> {
    static constexpr int AlignmentType = Eigen::Unaligned;
};
template<> struct _AlignHelper<kDefaultAlignBytes> {
    static constexpr int AlignmentType = Eigen::Aligned;
};

template<class T, int AlignBytes>
static inline T* aligned_alloc(size_t n) {
    static_assert(AlignBytes == 0 || AlignBytes == kDefaultAlignBytes,
        "Only AlignBytes values of 0 and kDefaultAlignBytes are supported!");
    static_assert(EIGEN_DEFAULT_ALIGN_BYTES == AlignBytes,
        "AlignBytes does not match Eigen default align bytes!");
    if (AlignBytes == kDefaultAlignBytes) {
        return Eigen::aligned_allocator<T>{}.allocate(n);
    } else {
        return new T[n];
    }
}
template<class T, int AlignBytes>
static inline void aligned_free(T* p) {
    static_assert(AlignBytes == 0 || AlignBytes == kDefaultAlignBytes,
        "Only AlignBytes values of 0 and kDefaultAlignBytes are supported!");
    if (AlignBytes == kDefaultAlignBytes) {
        Eigen::aligned_allocator<T>{}.deallocate(p, 0); // 0 unused
    } else {
        delete[] p;
    }
}


// ------------------------------------------------ neighbor munging

template<template<class...> class Container>
inline void sort_neighbors_ascending_distance(Container<Neighbor>& neighbors) {
    std::sort(std::begin(neighbors), std::end(neighbors),
        [](const Neighbor& a, const Neighbor& b) -> bool
        {
            return a.dist < b.dist;
        }
    );
}

template<template<class...> class Container>
inline void sort_neighbors_ascending_idx(Container<Neighbor>& neighbors) {
    std::sort(std::begin(neighbors), std::end(neighbors),
        [](const Neighbor& a, const Neighbor& b) -> bool
        {
            return a.idx < b.idx;
        }
    );
}

/** given a sorted collection of the best neighbors found so far, (potentially)
 * inserts a new neighbor into the list such that the sorting is preserved;
 * assumes the neighbors container contains only valid neighbors and is sorted
 *
 * Returns the distance to the last (farthest) neighbor after possible insertion
 */
template<template<class...> class Container>
inline dist_t maybe_insert_neighbor(
	Container<Neighbor>& neighbors_bsf, Neighbor newNeighbor)
{
    assert(neighbors_bsf.size() > 0);
	size_t len = neighbors_bsf.size();
    size_t i = len - 1;
    auto dist = newNeighbor.dist;

    if (dist < neighbors_bsf[i].dist) {
        neighbors_bsf[i] = newNeighbor;
    }

    while (i > 0 && neighbors_bsf[i-1].dist > dist) {
        // swap new and previous neighbor
        Neighbor tmp = neighbors_bsf[i-1];

        neighbors_bsf[i-1] = neighbors_bsf[i];
        neighbors_bsf[i] = tmp;
        i--;
    }
    return neighbors_bsf[len - 1].dist;
}
template<template<class...> class Container>
inline dist_t maybe_insert_neighbor(Container<Neighbor>& neighbors_bsf,
    double dist, typename Neighbor::idx_t idx)
{
	return maybe_insert_neighbor(neighbors_bsf, Neighbor{idx, dist});
}

template<template<class...> class Container,
    template<class...> class Container2>
inline dist_t maybe_insert_neighbors(Container<Neighbor>& neighbors_bsf,
    const Container2<Neighbor>& potential_neighbors)
{
    dist_t d_bsf = kMaxDist;
    for (auto& n : potential_neighbors) {
        d_bsf = maybe_insert_neighbor(neighbors_bsf, n);
    }
    return d_bsf;
}


template<class T>
inline vector<Neighbor> knn_from_dists(const T* dists, size_t len, size_t k) {
    assert(k > 0);
    assert(len > 0);
    k = ar::min(k, len);
    vector<Neighbor> ret(k); // warning: populates it with k 0s
    for (idx_t i = 0; i < k; i++) {
		ret[i] = Neighbor{i, dists[i]};
    }
    sort_neighbors_ascending_distance(ret);
    for (idx_t i = k; i < len; i++) {
        maybe_insert_neighbor(ret, dists[i], i);
    }
    return ret;
}

template<class T, class R>
inline vector<Neighbor> neighbors_in_radius(const T* dists, size_t len, R radius_sq) {
    vector<Neighbor> neighbors;
    for (idx_t i = 0; i < len; i++) {
        auto dist = dists[i];
        if (dists[i] < radius_sq) {
			neighbors.emplace_back(Neighbor{static_cast<dist_t>(dist), i});
        }
    }
    return neighbors;
}

inline vector<vector<idx_t> > idxs_from_nested_neighbors(
    const vector<vector<Neighbor> >& neighbors)
{
    return map([](const vector<Neighbor>& inner_neighbors) -> std::vector<idx_t> {
        return map([](const Neighbor& n) -> idx_t {
            return n.idx;
        }, inner_neighbors);
    }, neighbors);
}

} // namespace nn
#endif // __NN_UTILS_HPP
