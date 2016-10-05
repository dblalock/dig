//
//  euclidean.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __EUCLIDEAN_HPP
#define __EUCLIDEAN_HPP

#include <sys/types.h>

#include "Dense"

#include "macros.hpp"
#include "eigen_utils.hpp"

namespace dist {

// TODO only define kMaxDist in one place (also in neighbors.hpp)
static constexpr float kMaxDist = std::numeric_limits<float>::max();
using idx_t = int64_t;
using _infer = char;

// ------------------------------------------------ brute force distances

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=_infer,
    class VectorT=_infer>
inline VectorT squared_dists_to_vector(const MatrixT& X, const VectorT& v) {
    return (X.rowwise() - v).rowwise().squaredNorm();
}
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=_infer,
    class VectorT=_infer>
inline VectorT dists_to_vector(const MatrixT& X, const VectorT& v) {
    return squared_dists_to_vector(X, v).array().sqrt().matrix();
}

/** compute distances between rows of X and rows of V */
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=_infer,
    class MatrixT2=_infer, class VectorT=_infer>
inline auto squared_dists_to_vectors(const MatrixT1& X, const MatrixT2& V,
    VectorT rowSquaredNorms) ->
        typename mat_product_traits<MatrixT1, MatrixT2, RetStorageOrder>::type
{
    // create a matrix of appropriate type; this way dists for each vector
    // can be contiguous (if ColumnMajor storage)
    typename mat_product_traits<MatrixT1, MatrixT2, RetStorageOrder>::type
        dists = -2. * (X * V.transpose());
    auto colSquaredNorms = V.rowwise().squaredNorm().eval();
    auto colSquaredNormsAsRow = colSquaredNorms.transpose().eval();
    dists.colwise() += rowSquaredNorms;
    dists.rowwise() += colSquaredNormsAsRow;
    // TODO profile and see if for loop that touches each element once is faster

    return dists;
}
/** compute distances between rows of X and rows of V */
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=_infer,
    class MatrixT2=_infer>
inline auto squared_dists_to_vectors(const MatrixT1& X, const MatrixT2& V) ->
    typename mat_product_traits<MatrixT1, MatrixT2, RetStorageOrder>::type
{
    Eigen::Matrix<typename MatrixT1::Scalar, Eigen::Dynamic, 1> rowSquaredNorms =
        X.rowwise().squaredNorm().eval();
    return squared_dists_to_vectors(X, V, rowSquaredNorms);
}

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT>
inline MatrixT dists_to_vectors(const MatrixT& X, const MatrixT& V) {
    return squared_dists_to_vectors(X, V).array().sqrt().matrix();
}

template<class VectorT1, class VectorT2>
auto dist_sq(const VectorT1& x, const VectorT2& y)
    -> decltype((x - y).squaredNorm())
{
    return (x - y).squaredNorm();
}

// ------------------------------------------------ early abandoning distances
namespace abandon {

// ================================ distance functions

//template<VectorT>
//	auto to_vect(VectorT x) {
//		return
//	}

// TODO this really should be in some shared global file
//template <typename T>
//class TTypes {
//	typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
//	typedef const Eigen::Matrix<T, Eigen::Dynamic, 1>& VectorCRef;
//};

//template<int BlockSize=8, class T, class dist_t>
//dist_t dist_sq(typename TTypes<T>::VectorCRef x,
//			   typename TTypes<T>::VectorCRef y, dist_t thresh=kMaxDist) {


template<int BlockSize=8, class VectorT1, class VectorT2, class dist_t>
dist_t dist_sq(const VectorT1& x, const VectorT2& y, dist_t thresh=kMaxDist) {
    // typedef decltype(x.data()[0] * y.data()[0]) dist_t;

//	STATIC_ASSERT_SAME_SHAPE(VectorT1, VectorT2);

//	PRINT_STATIC_TYPE(y);
//	PRINT_STATIC_TYPE(x);

//	Eigen::Matrix<typename VectorT1::Scalar, 1, -1> x_ = x;
//	Eigen::Matrix<typename VectorT2::Scalar, 1, -1> y_ = y;

    auto n = x.size();
    dist_t dist = 0;
    int32_t i = 0;
    for (; i < (n - BlockSize) ; i += BlockSize) {
//		auto tmp_x = x_.segment<BlockSize>(i); // ref to non-static member func must be called
//		auto tmp_x = x_.segment(i, 8); // okay
//		auto tmp_y = y_.segment<BlockSize>(i, BlockSize);

//		auto tmp_x = x.segment(i, 8); // okay
//		auto tmp_y = y.segment<BlockSize>(i, BlockSize);
//		auto tmp = tmp_x - tmp_y;
//		auto tmp = (x.segment<BlockSize>(i, BlockSize) - y.segment<BlockSize>(i, BlockSize));
//		dist += tmp.squaredNorm();

//        dist += (x_.segment<BlockSize>(i, BlockSize) - y_.segment<BlockSize>(i, BlockSize)).squaredNorm();

		// TODO get static segment lengths to work
		dist += (x.segment(i, BlockSize) - y.segment(i, BlockSize)).squaredNorm();
        if (dist > thresh) {
            return dist;
        }
    }
    for (; i < n; i++) {
		auto diff = x(i) - y(i);
        dist += diff * diff;
    }
	return dist;
}

template<int BlockSize=8, class VectorT1, class VectorT2, class IdxVectorT,
    class dist_t>
dist_t dist_sq_order(const VectorT1& x, const VectorT2& y,
    const IdxVectorT& order, dist_t thresh)
{
    auto n = x.size();
    int32_t numBlocks = order.size();
    assert(numBlocks == n / BlockSize);

    dist_t dist = 0;
    for (int32_t i = 0; i < numBlocks; i++) {
        auto idx = order(i);
        dist += (x.segment<BlockSize>(idx) - y.segment<BlockSize>(idx)).squaredNorm();
        if (dist > thresh) {
            return dist;
        }
    }
    int32_t finalBlockEnd = n - (n % BlockSize);
    for (int32_t i = finalBlockEnd; i < n; i++) {
        auto diff = x(i) - y(i);
        dist += diff * diff;
    }
}

template<int BlockSize=8, class VectorT1, class VectorT2, class IdxVectorT,
    class dist_t>
dist_t dist_sq_order_presorted(const VectorT1& x_sorted, const VectorT2& y,
    const IdxVectorT& order, dist_t thresh)
{
    auto n = x_sorted.size();
    int32_t numBlocks = order.size();
    assert(numBlocks == n / BlockSize);

    dist_t dist = 0;
    for (int32_t i = 0; i < numBlocks; i++) {
        auto idx = order(i);
        dist += (x_sorted.segment<BlockSize>(i * BlockSize) - y.segment<BlockSize>(idx)).squaredNorm();
        if (dist > thresh) {
            return dist;
        }
    }
    int32_t finalBlockEnd = n - (n % BlockSize);
    for (int32_t i = finalBlockEnd; i < n; i++) {
        auto diff = x_sorted(i) - y(i);
        dist += diff * diff;
    }
}

// ================================ query preprocessing


// TODO funcs here to pad query + data to multiple of BlockSize
//     just need L2IndexAbandon to work, cuz thats what Ill actually use
//         -actually, prolly wrap in L2IndexCascaded


// NOTE: query_out and query must be padded to multiple of block size
template<int BlockSize=8, class VectorT, class IdxT>
void create_ordered_query(const VectorT& query, const VectorT& means,
    typename VectorT::Scalar * query_out, IdxT* order_out)
{
    typedef typename VectorT::Scalar data_t;
    typedef std::pair<IdxT, data_t> pair;

    // XXX query must be padded to be a multiple of BlockSize
    IdxT len = query.size();
    IdxT num_blocks = len / BlockSize;

    // compute squared distances to mean for each block, reusing query storage
    //
    // TODO sum these dists and return it so we can get LB based on centroid
    //
    idx_t idx = 0;
    for (idx_t i = 0; i < len; i++) {
        query_out(i) = (query.segment<BlockSize>(idx) - means.segment<BlockSize>(idx)).squaredNorm();
        idx += BlockSize;
    }

    // pair idx with value // TODO pass in storage for this
    // assert((BlockSize * sizeof(pair)) <= len * sizeof(IdxT)); // will it fit?
    auto pairs = std::unique_ptr<pair>(new pair[num_blocks]);
    for (idx_t block_num = 0; block_num < num_blocks; block_num++) {
        pairs[block_num] = pair(block_num, query_out(block_num)); // sum stored in query
    }
    // auto pairs = mapi([](length_t i, data_t x) {
    //  return pair(i, x);
    // }, data, len);

    // sort pairs by summed distance
    std::sort(pairs.get(), pairs.get() + len,
    [](const pair& lhs, const pair& rhs) {
        return lhs.second > rhs.second;
    });

    // unpack pairs into sorted query and order; order is just an order
    // for the blocks, so sorted query has groups of BlockSize consecutive
    // idxs reordered
    idx = 0;
    for (IdxT i = 0; i < num_blocks; i++) {
        order_out[i] = pairs[i].first;
        for (IdxT j = 0; j < BlockSize; j++) {
            query_out[idx] = pairs[i].second + j;
            idx++;
        }
    }
}

} // namespace abandon

} // namespace dist
#endif // __EUCLIDEAN_HPP
