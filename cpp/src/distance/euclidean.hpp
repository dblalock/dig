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

template<class T>
constexpr T max_dist() { return std::numeric_limits<T>::max(); }

// TODO only define kMaxDist in one place (also in neighbors.hpp)
static constexpr float kMaxDist = std::numeric_limits<float>::max();
using idx_t = int64_t;
using _infer = char;

// ------------------------------------------------ brute force distances

// ------------------------ dists to batch of vectors, with row norms

/** compute distances between rows of X and rows of V */
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=_infer,
    class MatrixT2=_infer, class VectorT=_infer>
inline auto squared_dists_to_vectors(const MatrixT1& X, const MatrixT2& V,
    const VectorT& rowSquaredNorms)
    -> typename mat_product_traits<MatrixT1, MatrixT2, RetStorageOrder>::type
{
    // create a matrix of appropriate type; this way dists for each vector
    // can be contiguous (if ColumnMajor storage)
    typename mat_product_traits<MatrixT1, MatrixT2, RetStorageOrder>::type
        dists = -2 * (X * V.transpose());
    auto colSquaredNorms = V.rowwise().squaredNorm().eval();
    auto colSquaredNormsAsRow = colSquaredNorms.transpose().eval();
    dists.colwise() += rowSquaredNorms;
    dists.rowwise() += colSquaredNormsAsRow;
    // TODO profile and see if for loop that touches each element once is faster

    return dists;
}

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=_infer,
    class MatrixT2=_infer, class VectorT=_infer>
inline auto dists_to_vectors(const MatrixT1& X, const MatrixT2& V,
    const VectorT& rowSquaredNorms)
    -> decltype(squared_dists_to_vectors(X, V, rowSquaredNorms))
{
    return squared_dists_to_vectors(X, V, rowSquaredNorms).array().sqrt().matrix();
}

// ------------------------ dists to batch of vectors, no row norms

/** compute distances between rows of X and rows of V */
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=_infer,
    class MatrixT2=_infer>
inline auto squared_dists_to_vectors(const MatrixT1& X, const MatrixT2& V)
    -> typename mat_product_traits<MatrixT1, MatrixT2, RetStorageOrder>::type
{
    Eigen::Matrix<typename MatrixT1::Scalar, Eigen::Dynamic, 1> rowSquaredNorms =
        X.rowwise().squaredNorm().eval();
    return squared_dists_to_vectors(X, V, rowSquaredNorms);
}

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT>
inline MatrixT dists_to_vectors(const MatrixT& X, const MatrixT& V) {
    return squared_dists_to_vectors(X, V).array().sqrt().matrix();
}

// ------------------------ dist to single vector, with row norms
// Note that this is after the batch functions because it calls them

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=_infer,
    class VectorT=_infer, class VectorT2=_infer>
inline auto squared_dists_to_vector(const MatrixT& X, const VectorT& v,
    const VectorT2& rowSquaredNorms)
	-> typename mat_product_traits<MatrixT, VectorT>::type
	// -> decltype(squared_dists_to_vectors(X, VectorT::IsRowMajor ? v : v.transpose(), rowSquaredNorms))
    //-> typename mat_traits<VectorT>::VectorT
{
    if (VectorT::IsRowMajor) {
        return squared_dists_to_vectors(X, v, rowSquaredNorms);
    } else {
        return squared_dists_to_vectors(X, v.transpose(), rowSquaredNorms);
    }
}
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=_infer,
    class VectorT=_infer, class VectorT2=_infer>
inline auto dists_to_vector(const MatrixT& X, const VectorT& v,
    const VectorT2& rowSquaredNorms)
    -> decltype(squared_dists_to_vector(X, v, rowSquaredNorms))
{
    return squared_dists_to_vector(X, v).array().sqrt().matrix();
}

// ------------------------ dist to single vector, no row norms

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=_infer,
    class VectorT=_infer>
inline auto squared_dists_to_vector(const MatrixT& X, const VectorT& v)
    -> typename mat_traits<VectorT>::VectorT
{
    return (X.rowwise() - v).rowwise().squaredNorm();
}
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=_infer,
    class VectorT=_infer>
inline auto dists_to_vector(const MatrixT& X, const VectorT& v)
    -> decltype(squared_dists_to_vector(X, v))
{
    return squared_dists_to_vector(X, v).array().sqrt().matrix();
}


// ------------------------------------------------ simple vectorized distances
namespace simple {

template<class VectorT1, class VectorT2>
inline auto dist_sq(const VectorT1& x, const VectorT2& y)
	-> typename mat_product_traits<VectorT1, VectorT2>::Scalar
{
	if (VectorT1::IsRowMajor != VectorT2::IsRowMajor) {
        return (x - y.transpose()).squaredNorm();
    }
	assert(x.rows() == y.rows());
	assert(x.cols() == y.cols());
    return (x - y).squaredNorm();
}

} // namespace simple

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


template<int BlockSize=32, class VectorT1, class VectorT2,
	class dist_t=typename VectorT1::Scalar>
inline dist_t dist_sq(const VectorT1& x, const VectorT2& y,
    dist_t thresh=kMaxDist, int64_t* num_abandons=nullptr,
    int64_t* abandon_iters=nullptr)
{
    using Scalar = typename VectorT1::Scalar;
    // using FixedVectorT = Eigen::Matrix<Scalar, 1, BlockSize, Eigen::RowMajor>;
    // using FixedVectorT = Eigen::Matrix<Scalar, BlockSize, 1>;
	using FixedVectorT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using FixedMapT = Eigen::Map<const FixedVectorT, Eigen::Aligned>;

    // using DynamicVectorT = typename scalar_traits<Scalar>::RowVectorT;
    using DynamicVectorT = typename scalar_traits<Scalar>::ColVectorT;
    using DynamicMapT = Eigen::Map<const DynamicVectorT, Eigen::Aligned>;
    // using DynamicMapT = Eigen::Map<const DynamicVectorT>;

	static_assert(std::is_same<Scalar, typename VectorT2::Scalar>::value,
		"Received vectors of different types!");
	static_assert((BlockSize * sizeof(Scalar)) % 32 == 0, // TODO use kDefaultAlignBytes
		"BlockSize * sizeof(Scalar) must be a multiple of default Alignment");
    assert(x.size() == y.size());

    // return (x - y).squaredNorm(); // significantly faster than anything below

    // SELF: see if we can match eigen performance if we take out the abandoning
    //     -and if that doesn't do it, try doing mul, then add, instead of fma


    // same speed as above under Ofast
    // return (x.segment(0, x.size()) - y.segment(0, x.size())).squaredNorm();

    // Same as orig squaredNorm, or maybe like 5% slower
    // DynamicMapT x_map0(x.data(), x.size());
    // DynamicMapT y_map0(y.data(), x.size());
    // dist_t dist1 = (x_map0 - y_map0).squaredNorm();
    // return dist1;

    // // maybe 10% slower than orig squaredNorm()
    // dist_t dist2 = 0;
    // FixedMapT x_map1(x.data(), BlockSize);
    // FixedMapT y_map1(y.data(), BlockSize);
    // dist2 += (x_map1 - y_map1).squaredNorm();
    // DynamicMapT x_map2(x.data() + BlockSize, x.size() - BlockSize);
    // DynamicMapT y_map2(y.data() + BlockSize, x.size() - BlockSize);
    // dist2 += (x_map2 - y_map2).squaredNorm();
    // return dist2;

	// assert(std::abs(dist1 - dist2) < .0001 );
    // return dist2;

    // this impl is at least 20% slower than just computing the squared norm
    // on random data, which it shouldn't be...but seems about the same on
    // random walks, suggesting that very occasionally it abandons
    // EIGEN_ASM_COMMENT("abandon_dist_sq_start");
    __asm__ volatile ("# abandon_dist_sq_start");
    auto n = x.size();
    Scalar dist = 0;

    namespace ei = Eigen::internal;

    static constexpr int packet_size = ei::packet_traits<Scalar>::size;
	using packet_t = typename ei::packet_traits<Scalar>::type;
    alignas(32) Scalar dist_packet_ar[packet_size] = {0};
	packet_t dist_packet = ei::pload<packet_t>(dist_packet_ar);

	// ------------------------ version without early abandoning
	// yes, this is basically just as fast as the direct (x-y).squaredNorm()
	
    for (int i = 0; i <= (n - packet_size); i += packet_size) {
        const packet_t x_pkt = ei::pload<packet_t>(x.data() + i);
        const packet_t y_pkt = ei::pload<packet_t>(y.data() + i);
        const packet_t diff = ei::psub(x_pkt, y_pkt);
        dist_packet = ei::pmadd(diff, diff, dist_packet);
    }
    dist = ei::predux<packet_t>(dist_packet);

    auto num_stragglers = n % packet_size;
    Scalar single_dist = 0;
    for (size_t i = n - num_stragglers; i < n; i++) {
        auto diff = x(i) - y(i);
        single_dist += diff * diff;
    }
    return dist + single_dist;

    // ------------------------ version with early abandoning

    for (int i = 0; i <= (n - BlockSize); i += BlockSize) { // TODO uncomment below

#define USE_PACKETS
#ifdef USE_PACKETS
        for (int j = 0; j < BlockSize; j += packet_size) {
			const packet_t x_pkt = ei::pload<packet_t>(x.data() + i + j);
			const packet_t y_pkt = ei::pload<packet_t>(y.data() + i + j);
            const packet_t diff = ei::psub(x_pkt, y_pkt);
            dist_packet = ei::pmadd(diff, diff, dist_packet);
        }
        dist = ei::predux<packet_t>(dist_packet);
        // const packet_t dist_packet_copy = dist; // different register
        // dist = ei::predux<packet_t>(dist_packet_copy);
#else
        // // 20% slower for no clear reason
        // FixedMapT x_map(x.data() + i, BlockSize);
        // FixedMapT y_map(y.data() + i, BlockSize);
        // dist += (x_map - y_map).squaredNorm();
#endif
        // dist += (FixedVectorT::MapAligned(x.data() + i, BlockSize) -
        //     FixedVectorT::MapAligned(y.data() + i, BlockSize)).squaredNorm();
        if (UNLIKELY(dist >= thresh)) {
            // even on rand unif data, this abandons 98%+ of the time,
            // typically after looking at like 2/3 of each one in 64D space;
            // on rand walks, abandons after .5-2.1 blocks with blocksz = 16

            // if (num_abandons != nullptr) { ++(*num_abandons); }
            // if (abandon_iters != nullptr) { *abandon_iters += (i / BlockSize); }
            return dist;
        }
    }
    auto end_len = n % BlockSize;
    // auto end_blocks = end_len / BlockSize;
    auto end_stragglers = end_len % BlockSize;
    if (end_len > 0) {
        for (size_t j = n - end_len; j <= (end_len - packet_size) ; j += packet_size) {
            const packet_t x_pkt = ei::pload<packet_t>(x.data() + j);
            const packet_t y_pkt = ei::pload<packet_t>(y.data() + j);
            const packet_t diff = ei::psub(x_pkt, y_pkt);
            dist_packet = ei::pmadd(diff, diff, dist_packet);
        }
        dist = ei::predux<packet_t>(dist_packet);

        Scalar other_dist = 0;
        // for (int j; j < end_stragglers; j++) {
        for (size_t j = n - end_stragglers; j < n; j++) {
            auto diff = (x.data() + j) - (y.data() + j);
            other_dist += diff * diff;
        }

        dist += other_dist;
    // if (dist < thresh && end_len > 0) {
        // DynamicMapT x_map(x.data() + n - end_len, end_len);
        // DynamicMapT y_map(y.data() + n - end_len, end_len);
        // dist += (x_map - y_map).squaredNorm();
    }
    // EIGEN_ASM_COMMENT("abandon_dist_sq_end");
    __asm__ volatile ("# abandon_dist_sq_end");
    // assert(std::abs(dist1 - dist) < .0001 || dist >= thresh );
	return dist;
}

template<int BlockSize=8, class VectorT1, class VectorT2, class IdxVectorT,
    class dist_t>
inline dist_t dist_sq_order(const VectorT1& x, const VectorT2& y,
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
inline dist_t dist_sq_order_presorted(const VectorT1& x_sorted, const VectorT2& y,
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
inline void create_ordered_query(const VectorT& query, const VectorT& means,
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
