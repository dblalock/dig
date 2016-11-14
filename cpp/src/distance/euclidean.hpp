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
// using void = char;

// ------------------------------------------------ brute force distances

// ------------------------ dists to batch of vectors, with row norms

/** compute distances between rows of X and rows of V */
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=void,
    class MatrixT2=void, class VectorT=void>
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

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=void,
    class MatrixT2=void, class VectorT=void>
inline auto dists_to_vectors(const MatrixT1& X, const MatrixT2& V,
    const VectorT& rowSquaredNorms)
    -> decltype(squared_dists_to_vectors(X, V, rowSquaredNorms))
{
    return squared_dists_to_vectors(X, V, rowSquaredNorms).array().sqrt().matrix();
}

// ------------------------ dists to batch of vectors, no row norms

/** compute distances between rows of X and rows of V */
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT1=void,
    class MatrixT2=void>
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

namespace internal {

template<class MatrixT=void, class VectorT=void, class VectorT2=void>
inline auto _squared_dists_to_vector(const MatrixT& X, const VectorT& v,
    const VectorT2& rowSquaredNorms)
    -> Eigen::Matrix<typename mat_product_traits<MatrixT, VectorT>::Scalar,
        Eigen::Dynamic, 1>
{
    using dist_t = typename mat_product_traits<MatrixT, VectorT>::Scalar;
    using RetT = Eigen::Matrix<dist_t, Eigen::Dynamic, 1>;
    auto n = X.rows();
    auto v_norm = v.squaredNorm();
    RetT ret(n);
    for (int i = 0; i < n; i++) {
        ret(i) = X.row(i).dot(v) * -2 + v_norm + rowSquaredNorms(i);
    }
    return ret;
}

} // namespace internal

template<class MatrixT=void, class VectorT=void, class VectorT2=void>
inline auto squared_dists_to_vector(const MatrixT& X, const VectorT& v,
    const VectorT2& rowSquaredNorms)
	-> typename mat_product_traits<MatrixT, VectorT>::type
	// -> decltype(squared_dists_to_vectors(X, VectorT::IsRowMajor ? v : v.transpose(), rowSquaredNorms))
    //-> typename mat_traits<VectorT>::VectorT
{
    static_assert(VectorT::RowsAtCompileTime == 1 || VectorT::ColsAtCompileTime == 1, "");
    return internal::_squared_dists_to_vector(X, v, rowSquaredNorms);

    // if (VectorT::ColsAtCompileTime == 1) {
    //     // return squared_dists_to_vectors(X, v, rowSquaredNorms);
    //     return internal::_squared_dists_to_vector(X, v, rowSquaredNorms);
    // } else {
    //     return internal::_squared_dists_to_vector(X, v.transpose(), rowSquaredNorms);
    //     // return squared_dists_to_vectors(X, v.transpose(), rowSquaredNorms);
    // }
}
template<class MatrixT=void, class VectorT=void, class VectorT2=void>
inline auto dists_to_vector(const MatrixT& X, const VectorT& v,
    const VectorT2& rowSquaredNorms)
    -> decltype(squared_dists_to_vector(X, v, rowSquaredNorms))
{
    return squared_dists_to_vector(X, v).array().sqrt().matrix();
}

// ------------------------ dist to single vector, no row norms

template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=void,
    class VectorT=void>
inline auto squared_dists_to_vector(const MatrixT& X, const VectorT& v)
    -> typename mat_traits<VectorT>::VectorT
{
    return (X.rowwise() - v).rowwise().squaredNorm();
}
template<int RetStorageOrder=Eigen::ColMajor, class MatrixT=void,
    class VectorT=void>
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

template<class VectorT1, class VectorT2>
inline auto dist_sq_scalar(const VectorT1& x, const VectorT2& y)
    -> typename mat_product_traits<VectorT1, VectorT2>::Scalar
{
    using dist_t = typename mat_product_traits<VectorT1, VectorT2>::Scalar;
    dist_t d = 0;
    for (int i = 0; i < x.size(); i++) {
        auto diff = x(i) - y(i);
        d += diff * diff;
    }
    return d;
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


template<int BlockSize=64, class VectorT1, class VectorT2,
	class dist_t=typename VectorT1::Scalar>
inline dist_t dist_sq(const VectorT1& x, const VectorT2& y,
    dist_t thresh=kMaxDist, int64_t* num_abandons=nullptr,
    int64_t* abandon_iters=nullptr)
{
    using Scalar = typename VectorT1::Scalar;
    // using FixedVectorT = Eigen::Matrix<Scalar, 1, BlockSize, Eigen::RowMajor>;
    // using FixedVectorT = Eigen::Matrix<Scalar, BlockSize, 1>;
	// using FixedVectorT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	// using FixedMapT = Eigen::Map<const FixedVectorT, Eigen::Aligned>;

    // using DynamicVectorT = typename scalar_traits<Scalar>::RowVectorT;
    // using DynamicVectorT = typename scalar_traits<Scalar>::ColVectorT;
    // using DynamicMapT = Eigen::Map<const DynamicVectorT, Eigen::Aligned>;
    // using DynamicMapT = Eigen::Map<const DynamicVectorT>;

	static_assert(std::is_same<Scalar, typename VectorT2::Scalar>::value,
		"Received vectors of different types!");
	static_assert((BlockSize * sizeof(Scalar)) % 32 == 0, // TODO use kDefaultAlignBytes
		"BlockSize * sizeof(Scalar) must be a multiple of 32B");
    assert(x.size() == y.size());
    assert((x.size() * sizeof(Scalar)) % 32 == 0); // assume ends padded

    // return (x - y).squaredNorm();

    __asm__("## abandon_dist_sq_start");
    auto n = x.size();

    // TODO could we tile thresh into a packet and then abandon if anything
    // is above the thresh?
        // not really, because d_bsf is sum of stuff in the dist packet


	// ------------------------ scalar version with early abandoning
    // about 90% slower than brute force simd on 40d randunif, but 1.5-3x
    // as fast on 500k 100d-randwalks

	// Scalar d = 0;
	// for (int i = 0; i < n; i++) {
	// 	Scalar diff = x(i) - y(i);
	// 	d += diff * diff; // with no abondoning, takes like 2x as long
 //        // if (d > thresh) { // with abandoning, also like 2x as long
	// 	if (UNLIKELY(d >= thresh)) { // with abandoning, also like 2x as long
	// 		return d;
	// 	}
	// }
	// return d;

    // ------------------------ stuff for both simd versions

    namespace ei = Eigen::internal;

    static constexpr int packet_size = ei::packet_traits<Scalar>::size;
    static constexpr int packets_per_block = BlockSize / packet_size;
    static_assert(packets_per_block >= 1,
        "block size must be at least packet size!");

    using packet_t = typename ei::packet_traits<Scalar>::type;
    // alignas(32) Scalar dist_packet_ar[packet_size] = {0};
    // packet_t dist_packet = ei::pload<packet_t>(dist_packet_ar);
    packet_t dist_packet = ei::pset1<packet_t>(0);
    // static const packet_t zero_packet = ei::pset1<packet_t>(0);
    // packet_t dist_packet_copy;// = ei::pset1<packet_t>(0);

	// ------------------------ simd version without early abandoning
	// yes, this is basically just as fast as the direct (x-y).squaredNorm()

    // for (int i = 0; i <= (n - packet_size); i += packet_size) {
    //     const packet_t x_pkt = ei::pload<packet_t>(x.data() + i);
    //     const packet_t y_pkt = ei::pload<packet_t>(y.data() + i);
    //     const packet_t diff = ei::psub(x_pkt, y_pkt);
    //     dist_packet = ei::pmadd(diff, diff, dist_packet);
    // }
    // Scalar dist = ei::predux<packet_t>(dist_packet);
    // auto num_stragglers = n % packet_size;
    // Scalar single_dist = 0;
    // for (size_t i = n - num_stragglers; i < n; i++) {
    //     auto diff = x(i) - y(i);
    //     single_dist += diff * diff;
    // }
    // return dist + single_dist;

    // ------------------------ simd version with early abandoning

    // alignas(32) static Scalar dist_packet_ar[packet_size] = {0};

    size_t num_blocks = n / BlockSize;
    size_t last_block_end = num_blocks * BlockSize;
    size_t num_trailing_packets = (n - last_block_end) / packet_size;
    size_t last_packet_end = last_block_end + num_trailing_packets * packet_size;
    for (int i = 0; i < last_block_end; ) {
        for (int count = 0; count < packets_per_block; count++) {
            const packet_t x_pkt = ei::pload<packet_t>(x.data() + i);
            const packet_t y_pkt = ei::pload<packet_t>(y.data() + i);
            const packet_t diff = ei::psub(x_pkt, y_pkt);
            dist_packet = ei::pmadd(diff, diff, dist_packet);
            i += packet_size;
        }
        // dist_packet_copy = ei::padd<packet_t>(zero_packet, dist_packet);
        Scalar dist = ei::predux<packet_t>(dist_packet);
        // Scalar dist = ei::predux<packet_t>(dist_packet_copy);
		// if (dist >= thresh) { // no effect on randwalks; worse on randunif
        if (UNLIKELY(dist >= thresh)) {
    		return dist;
		}

        // // manual accumulating just makes it like 20% slower for everything
        // Scalar dist = 0;
        // ei::pstore(dist_packet_ar, dist_packet);
        // for (int i = 0; i < packet_size; i++) {
        //     dist += dist_packet_ar[i];
        //     if (UNLIKELY(dist >= thresh)) {
        //         return dist;
        //     }
        // }
    }
    for (size_t i = last_block_end; i < last_packet_end; i += packet_size) {
		const packet_t x_pkt = ei::pload<packet_t>(x.data() + i);
		const packet_t y_pkt = ei::pload<packet_t>(y.data() + i);
		const packet_t diff = ei::psub(x_pkt, y_pkt);
		dist_packet = ei::pmadd(diff, diff, dist_packet);
    }
	return ei::predux<packet_t>(dist_packet);

    // uncomment this block if we remove the assert enforcing padding
	// Scalar dist = ei::predux<packet_t>(dist_packet);
 //    auto num_stragglers = n % packet_size;
 //    Scalar single_dist = 0;
 //    for (size_t i = n - num_stragglers; i < n; i++) {
 //        auto diff = x(i) - y(i);
 //        single_dist += diff * diff;
 //    }
 //    return dist + single_dist;
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
