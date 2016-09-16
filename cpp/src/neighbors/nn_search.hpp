//
//  nn_search.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __NN_SEARCH_HPP
#define __NN_SEARCH_HPP

#include "nn_utils.hpp"
#include "euclidean.hpp"

namespace nn {

// typedef float typename Neighbor::dist_t;

// ------------------------------------------------ early abandoning search

// ================================ early-abandoning search

// ------------------------ radius

template<class RowMatrixT, class VectorT, class DistT>
vector<Neighbor> radius(const RowMatrixT& X,
    const VectorT& query, DistT radius_sq)
    // const VectorT& query, idx_t num_rows, DistT radius_sq)
{
    vector<Neighbor> ret;
    for (idx_t i = 0; i < X.rows(); i++) {
        auto dist = dist_sq(X.row(i).eval(), query, radius_sq);
        if (dist <= radius_sq) {
            ret.emplace_back(dist, i);
        }
    }
    return ret;
}

template<class RowMatrixT, class VectorT, class IdxVectorT, class DistT>
vector<Neighbor> radius_order(const RowMatrixT& X,
    const VectorT& query_sorted, const IdxVectorT& order,
    DistT radius_sq)
    // const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
{
    vector<Neighbor> ret;
    for (idx_t i = 0; i < X.rows(); i++) {
        auto dist = dist_sq_order_presorted(query_sorted, X.row(i).eval(),
            order, radius_sq);
        if (dist <= radius_sq) {
            ret.emplace_back(dist, i);
        }
    }
    return ret;
}

template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
    class DistT>
vector<Neighbor> radius_adaptive(const RowMatrixT& X, const VectorT1& query,
    const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
    idx_t num_rows_thresh, DistT radius_sq)
    // idx_t num_rows, idx_t num_rows_thresh, DistT radius_sq)
{
    if (X.rows() >= num_rows_thresh) {
        create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
        return radius_order(X, query_tmp, order_tmp, radius_sq);
    } else {
        return radius(X, query, radius_sq);
    }
}

// ------------------------ 1nn

template<class RowMatrixT, class VectorT, class DistT=typename Neighbor::dist_t>
Neighbor onenn(const RowMatrixT& X, const VectorT& query,
    DistT d_bsf=kMaxDist)
{
    Neighbor ret{.dist = d_bsf, .idx = kInvalidIdx };
    for (idx_t i = 0; i < X.rows(); i++) {
        // auto dist = dist::abandon::dist_sq(X.row(i).eval(), query, d_bsf);
		auto d = dist::dist_sq(X.row(i).eval(), query);
        assert(d >= 0);
        if (d < d_bsf) {
            d_bsf = d;
            ret = {.dist = d, .idx = i};
        }
    }
    return ret;
}

template<class RowMatrixT, class VectorT, class IdxVectorT,
	class DistT=typename Neighbor::dist_t>
Neighbor onenn_order(const RowMatrixT& X,
    const VectorT& query_sorted, const IdxVectorT& order,
    DistT d_bsf=kMaxDist)
    // const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
{
    Neighbor ret{.dist = d_bsf, .idx = kInvalidIdx};
    for (idx_t i = 0; i < X.rows(); i++) {
        auto dist = dist::abandon::dist_sq_order_presorted(query_sorted,
            X.row(i).eval(), order, d_bsf);
        if (dist < ret.dist) {
            d_bsf = dist;
            ret = {.dist = dist, .idx = i};
        }
    }
    return ret;
}

template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
    class DistT=typename Neighbor::dist_t>
Neighbor onenn_adaptive(const RowMatrixT& X, const VectorT1& query,
    const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
    idx_t num_rows_thresh, DistT d_bsf=kMaxDist)
    // idx_t num_rows, idx_t num_rows_thresh, DistT d_bsf=kMaxDist)
{
    if (X.rows() >= num_rows_thresh) {
        create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
        return onenn_order(X, query_tmp, order_tmp, d_bsf);
    } else {
        return onenn(X, query, d_bsf);
    }
}

// ------------------------ knn

template<class RowMatrixT, class VectorT, class DistT=typename Neighbor::dist_t>
vector<Neighbor> knn(const RowMatrixT& X,
    const VectorT& query, int k, DistT d_bsf=kMaxDist)
    // const VectorT& query, idx_t num_rows, int k, DistT d_bsf=kMaxDist)
{
    assert(k > 0);
    vector<Neighbor> ret(k, {.dist = d_bsf, .idx = kInvalidIdx});
    for (idx_t i = 0; i < X.rows(); i++) {
		auto d = dist::abandon::dist_sq(X.row(i).eval(), query, d_bsf);
        d_bsf = maybe_insert_neighbor(ret, d, i); // figures out whether dist is lower
    }
    return ret;
}

template<class RowMatrixT, class VectorT, class IdxVectorT,
	class DistT=typename Neighbor::dist_t>
vector<Neighbor> knn_order(const RowMatrixT& X,
    const VectorT& query_sorted, const IdxVectorT& order,
    int k, DistT d_bsf=kMaxDist)
    // const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
{
    assert(k > 0);
    vector<Neighbor> ret(k, {.dist = d_bsf, .idx = kInvalidIdx});
    for (idx_t i = 0; i < X.rows(); i++) {
        auto dist = dist_sq_order_presorted(query_sorted, X.row(i).eval(),
            order, d_bsf);
        d_bsf = maybe_insert_neighbor(ret, dist, i); // figures out whether dist is lower
    }
    return ret;
}

template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
    class DistT=typename Neighbor::dist_t>
vector<Neighbor> knn_adaptive(const RowMatrixT& X, const VectorT1& query,
    const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
    idx_t num_rows_thresh, int k, DistT d_bsf=kMaxDist)
    // idx_t num_rows, idx_t num_rows_thresh, int k, DistT d_bsf=kMaxDist)
{
    assert(k > 0);
    if (X.rows() >= num_rows_thresh) {
        create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
        return knn_order(X, query_tmp, order_tmp, d_bsf);
    } else {
        return knn(X, query, d_bsf);
    }
}

} // namespace nn
#endif // __NN_SEARCH_HPP
