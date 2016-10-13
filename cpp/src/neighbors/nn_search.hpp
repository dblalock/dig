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

namespace internal {
	template<class RowMatrixT, class _=decltype(std::declval<RowMatrixT>().rows())>
    idx_t _num_rows(const RowMatrixT& X, int64_t nrows_hint) {
        if (nrows_hint > 0) {
            return nrows_hint;
        }
        return X.rows();
    }
    template<class RowMatrixT> // doesn't have rows()
    idx_t _num_rows(const RowMatrixT& X, int64_t nrows_hint) {
        assert(nrows_hint > 0);
        return nrows_hint;
    }

    // either return vector of neighbors or just indices
    template<class Ret> struct emplace_neighbor {
        template<class Dist, class Idx>
        void operator()(std::vector<Ret>& vect, Dist d, Idx idx) {
            vect.emplace_back(static_cast<Ret>(idx));
        }
    };
    template<> struct emplace_neighbor<Neighbor> {
        template<class Dist, class Idx>
        void operator()(std::vector<Neighbor>& vect, Dist d, Idx idx) {
			vect.emplace_back(Neighbor{.dist = d, .idx = idx});
        }
    };
}


// ------------------------------------------------ brute force search

namespace brute {

    // ================================ single query, with + without row norms

    // ------------------------ radius
    template<class RowMatrixT, class VectorT>
    inline vector<Neighbor> radius(const RowMatrixT& X, const VectorT& query,
                                         float radius_sq)
    {
        auto dists = dist::squared_dists_to_vector(X, query);
        return neighbors_in_radius(dists.data(), dists.size(), radius_sq);
    }
    template<class RowMatrixT, class VectorT, class ColVectorT>
    inline vector<Neighbor> radius(const RowMatrixT& X, const VectorT& query,
        float radius_sq, const ColVectorT& rowSquaredNorms)
    {
        auto dists = dist::squared_dists_to_vector(X, query, rowSquaredNorms);
        return neighbors_in_radius(dists.data(), dists.size(), radius_sq);
    }

    // ------------------------ onenn
    template<class RowMatrixT, class VectorT>
    inline Neighbor onenn(const RowMatrixT& X, const VectorT& query) {
        typename RowMatrixT::Index idx;
        static_assert(VectorT::IsRowMajor, "query must be row-major");
		auto diffs = X.rowwise() - query;
		auto norms = diffs.rowwise().squaredNorm();
		auto dist = norms.minCoeff(&idx);
        return {.dist = dist, .idx = static_cast<idx_t>(idx)};
    }
    template<class RowMatrixT, class VectorT, class ColVectorT>
    inline Neighbor onenn(const RowMatrixT& X, const VectorT& query,
        const ColVectorT& rowSquaredNorms)
    {
        // static_assert(VectorT::IsRowMajor, "query must be row-major");
        typename RowMatrixT::Index idx;
        if (VectorT::IsRowMajor) {
            auto dists = rowSquaredNorms - (2 * (X * query.transpose()));
            auto min_dist = dists.minCoeff(&idx);
            min_dist += query.squaredNorm();
            return {.dist = min_dist, .idx = static_cast<idx_t>(idx)};
		} else {
            auto dists = rowSquaredNorms - (2 * (X * query));
            auto min_dist = dists.minCoeff(&idx) + query.squaredNorm();
            min_dist += query.squaredNorm();
            return {.dist = min_dist, .idx = static_cast<idx_t>(idx)};
        }

        // SELF: pick up here by reimplementing this using rowSquaredNorms

        // // typename RowMatrixT::Index idx;
        // auto diffs = X.rowwise() - query;
        // auto norms = diffs.rowwise().squaredNorm();
        // auto dist = norms.minCoeff(&idx);
        // return {.dist = dist, .idx = static_cast<idx_t>(idx)};
    }

    // ------------------------ knn
    template<class RowMatrixT, class VectorT>
    vector<Neighbor> knn(const RowMatrixT& X, const VectorT& query, size_t k)
    {
        if (VectorT::IsRowMajor) {
    		auto dists = dist::squared_dists_to_vector(X, query);
            return knn_from_dists(dists.data(), dists.size(), k);
        } else {
            auto dists = dist::squared_dists_to_vector(X, query.transpose());
            return knn_from_dists(dists.data(), dists.size(), k);
        }
    }
    template<class RowMatrixT, class VectorT, class ColVectorT>
    vector<Neighbor> knn(const RowMatrixT& X, const VectorT& query,
        size_t k, const ColVectorT& rowSquaredNorms)
    {
        auto dists = dist::squared_dists_to_vector(X, query, rowSquaredNorms);
        return knn_from_dists(dists.data(), dists.size(), k);
    }

    // ================================ batch of queries

    template<class RowMatrixT, class RowMatrixT2, class ColVectorT>
    inline vector<vector<Neighbor> > radius_batch(const RowMatrixT& X,
        const RowMatrixT2& queries, float radius_sq, const ColVectorT& rowNorms)
    {
        auto dists = dist::squared_dists_to_vectors(X, queries, rowNorms);
        assert(queries.rows() == dists.cols());
        auto num_queries = queries.rows();

        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < num_queries; j++) {
            ret.emplace_back(neighbors_in_radius(dists.data(),
                                                 dists.size(), radius_sq));
        }
        return ret;
    }

    template<class RowMatrixT, class RowMatrixT2, class ColVectorT>
    inline vector<vector<Neighbor> > knn_batch(const RowMatrixT& X,
        const RowMatrixT2& queries, size_t k, const ColVectorT& rowNorms)
    {
        auto dists = dist::squared_dists_to_vectors(X, queries, rowNorms);
        assert(queries.rows() == dists.cols());
        auto num_queries = queries.rows();
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < num_queries; j++) {
            ret.emplace_back(knn_from_dists(dists.col(j).data(),
                                            dists.rows(), k));
        }
        return ret;
    }

    template<class RowMatrixT, class RowMatrixT2, class ColVectorT>
    inline vector<Neighbor> onenn_batch(const RowMatrixT& X,
        const RowMatrixT& queries, const ColVectorT& rowNorms)
    {
        auto wrapped_neighbors = knn_batch(X, queries, 1, rowNorms);
        return ar::map([](const vector<Neighbor> el) {
            return el[0];
        }, wrapped_neighbors);
    }

} // namespace brute

// ------------------------------------------------ early abandoning search

namespace abandon {

// ================================ single query

// ------------------------ radius

template<class Ret=Neighbor, class RowMatrixT=char, class VectorT=char,
    class DistT=char>
vector<Ret> radius(const RowMatrixT& X, const VectorT& query,
    DistT radius_sq, idx_t nrows=-1)
    // const VectorT& query, idx_t num_rows, DistT radius_sq)
{
    vector<Ret> ret;
    for (idx_t i = 0; i < internal::_num_rows(X, nrows); i++) {
        auto dist = dist::abandon::dist_sq(X.row(i).eval(), query, radius_sq);
        if (dist <= radius_sq) {
            internal::emplace_neighbor<Ret>{}(ret, dist, i);
        }
    }
    return ret;
}
// template<class RowMatrixT, class VectorT, class DistT>
// vector<Neighbor> radius(const RowMatrixT& X, const VectorT& query,
//     DistT radius_sq)
//     // const VectorT& query, idx_t num_rows, DistT radius_sq)
// {
//     vector<Neighbor> ret;
//     for (idx_t i = 0; i < X.rows(); i++) {
//         auto dist = dist::abandon::dist_sq(X.row(i).eval(), query, radius_sq);
//         if (dist <= radius_sq) {
//             ret.emplace_back(dist, i);
//         }
//     }
//     return ret;
// }

// template<class RowMatrixT, class VectorT, class DistT>
// vector<vector<Neighbor> > radius_batch(const RowMatrixT& X,
//     const RowMatrixT& queries, dist_t radius_sq)
// {
//     vector<vector<Neighbor> > ret;
//     for (idx_t j = 0; j < queries.rows(); j++) {
//         ret.emplace_back(radius(X, queries.row(j).eval(), radius_sq));
//     }
//     return ret;
// }

// template<class RowMatrixT, class VectorT, class IdxVectorT, class DistT>
// vector<Neighbor> radius_order(const RowMatrixT& X,
//     const VectorT& query_sorted, const IdxVectorT& order,
//     DistT radius_sq)
//     // const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
// {
//     vector<Neighbor> ret;
//     for (idx_t i = 0; i < X.rows(); i++) {
//         auto dist = dist::abandon::dist_sq_order_presorted(query_sorted, X.row(i).eval(),
//             order, radius_sq);
//         if (dist <= radius_sq) {
//             ret.emplace_back(dist, i);
//         }
//     }
//     return ret;
// }


// template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
//     class DistT>
// vector<Neighbor> radius_adaptive(const RowMatrixT& X, const VectorT1& query,
//     const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
//     idx_t num_rows_thresh, DistT radius_sq)
// {
//     if (X.rows() >= num_rows_thresh) {
//         dist::abandon::create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
//         return radius_order(X, query_tmp, order_tmp, radius_sq);
//     } else {
//         return radius(X, query, radius_sq);
//     }
// }



// ------------------------ 1nn

template<class RowMatrixT, class VectorT, class DistT=neighbor_dist_t>
Neighbor onenn(const RowMatrixT& X, const VectorT& query,
    DistT d_bsf=kMaxDist, idx_t nrows=-1)
{
    Neighbor ret{.dist = d_bsf, .idx = kInvalidIdx };
	for (idx_t i = 0; i < internal::_num_rows(X, nrows); i++) {
        auto dist = dist::abandon::dist_sq(X.row(i).eval(), query, d_bsf);
		// auto d = dist::dist_sq(X.row(i).eval(), query);
        assert(dist >= 0);
        if (dist < d_bsf) {
            d_bsf = dist;
            ret = {.dist = dist, .idx = i};
        }
    }
    return ret;
}

// template<class RowMatrixT, class VectorT, class IdxVectorT,
// 	class DistT=neighbor_dist_t>
// Neighbor onenn_order(const RowMatrixT& X,
//     const VectorT& query_sorted, const IdxVectorT& order,
//     DistT d_bsf=kMaxDist)
//     // const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
// {
//     Neighbor ret{.dist = d_bsf, .idx = kInvalidIdx};
//     for (idx_t i = 0; i < X.rows(); i++) {
//         auto dist = dist::abandon::dist_sq_order_presorted(query_sorted,
//             X.row(i).eval(), order, d_bsf);
//         if (dist < ret.dist) {
//             d_bsf = dist;
//             ret = {.dist = dist, .idx = i};
//         }
//     }
//     return ret;
// }

// template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
//     class DistT=neighbor_dist_t>
// Neighbor onenn_adaptive(const RowMatrixT& X, const VectorT1& query,
//     const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
//     idx_t num_rows_thresh, DistT d_bsf=kMaxDist)
//     // idx_t num_rows, idx_t num_rows_thresh, DistT d_bsf=kMaxDist)
// {
//     if (X.rows() >= num_rows_thresh) {
//         create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
//         return onenn_order(X, query_tmp, order_tmp, d_bsf);
//     } else {
//         return onenn(X, query, d_bsf);
//     }
// }

// ------------------------ knn

template<class RowMatrixT, class VectorT, class DistT=neighbor_dist_t>
vector<Neighbor> knn(const RowMatrixT& X,
    const VectorT& query, int k, DistT d_bsf=kMaxDist, idx_t nrows=-1)
    // const VectorT& query, idx_t num_rows, int k, DistT d_bsf=kMaxDist)
{
    assert(k > 0);
    vector<Neighbor> ret(k, {.dist = d_bsf, .idx = kInvalidIdx});
	for (idx_t i = 0; i < internal::_num_rows(X, nrows); i++) {
		auto d = dist::abandon::dist_sq(X.row(i).eval(), query, d_bsf);
        d_bsf = maybe_insert_neighbor(ret, d, i); // figures out whether dist is lower
    }
    return ret;
}

// template<class RowMatrixT, class VectorT, class IdxVectorT,
// 	class DistT=neighbor_dist_t>
// vector<Neighbor> knn_order(const RowMatrixT& X,
//     const VectorT& query_sorted, const IdxVectorT& order,
//     int k, DistT d_bsf=kMaxDist, idx_t nrows=-1)
//     // const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
// {
//     assert(k > 0);
//     vector<Neighbor> ret(k, {.dist = d_bsf, .idx = kInvalidIdx});
//     for (idx_t i = 0; i < X.rows(); i++) {
//         auto dist = dist_sq_order_presorted(query_sorted, X.row(i).eval(),
//             order, d_bsf);
//         d_bsf = maybe_insert_neighbor(ret, dist, i); // figures out whether dist is lower
//     }
//     return ret;
// }

// template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
//     class DistT=neighbor_dist_t>
// vector<Neighbor> knn_adaptive(const RowMatrixT& X, const VectorT1& query,
//     const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
//     idx_t num_rows_thresh, int k, DistT d_bsf=kMaxDist, idx_t nrows=-1)
// {
//     assert(k > 0);
//     if (X.rows() >= num_rows_thresh) {
//         create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
//         return knn_order(X, query_tmp, order_tmp, d_bsf, nrows);
//     } else {
//         return knn(X, query, d_bsf, nrows);
//     }
// }

} // namespace abandon

} // namespace nn
#endif // __NN_SEARCH_HPP
