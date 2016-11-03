//  neighbor_testing_utils.hpp
//
//  Dig
//
//  Created by DB on 9/15/16
//  Copyright © 2016 D Blalock. All rights reserved.

#ifndef __NEIGHBOR_TESTING_UTILS_HPP
#define __NEIGHBOR_TESTING_UTILS_HPP

#include "euclidean.hpp"
#include "nn_utils.hpp"

#include "debug_utils.hpp"

using dist_t = Neighbor::dist_t;

// macro so that, if it fails, failing line is within the test
#define REQUIRE_NEIGHBORS_SAME(nn, trueNN) \
    REQUIRE(nn.idx == trueNN.idx); \
    REQUIRE(std::abs(nn.dist - trueNN.dist) < .0001);


template<class Container>
vector<typename Neighbor::idx_t> idxs_from_neighbors(const Container& neighbors)
{
    return ar::map([](const Neighbor& n) { return n.idx; }, neighbors);
}
template<class Container>
vector<typename Neighbor::dist_t> dists_from_neighbors(const Container& neighbors)
{
    return ar::map([](const Neighbor& n) { return n.dist; }, neighbors);
}

template<class Container1, class Container2>
void require_neighbor_lists_same(const Container1& nn, const Container2& trueNN)
{

    auto dists = dists_from_neighbors(nn);
    auto true_dists = dists_from_neighbors(trueNN);
    auto sorted_dists = ar::sort(dists);
    auto true_sorted_dists = ar::sort(true_dists);

    CAPTURE(ar::to_string(ar::sort(idxs_from_neighbors( nn ) ) ));
    CAPTURE(ar::to_string(ar::sort(idxs_from_neighbors(trueNN))));
    CAPTURE(ar :: to_string ( sorted_dists )); // spacing to align output
    CAPTURE(ar::to_string(true_sorted_dists));

    // print out last elements to check if failures stem from numerical errors;
    // we manually check whether there will be a problem because Catch refuses
    // to catpure these variables for no clear reason
    if (nn.size() != trueNN.size()) {
        if (auto sz = sorted_dists.size()) {
            PRINT_VAR(sorted_dists[sz-1]);
        }
        if (auto sz = true_sorted_dists.size()) {
            PRINT_VAR(true_sorted_dists[sz-1]);
        }
    }

    REQUIRE(nn.size() == trueNN.size());
    for (int i = 0; i < nn.size(); i++) {
        REQUIRE_NEIGHBORS_SAME(nn[i], trueNN[i]);
    }
}

template<class Container1, class Container2>
void require_neighbor_idx_lists_same(const Container1& nn_idxs,
    const Container2& trueNN_idxs)
{
    REQUIRE(nn_idxs.size() == trueNN_idxs.size());
    REQUIRE(ar::all_eq(nn_idxs, trueNN_idxs));
}


template<class MatrixT, class VectorT, class DistT = typename MatrixT::Scalar>
inline vector<Neighbor> radius_simple(const MatrixT& X, const VectorT& q,
									  DistT radius_sq)
{
    vector<Neighbor> trueKnn;
    for (int32_t i = 0; i < X.rows(); i++) {
        DistT d = (X.row(i) - q).squaredNorm();
        if (d < radius_sq) {
			trueKnn.emplace_back(i, d);
        }
    }
    return trueKnn;
}

template<class MatrixT, class VectorT>
inline Neighbor onenn_simple(const MatrixT& X, const VectorT& q) {
    Neighbor trueNN;
    double d_bsf = INFINITY;
    for (int32_t i = 0; i < X.rows(); i++) {
        double d = (X.row(i) - q).squaredNorm();
//      double dist2 = dist_sq(X.row(i).data(), q.data(), q.size());
        // double dist2 = dist::dist_sq(X.row(i), q);
//      REQUIRE(approxEq(dist1, dist2));
        if (d < d_bsf) {
            d_bsf = d;
			trueNN = Neighbor{i, d};
        }
    }
    return trueNN;
}

template<class MatrixT, class VectorT>
inline vector<Neighbor> knn_simple(const MatrixT& X, const VectorT& q, int k) {
    assert(k <= X.rows());

    vector<Neighbor> trueKnn;
    for (int32_t i = 0; i < k; i++) {
        auto dist = (X.row(i) - q).squaredNorm();
        trueKnn.emplace_back(i, dist);
    }
    nn::sort_neighbors_ascending_distance(trueKnn);

    typename MatrixT::Scalar d_bsf = INFINITY;
    for (int32_t i = k; i < X.rows(); i++) {
		auto dist = dist::simple::dist_sq(X.row(i), q);
        nn::maybe_insert_neighbor(trueKnn, dist, i);
    }
    return trueKnn;
}


template<class MatrixT, class IndexT, class QueryT>
inline void _test_index_with_query(MatrixT& X, IndexT& index,
    QueryT& q)
{
    for (int i = 0; i < 100; i++) {
    // for (int i = 0; i < 10; i++) {
    // for (int i = 0; i < 1; i++) {
        q.setRandom();

        // ------------------------ onenn
        auto nn = index.onenn(q);
        auto trueNN = onenn_simple(X, q);
        CAPTURE(nn.dist);
        CAPTURE(trueNN.dist);
        REQUIRE_NEIGHBORS_SAME(nn, trueNN);

        auto nn_idx = index.onenn_idxs(q);
        auto trueNN_idx = trueNN.idx;
        CAPTURE(nn_idx);
        CAPTURE(trueNN_idx);
        REQUIRE(nn_idx == trueNN_idx);

        // ------------------------ radius
        // use the dist to the first point as a radius that should include
        // about half of the points; we round to a few decimal places to avoid
        // numerical errors in distance computations causing test failures
        auto reasonable_dist = (X.row(0) - q).squaredNorm();
        reasonable_dist = round(reasonable_dist * 1024.f - 1) / 1024.f;
        auto allnn = index.radius(q, reasonable_dist);
        auto all_trueNN = radius_simple(X, q, reasonable_dist);

        CAPTURE(reasonable_dist);
        require_neighbor_lists_same(allnn, all_trueNN);

        auto allnn_idxs = index.radius_idxs(q, reasonable_dist);
        auto trueNN_idxs = idxs_from_neighbors(all_trueNN);
        CAPTURE(ar::to_string(ar::sort(allnn_idxs )));
        CAPTURE(ar::to_string(ar::sort(trueNN_idxs)));
        require_neighbor_idx_lists_same(allnn_idxs, trueNN_idxs);

        // ------------------------ knn
        for (int k = 1; k <= 5; k += 2) {
            auto knn = index.knn(q, k);
            auto trueKnn = knn_simple(X, q, k);
            require_neighbor_lists_same(knn, trueKnn);

            auto knn_idxs = index.knn_idxs(q, k);
            auto trueKnn_idxs = idxs_from_neighbors(trueKnn);
            CAPTURE(k);
            CAPTURE(knn[0].dist);
            CAPTURE(knn[0].idx);
            CAPTURE(trueKnn[0].dist);
            CAPTURE(trueKnn[0].idx);
            CAPTURE(ar::to_string(ar::sort(knn_idxs    )));
            CAPTURE(ar::to_string(ar::sort(trueKnn_idxs)));
            REQUIRE(ar::all_eq(knn_idxs, trueKnn_idxs));
        }
    }
}

template<class IndexT>
void _test_index(int64_t N=100, int64_t D=16) {
    using Scalar = typename IndexT::Scalar;
    RowMatrix<Scalar> X(N, D);
    X.setRandom();

    IndexT index(X);
    RowVector<Scalar> q(D);
    _test_index_with_query(X, index, q);
}

template<class IndexT>
void _test_cluster_index(int64_t N=100, int64_t D=16, int num_clusters=2) {
    using Scalar = typename IndexT::Scalar;
    RowMatrix<Scalar> X(N, D);
    X.setRandom();

    IndexT index(X, num_clusters);
    RowVector<Scalar> q(D);
    _test_index_with_query(X, index, q);
}


#endif // __NEIGHBOR_TESTING_UTILS_HPP
