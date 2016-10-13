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

//void require_neighbors_same(const Neighbor& nn, const Neighbor& trueNN) {
//    REQUIRE(nn.idx == trueNN.idx);
//    REQUIRE(std::abs(nn.dist - trueNN.dist) < .0001);
//}

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
    CAPTURE(ar::to_string(idxs_from_neighbors(nn)));
    CAPTURE(ar::to_string(idxs_from_neighbors(trueNN)));
    CAPTURE(ar::to_string(dists_from_neighbors(nn)));
    CAPTURE(ar::to_string(dists_from_neighbors(trueNN)));
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
            trueKnn.emplace_back(Neighbor{.dist = d, .idx = i});
        }
    }
    return trueKnn;
}

template<class MatrixT, class VectorT>
inline Neighbor onenn_simple(const MatrixT& X, const VectorT& q) {
    Neighbor trueNN;
    double d_bsf = INFINITY;
    for (int32_t i = 0; i < X.rows(); i++) {
        double dist1 = (X.row(i) - q).squaredNorm();
//      double dist2 = dist_sq(X.row(i).data(), q.data(), q.size());
        // double dist2 = dist::dist_sq(X.row(i), q);
//      REQUIRE(approxEq(dist1, dist2));
        if (dist1 < d_bsf) {
            d_bsf = dist1;
            trueNN = Neighbor{.idx = i, .dist = dist1};
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
        trueKnn.push_back(Neighbor{.idx = i, .dist = dist});
    }
    nn::sort_neighbors_ascending_distance(trueKnn);

    typename MatrixT::Scalar d_bsf = INFINITY;
    for (int32_t i = k; i < X.rows(); i++) {
        auto dist = dist::dist_sq(X.row(i), q);
        nn::maybe_insert_neighbor(trueKnn, dist, i);
    }
    return trueKnn;
}


// struct undef;

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
        auto reasonable_dist = (X.row(0) - q).squaredNorm() + .00001;
        auto allnn = index.radius(q, reasonable_dist);
        auto all_trueNN = radius_simple(X, q, reasonable_dist);
        require_neighbor_lists_same(allnn, all_trueNN);

        auto allnn_idxs = index.radius_idxs(q, reasonable_dist);
        auto trueNN_idxs = idxs_from_neighbors(all_trueNN);
        CAPTURE(ar::to_string(allnn_idxs));
        CAPTURE(ar::to_string(trueNN_idxs));
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
            CAPTURE(ar::to_string(knn_idxs));
            CAPTURE(ar::to_string(trueKnn_idxs));
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
