//
//  test_neighbors.cpp
//  Dig
//
//  Created by DB on 10/13/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "neighbors.hpp"
#include "catch.hpp"

#include "neighbor_testing_utils.hpp"


// template<class IndexT>
// struct index_traits {
//  using Scalar = typename IndexT::Scalar;
// };
// template<>
// struct index_traits<MatmulIndex> {
//  using Scalar = double;
// };

template<class MatrixT, class IndexT, class QueryT>
inline void _test_wrapper_index_with_query(MatrixT& X, IndexT& index,
    QueryT& q)
{
    for (int i = 0; i < 100; i++) {
    // for (int i = 0; i < 10; i++) {
    // for (int i = 0; i < 1; i++) {
        q.setRandom();

        // ------------------------ radius
        auto reasonable_dist = (X.row(0) - q).squaredNorm() + .00001;
        auto allnn_idxs = index.radius(q, reasonable_dist);
        auto trueNN = radius_simple(X, q, reasonable_dist);
        auto trueNN_idxs = idxs_from_neighbors(trueNN);

        CAPTURE(ar::to_string(allnn_idxs));
        CAPTURE(ar::to_string(trueNN_idxs));
        require_neighbor_idx_lists_same(allnn_idxs, trueNN_idxs);

        // ------------------------ knn
        for (int k = 1; k <= 5; k += 2) {
            auto knn_idxs = index.knn(q, k);
            auto trueKnn = knn_simple(X, q, k);
			auto trueKnn_idxs = idxs_from_neighbors(trueKnn);

            CAPTURE(k);
            CAPTURE(knn_idxs[0]);
            CAPTURE(trueKnn_idxs[0]);
            CAPTURE(ar::to_string(knn_idxs));
            CAPTURE(ar::to_string(trueKnn_idxs));
            require_neighbor_idx_lists_same(knn_idxs, trueKnn_idxs);
        }
    }
}


template<class IndexT>
void _test_wrapper_index(int64_t N=100, int64_t D=16) {
    using Scalar = typename IndexT::Scalar;
    RowMatrix<Scalar> X(N, D);
    X.setRandom();

    IndexT index(X);
    RowVector<Scalar> q(D);
    _test_wrapper_index_with_query(X, index, q);
}

TEST_CASE("MatmulIndex", "neighbors") {
    _test_wrapper_index<MatmulIndex>();
    _test_wrapper_index<MatmulIndex>(100, 10);
    _test_wrapper_index<MatmulIndex>(64, 19);
}
TEST_CASE("MatmulIndexF", "neighbors") {
    _test_wrapper_index<MatmulIndexF>();
    _test_wrapper_index<MatmulIndexF>(100, 10);
    _test_wrapper_index<MatmulIndexF>(64, 19);
}

TEST_CASE("AbandonIndex", "neighbors") {
    _test_wrapper_index<AbandonIndex>();
    _test_wrapper_index<AbandonIndex>(100, 10);
    _test_wrapper_index<AbandonIndex>(64, 19);
}
TEST_CASE("AbandonIndexF", "neighbors") {
    _test_wrapper_index<AbandonIndexF>();
    _test_wrapper_index<AbandonIndexF>(100, 10);
    _test_wrapper_index<AbandonIndexF>(64, 19);
}
