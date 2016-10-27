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
#include "nn_search.hpp"
#include "timing_utils.hpp"


// SELF: this is breaking because we're passing in search_frac after k, which
// is getting interpreted as the number of rows.


template<class MatrixT, class IndexT, class QueryT, class... Args>
inline void _test_wrapper_index_with_query(MatrixT& X, IndexT& index,
    QueryT& q, const char* msg, Args&&... args)
{
    PrintTimer t(msg);
    // for (int i = 0; i < 100; i++) {
    for (int i = 0; i < 10; i++) {
    // for (int i = 0; i < 1; i++) {
        q.setRandom();

        // ------------------------ radius
        // auto reasonable_dist = (X.row(0) - q).squaredNorm() + .00001;
        // auto allnn_idxs = index.radius(q, reasonable_dist,
        //     std::forward<Args>(args)...);
        // auto trueNN = radius_simple(X, q, reasonable_dist);
        // auto trueNN_idxs = idxs_from_neighbors(trueNN);

        // CAPTURE(ar::to_string(allnn_idxs));
        // CAPTURE(ar::to_string(trueNN_idxs));
        // require_neighbor_idx_lists_same(allnn_idxs, trueNN_idxs);

        // ------------------------ knn
        for (int k = 1; k <= 5; k += 2) {
            auto knn_idxs = index.knn(q, k, std::forward<Args>(args)...);
			auto trueKnn = nn::simple::knn(X, q, k);
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

// nope, separating the defaults into a declaration doesn't fix the err...
// template<class IndexT, class... Args>
// void _test_wrapper_index(int64_t N=100, int64_t D=16, const char* msg="",
// 						 Args&&... args);
// template<class IndexT, class... Args>
// void _test_wrapper_index(int64_t N, int64_t D, const char* msg,
//                          Args&&... args)



// template<class IndexT, class F>
// void _test_wrapper_index(int64_t N=100, int64_t D=16, const char* msg="",
// 	F&& index_func)

// SELF: add option to pass in a lambda that builds the index given X, since clang
// is apparently dumb and wont let us use a param pack

template<class IndexT>
void _test_wrapper_index(int64_t N=100, int64_t D=16, const char* msg="")
{
    using Scalar = typename IndexT::Scalar;
    RowMatrix<Scalar> X(N, D);
    X.setRandom();

	IndexT index(X);
    RowVector<Scalar> q(D);
    _test_wrapper_index_with_query(X, index, q, msg);
}

template<class IndexT>
void _test_cluster_wrapper_index(int64_t N=100, int64_t D=16, const char* msg="",
    int num_clusters=-1, float search_frac=-1)
{
    using Scalar = typename IndexT::Scalar;
    RowMatrix<Scalar> X(N, D);
    X.setRandom();

    IndexT index(X, num_clusters);
    RowVector<Scalar> q(D);
    _test_wrapper_index_with_query(X, index, q, msg, search_frac);
}


#define TEST_WRAPPER_INDEX_ONCE(CLS, N, D) \
    _test_wrapper_index<CLS>(N, D, "\t" #N "x" #D);

#define TEST_WRAPPER_INDEX(CLS) \
    std::cout << #CLS << ":\n"; \
    TEST_WRAPPER_INDEX_ONCE(CLS, 100, 10); \
    TEST_WRAPPER_INDEX_ONCE(CLS, 250, 40); \
    TEST_WRAPPER_INDEX_ONCE(CLS, 1000, 64);

#define TEST_CLUSTER_WRAPPER_INDEX_ONCE(CLS, N, D, ...) \
	_test_cluster_wrapper_index<CLS>(N, D, "\t" #N "x" #D, __VA_ARGS__);

#define TEST_CLUSTER_WRAPPER_INDEX(CLS, ...) \
	std::cout << #CLS << ":\n"; \
    for (int i = 0; i < 10; i++) { \
        TEST_CLUSTER_WRAPPER_INDEX_ONCE(CLS, 10*1000, 100, __VA_ARGS__); \
    }

TEST_CASE("KmeansIndex", "[neighbors_wrappers]") {
    TEST_CLUSTER_WRAPPER_INDEX(KmeansIndex, 53); // weird number of clusters
	// _test_cluster_wrapper_index<KmeansIndex>(100, 16, "foo", 50);
}
TEST_CASE("KmeansIndexF", "[neighbors_wrappers]") {
    TEST_CLUSTER_WRAPPER_INDEX(KmeansIndexF, 47); // weird number of clusters
}

TEST_CASE("MatmulIndex", "[neighbors_wrappers]") {
    TEST_WRAPPER_INDEX(MatmulIndex);
}
TEST_CASE("MatmulIndexF", "[neighbors_wrappers]") {
    TEST_WRAPPER_INDEX(MatmulIndexF);
}

TEST_CASE("SimpleIndex", "[neighbors_wrappers]") {
    TEST_WRAPPER_INDEX(SimpleIndex);
}
TEST_CASE("SimpleIndexF", "[neighbors_wrappers]") {
    TEST_WRAPPER_INDEX(SimpleIndexF);
}

// TEST_CASE("AbandonIndex", "[neighbors_wrappers]") {
//     TEST_WRAPPER_INDEX(AbandonIndex);
// }
// TEST_CASE("AbandonIndexF", "[neighbors_wrappers]") {
//     TEST_WRAPPER_INDEX(AbandonIndexF);
// }
