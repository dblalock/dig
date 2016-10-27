//
//  prof_neighbors.cpp
//  Dig
//
//  Created by DB on 10/25/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "neighbors.hpp"
#include "catch.hpp"

#include <string>

#include "neighbor_testing_utils.hpp"
#include "nn_search.hpp"
#include "timing_utils.hpp"

// constexpr int kDefaultN = 100 * 1000;
constexpr int kDefaultN = 500;
constexpr int kDefaultD = 100;

template<class MatrixT, class IndexT, class QueryT, class... Args>
inline void _prof_wrapper_index_with_query(MatrixT& X, IndexT& index,
    QueryT& q, std::string msg, double* total_time, Args&&... args)
{
    // std::cout << "running stuff... with msg: " << msg << "\n";
    PrintTimer t(msg);
    // for (int i = 0; i < 100; i++) {
    for (int i = 0; i < 10; i++) {
    // for (int i = 0; i < 1; i++) {
        q.setRandom();

        EasyTimer _(total_time, true); // true = add to value

        // ------------------------ radius
        // auto reasonable_dist = (X.row(0) - q).squaredNorm() + .00001;
        // auto allnn_idxs = index.radius(q, reasonable_dist,
        //     std::forward<Args>(args)...);

        // ------------------------ knn

        // 10nn
        volatile auto knn_idxs = index.knn(q, 10, std::forward<Args>(args)...);

        // for (int k = 1; k <= 5; k += 2) {
            // volatile auto knn_idxs = index.knn(q, k, std::forward<Args>(args)...);
        // }
    }
}

typedef struct search_config {
    // enum class Dataset { RANDUNIF, RANDWALK };
    std::string msg;
    int64_t N;
    int64_t D;
    int num_clusters = -1;
    float search_frac = -1;
    double total_time = 0;
} search_config;

search_config create_search_config(int64_t N, int64_t D, std::string msg="",
    int num_clusters=-1, float search_frac=-1)
{
	search_config cfg;
    cfg.msg = msg;
    cfg.N = N;
    cfg.D = D;
    cfg.num_clusters = num_clusters;
    cfg.search_frac = search_frac;
    if (msg.size() < 1) {
        msg = string_with_format("%lldx%lld_%d_%d,",
            N, D, num_clusters, search_frac);
    }
	return cfg;
}

search_config default_search_cfg() {
    int N = 10 * 1000;
    int D = 100;
    // auto msg = string_with_format("\t%dx%d", N, D);
    // return create_search_config(N, D, msg);
    return create_search_config(N, D);
}

template<class Scalar>
RowMatrix<Scalar> rand_mat(int64_t N, int64_t D) {
    RowMatrix<Scalar> X(N, D);
    X.setRandom();
    return X;
}

template<class Scalar>
RowMatrix<Scalar> rand_mat(const search_config& cfg) {
    return rand_mat<Scalar>(cfg.N, cfg.D);
}

template<class IndexT>
// void _prof_wrapper_index(int64_t N=200, int64_t D=100) {
void _prof_wrapper_index(search_config& cfg) {
    // auto msg = string_with_format("{}x{}", N, D);
    // search_config cfg = create_search_config("\t" #N "x" #D, N, D);

    using Scalar = typename IndexT::Scalar;
    auto X = rand_mat<Scalar>(cfg);
    RowVector<Scalar> q(cfg.D);

    IndexT index(X);
    _prof_wrapper_index_with_query(X, index, q, cfg.msg, &cfg.total_time);
}

template<class IndexT>
void _prof_cluster_wrapper_index(search_config& cfg)
{
    using Scalar = typename IndexT::Scalar;
    auto X = rand_mat<Scalar>(cfg);
    RowVector<Scalar> q(cfg.D);

    IndexT index(X, cfg.num_clusters);
    _prof_wrapper_index_with_query(X, index, q, cfg.msg, &cfg.total_time,
        cfg.search_frac);
}


// #define PROF_WRAPPER_INDEX_ONCE(CLS, cfg) \
    // int N = 10*1000;
    // int D = 100;
    // auto msg = string_with_format("{}x{}", N, D);
    // _prof_wrapper_index<CLS>(cfg);


#define PROF_WRAPPER_INDEX(CLS) \
    do { std::cout << #CLS << ":\n"; \
    search_config cfg = default_search_cfg(); \
	for (int i = 0; i < 10; i++) { \
		_prof_wrapper_index<CLS>(cfg); \
	} \
    std::cout << "total time: " << cfg.total_time << "\n"; } while(0);

// #define PROF_CLUSTER_WRAPPER_INDEX_ONCE(CLS, N, D, NUM_CLUSTERS, SEARCH_FRAC) \
//     do { search_config cfg = create_search_config("\t" #N "x" #D, N, D, \
//         NUM_CLUSTERS, SEARCH_FRAC); \
//     _prof_cluster_wrapper_index<CLS>(cfg); } while (0);

#define PROF_CLUSTER_WRAPPER_INDEX(CLS, NUM_CLUSTERS, SEARCH_FRAC) \
    do { \
    std::cout << #CLS << "_" << #NUM_CLUSTERS << "_" #SEARCH_FRAC << ":\n"; \
	search_config cfg = default_search_cfg(); \
	cfg.num_clusters = NUM_CLUSTERS; \
	cfg.search_frac = SEARCH_FRAC; \
    for (int i = 0; i < 10; i++) { \
        _prof_cluster_wrapper_index<CLS>(cfg); \
    } \
    std::cout << "total time: " << cfg.total_time << "\n"; } while(0);

        // PROF_CLUSTER_WRAPPER_INDEX_ONCE(CLS, 10*1000, 100, NUM_CLUSTERS, SEARCH_FRAC); \
    // PROF_CLUSTER_WRAPPER_INDEX_ONCE(CLS, 100, 10, __VA_ARGS__); \
    // PROF_CLUSTER_WRAPPER_INDEX_ONCE(CLS, 250, 40, __VA_ARGS__); \
    // PROF_CLUSTER_WRAPPER_INDEX_ONCE(CLS, 1000, 64, __VA_ARGS__);


TEST_CASE("Prof_KmeansIndex", "[profile][neighbors]") {
    PROF_CLUSTER_WRAPPER_INDEX(KmeansIndex, 64, -1);
    PROF_CLUSTER_WRAPPER_INDEX(KmeansIndex, 64, .5);
    PROF_CLUSTER_WRAPPER_INDEX(KmeansIndex, 64, .25);
}
// TEST_CASE("KmeansIndexF", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(KmeansIndexF);
// }

// TEST_CASE("Prof_MatmulIndex", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(MatmulIndex);
// }
// TEST_CASE("MatmulIndexF", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(MatmulIndexF);
// }

// TEST_CASE("SimpleIndex", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(SimpleIndex);
// }
// TEST_CASE("SimpleIndexF", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(SimpleIndexF);
// }

// TEST_CASE("AbandonIndex", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(AbandonIndex);
// }
// TEST_CASE("AbandonIndexF", "[profile][neighbors]") {
//     PROF_WRAPPER_INDEX(AbandonIndexF);
// }
