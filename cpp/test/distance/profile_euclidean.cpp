//
//  profile_euclidean.cpp
//  Dig
//
//  Created by DB on 11/14/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "catch.hpp"
#include "euclidean.hpp"

#include "Dense"

#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "testing_utils.hpp"
#include "timing_utils.hpp"

using RowMatrixXd = RowMatrix<double>;
using RowVectorXd = RowVector<double>;
using RowMatrixXf = RowMatrix<float>;
using RowVectorXf = RowVector<float>;

static const int kDefaultNumQueriesPerRun = 10;
static const int kDefaultNumRuns = 1;

template<class F>
inline void _randunif_knn_time(F&& dist_func, const char* func_name,
    int N, int D, int k, int num_runs=kDefaultNumRuns,
    int num_queries=kDefaultNumQueriesPerRun)
{
    RowMatrixXf X(N, D);
    RowVectorXf q(D);
    double total_time = 0;
    for (int r = 0; r < num_runs; r++) {
        X.setRandom();
        for (int j = 0; j < num_queries; j++) {
            q.setRandom();
            EasyTimer _(total_time, true); // true = add to value
            auto dists = dist_func(X, q, k);
            // for (int i = 0; i < N; i++) {
            //     auto dist = dist_func(X.row(i), q);
            // }
        }
    }
    int total_queries = num_runs * num_queries;
    int Mops = N * D * total_queries / (1000 * 1000); // mega-ops
    double gflops = (Mops / total_time); // million ops / ms = billion ops / s
    double ms_per_query = total_time / total_queries;
    std::cout << "func\t\tN\t\tD \tMops \tGFLOPS \ttotal \tms/q\n";
    printf("%s\t%d\t%d\t%d\t\t%.3f\t%.1f\t\t%.3f\n", func_name, N, D,
           Mops, gflops, total_time, ms_per_query);
}

TEST_CASE("l2_10nn", "[profile][distance][euclidean]") {
    // _randunif_knn_time([](const auto& x, const auto& q) {
    //     return (x * q.transpose()).eval() * -2;
    // }, "matmul", 100 * 10000, 8);
    int k = 10;
    int N = 100 * 1000;
    int D = 128;
    Eigen::VectorXf spoof_rownorms(N);
    _randunif_knn_time([&spoof_rownorms](const auto& X, const auto& q, const auto k) {
        return dist::squared_dists_to_vector(X, q, spoof_rownorms); // ignore k
    }, "matmul", N, D, k);

    _randunif_knn_time([](const auto& X, const auto& q, const auto k) {
        for (int i = 0; i < X.rows(); i++) {
            volatile auto dist = dist::simple::dist_sq(X.row(i), q);
        }
        return 0;
    }, "simple", N, D, k);

    _randunif_knn_time([](const auto& X, const auto& q, const auto k) {
        for (int i = 0; i < X.rows(); i++) {
            volatile auto dist = dist::simple::dist_sq_scalar(X.row(i), q);
        }
        return 0;
    }, "scalar", N, D, k); // about 6x slower than simple, 5x slower than matmul

}
