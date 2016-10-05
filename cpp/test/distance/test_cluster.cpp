//
//  test_cluster.cpp
//  Dig
//
//  Created by DB on 10/2/16.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include "cluster.hpp"

#include "catch.hpp"
#include "testing_utils.hpp"

#include "euclidean.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;
using namespace cluster;

template<class Scalar>
void _test_kmeans(int64_t N, int64_t D, int k) {
    Eigen::Matrix<float, Eigen::Dynamic, Dynamic, RowMajor> X(N, D);
    X.setRandom();
    auto centroids_assignments = kmeans(X, k);
    auto centroids = centroids_assignments.first;
    auto assignments = centroids_assignments.second;

    CAPTURE(N);
    CAPTURE(D);
    CAPTURE(k);
    // verify that assigned cluster actually is the closest to each point
    auto dists = dist::squared_dists_to_vectors(X, centroids);
    decltype(X)::Index idx;
    for (int i = 0; i < N; i++) {
        // auto min_idx = ar::argmin(dists.row(i).data(), D);
        CAPTURE(i);
        dists.row(i).minCoeff(&idx);
        REQUIRE(idx == assignments[i]);
    }
}

TEST_CASE("kmeans", "cluster") { // TODO move to own test file
    _test_kmeans<float>(100, 16, 1); // N, D, k
    _test_kmeans<float>(100, 3, 2);
    _test_kmeans<float>(100, 20, 10);
    _test_kmeans<double>(100, 16, 1);
    _test_kmeans<double>(100, 20, 2);
    _test_kmeans<double>(100, 2, 10);
}
