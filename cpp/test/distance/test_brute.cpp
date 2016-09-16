//
//  test_brute.cpp
//  Dig
//
//  Created by DB on 4/22/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "brute.hpp"
#include "catch.hpp"

#include "Dense"

#include "array_utils.hpp"
#include "testing_utils.hpp"
#include "neighbor_testing_utils.hpp"

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

using ar::dist_sq;

typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;

template<class MatrixT, class VectorT>
void _test_squared_dists_to_vector(const MatrixT& X, const VectorT& q) {
	VectorT dists = squaredDistsToVector(X, q);
	for (int32_t i = 0; i < X.rows(); i++) {
		double simple_dist = dist_sq(X.row(i).data(), q.data(), q.size());
		REQUIRE(approxEq(dists(i), simple_dist));
	}
}

template<class MatrixT>
void _test_squared_dists_to_vectors(const MatrixT& X, const MatrixT& V) {
	MatrixT dists = squaredDistsToVectors(X, V);
	for (int32_t i = 0; i < X.rows(); i++) {
		for (int32_t j = 0; j < V.rows(); j++) {
			double simple_dist = dist_sq(X.row(i).data(), V.row(j).data(),
										 V.row(j).size());
			REQUIRE(approxEq(dists(i, j), simple_dist));
		}
	}
}

TEST_CASE("squaredDistsToVector(s)", "distance") {
	for (int i = 0; i < 5; i++) {
		for (int n = 1; n <= 125; n *= 5) {
			for (int d = 1; d <= 100; d *= 10) {
				RowMatrixXd X(n, d);
				X.setRandom();

				VectorXd q(d);
				q.setRandom();
				_test_squared_dists_to_vector(X, q);

				int num_queries = std::max(n / 10, 1);
				RowMatrixXd V(num_queries, d);
				V.setRandom();
				_test_squared_dists_to_vectors(X, V);
			}
		}
	}
}

template<class MatrixT, class VectorT>
Neighbor onenn_simple(const MatrixT& X, const VectorT& q) {
	Neighbor trueNN;
	double d_bsf = INFINITY;
	for (int32_t i = 0; i < X.rows(); i++) {
		double dist1 = (X.row(i) - q.transpose()).squaredNorm();
		double dist2 = dist_sq(X.row(i).data(), q.data(), q.size());
		REQUIRE(approxEq(dist1, dist2));
		if (dist1 < d_bsf) {
			d_bsf = dist1;
			trueNN = Neighbor{.idx = i, .dist = dist1};
		}
	}
	return trueNN;
}

template<class MatrixT, class VectorT>
vector<Neighbor> knn_simple(const MatrixT& X, const VectorT& q, int k) {
	assert(k <= X.rows());

	vector<Neighbor> trueKnn;
	for (int32_t i = 0; i < k; i++) {
		double dist = (X.row(i) - q.transpose()).squaredNorm();
		trueKnn.push_back(Neighbor{.idx = i, .dist = dist});
	}
	sort_neighbors_ascending_distance(trueKnn);

	double d_bsf = INFINITY;
	for (int32_t i = 0; i < X.rows(); i++) {
		double dist = dist_sq(X.row(i).data(), q.data(), q.size());
		trueKnn.push_back(Neighbor{.idx = i, .dist = dist});
		maybe_insert_neighbor(trueKnn, dist, i);
	}
	return trueKnn;
}


TEST_CASE("L2Index", "distance") {
	int64_t N = 100;
	int64_t D = 20;

	RowMatrixXd X(N, D);
	X.setRandom();

	L2IndexBrute<RowMatrixXd> index(X);

	VectorXd q(D);
	for (int i = 0; i < 1000; i++) {
		q.setRandom();
		auto nn = index.onenn(q);
		auto trueNN = onenn_simple(X, q);
		require_neighbors_same(nn, trueNN);

		for (int k = 1; k < 10; k += 2) {
			auto knn = index.knn(q, k);
			auto trueKnn = knn_simple(X, q, k);
			require_neighbors_same(nn, trueNN);
		}
	}
}



