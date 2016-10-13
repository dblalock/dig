//
//  test_nn_index.cpp
//  Dig
//
//  Created by DB on 4/22/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "nn_index.hpp"
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
using Eigen::RowVectorXd;


typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
typedef Eigen::Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;

template<class MatrixT, class VectorT>
void _test_squared_dists_to_vector(const MatrixT& X, const VectorT& q) {
	VectorT dists = dist::squared_dists_to_vector(X, q);
	for (int32_t i = 0; i < X.rows(); i++) {
		double simple_dist = ar::dist_sq(X.row(i).data(), q.data(), q.size());
		REQUIRE(approxEq(dists(i), simple_dist));
	}
}

template<class MatrixT>
void _test_squared_dists_to_vectors(const MatrixT& X, const MatrixT& V) {
	MatrixT dists = dist::squared_dists_to_vectors(X, V);
	for (int32_t i = 0; i < X.rows(); i++) {
		for (int32_t j = 0; j < V.rows(); j++) {
			double simple_dist = ar::dist_sq(X.row(i).data(), V.row(j).data(),
										 V.row(j).size());
			REQUIRE(approxEq(dists(i, j), simple_dist));
		}
	}
}

// TODO put this in a different test file
TEST_CASE("squared_dists_to_vector(s)", "distance") {
	for (int i = 0; i < 5; i++) {
		for (int n = 1; n <= 125; n *= 5) {
			for (int d = 1; d <= 100; d *= 10) {
				RowMatrixXd X(n, d);
				X.setRandom();

				RowVectorXd q(d);
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


TEST_CASE("print sizes", "tmp") {
	printf("sizeof(float vector) = %ld\n", sizeof(std::vector<float>));
	printf("sizeof(double vector) = %ld\n", sizeof(std::vector<float>));
	printf("sizeof(eigen Matrix) = %ld\n", sizeof(Eigen::MatrixXd));
	printf("sizeof(eigen ArrayXXd) = %ld\n", sizeof(Eigen::ArrayXXd));
	printf("sizeof(DynamicRowArray<float>) = %ld\n", sizeof(nn::DynamicRowArray<float>));
	printf("sizeof(FixedRowArray<float, 8>) = %ld\n", sizeof(nn::FixedRowArray<float, 8>));
	printf("sizeof(FixedRowArray<float, 16>) = %ld\n", sizeof(nn::FixedRowArray<float, 16>));
	printf("sizeof(L2IndexBrute<float>) = %ld\n", sizeof(nn::L2IndexBrute<float>));
	printf("sizeof(L2IndexAbandon<float>) = %ld\n", sizeof(nn::L2IndexAbandon<float>));
}

TEST_CASE("L2IndexBrute", "neighbors") {
	_test_index<nn::L2IndexBrute<double> >();
	_test_index<nn::L2IndexBrute<float> >();
}
TEST_CASE("L2IndexAbandon", "neighbors") {
	_test_index<nn::L2IndexAbandon<double> >();
	_test_index<nn::L2IndexAbandon<float> >();
}

TEST_CASE("NNIndex_IdentityPreproc", "neighbors") {
	using PreprocT = nn::IdentityPreproc;
	SECTION("L2IndexAbandon") {
		SECTION("float") {
			using IndexT = nn::L2IndexAbandon<float, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 24);
		}
		SECTION("double") {
			using IndexT = nn::L2IndexAbandon<double, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 24);
		}
	}
	SECTION("L2IndexBrute") {
		SECTION("float") {
			using IndexT = nn::L2IndexBrute<float, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 10);
			_test_index<IndexT>(100, 24);
		}
		SECTION("double") {
			using IndexT = nn::L2IndexBrute<double, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 10);
			_test_index<IndexT>(100, 24);
		}
	}
}

TEST_CASE("NNIndex_ReorderPreproc", "neighbors") {
	SECTION("L2IndexBrute") {
		SECTION("float") {
			using PreprocT = nn::ReorderPreproc<float>;
			using IndexT = nn::L2IndexBrute<float, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 20); // D=20
		}
		SECTION("double") {
			using PreprocT = nn::ReorderPreproc<double>;
			using IndexT = nn::L2IndexBrute<double, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 20); // D=20
		}
	}
	SECTION("L2IndexAbandon") {
		SECTION("float") {
			using PreprocT = nn::ReorderPreproc<float>;
			using IndexT = nn::L2IndexAbandon<float, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 20); // D=20
		}
		SECTION("double") {
			using PreprocT = nn::ReorderPreproc<double>;
			using IndexT = nn::L2IndexAbandon<double, PreprocT>;
			_test_index<IndexT>();
			_test_index<IndexT>(100, 20); // D=20
		}
	}
}

TEST_CASE("KmeansIndex", "neighbors") {
	SECTION("L2IndexBrute+IdentityPreproc") {
		using Scalar = float;
		using ClusterIndexT = nn::L2IndexBrute<Scalar>;
		using PreprocT = nn::IdentityPreproc;
		using IndexT = nn::L2KmeansIndex<Scalar, ClusterIndexT, PreprocT>;
		_test_cluster_index<IndexT>(100, 16, 1);
		_test_cluster_index<IndexT>(100, 20, 2);
		_test_cluster_index<IndexT>(100, 10, 7);
	}
	SECTION("L2IndexBrute+ReorderPreproc") {
		using Scalar = double;
		using ClusterIndexT = nn::L2IndexBrute<Scalar>;
		using PreprocT = nn::ReorderPreproc<Scalar>;
		using IndexT = nn::L2KmeansIndex<Scalar, ClusterIndexT, PreprocT>;
		_test_cluster_index<IndexT>(100, 16, 1);
		_test_cluster_index<IndexT>(100, 20, 2);
		_test_cluster_index<IndexT>(100, 10, 7);
	}
	SECTION("L2IndexAbandon+ReorderPreproc") {
		using Scalar = float;
		using ClusterIndexT = nn::L2IndexAbandon<Scalar>;
		using PreprocT = nn::ReorderPreproc<Scalar>;
		using IndexT = nn::L2KmeansIndex<Scalar, ClusterIndexT, PreprocT>;
		_test_cluster_index<IndexT>(100, 16, 1);
		_test_cluster_index<IndexT>(100, 20, 2);
		_test_cluster_index<IndexT>(100, 10, 7);
	}
}
