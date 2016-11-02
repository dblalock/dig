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


TEST_CASE("print sizes", "tmp") { // not really a test; just prints sizes
	printf("sizeof(float vector) = %ld\n", sizeof(std::vector<float>));
	printf("sizeof(double vector) = %ld\n", sizeof(std::vector<float>));
	printf("sizeof(eigen Matrix) = %ld\n", sizeof(Eigen::MatrixXd));
	printf("sizeof(eigen ArrayXXd) = %ld\n", sizeof(Eigen::ArrayXXd));
	printf("sizeof(DynamicRowArray<float>) = %ld\n", sizeof(nn::DynamicRowArray<float>));
	printf("sizeof(FixedRowArray<float, 8>) = %ld\n", sizeof(nn::FixedRowArray<float, 8>));
	printf("sizeof(FixedRowArray<float, 16>) = %ld\n", sizeof(nn::FixedRowArray<float, 16>));
	printf("sizeof(L2IndexBrute<float>) = %ld\n", sizeof(nn::L2IndexBrute<float>));
	// printf("sizeof(L2IndexAbandon<float>) = %ld\n", sizeof(nn::L2IndexAbandon<float>));
}

TEST_CASE("L2IndexBrute", "[neighbors]") {
	_test_index<nn::L2IndexBrute<double> >();
	_test_index<nn::L2IndexBrute<float> >();
}
TEST_CASE("L2IndexAbandon", "[neighbors]") {
	// PRINT("running this...");
	_test_index<nn::L2IndexAbandon<double> >();
	_test_index<nn::L2IndexAbandon<float> >();
}
TEST_CASE("L2IndexSimple", "[neighbors]") {
	_test_index<nn::L2IndexSimple<double> >();
	_test_index<nn::L2IndexSimple<float> >();
}

TEST_CASE("NNIndex_IdentityPreproc", "[neighbors]") {
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

TEST_CASE("NNIndex_ReorderPreproc", "[neighbors]") {
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

TEST_CASE("L2KmeansIndex", "[neighbors][kmeans]") {
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

	SECTION("L2KmeansIndex+L2KmeansIndex+L2IndexSimple+IdentityPreproc") {
		using Scalar = float;
		using LeafIndexT = nn::L2IndexSimple<Scalar>;
		using Level1IndexT = nn::L2KmeansIndex<Scalar, LeafIndexT>;
		using IndexT = nn::L2KmeansIndex<Scalar, Level1IndexT>;
		_test_cluster_index<IndexT>(100, 16, 1);
	}
}
