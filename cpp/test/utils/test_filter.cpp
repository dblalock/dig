//
//  test_filter.cpp
//  Dig
//
//  Created by DB on 3/30/16
//  Copyright (c) 2016 DB. All rights reserved.
//

// #include <algorithm>
// #include <math.h>

#include "filter.hpp"

#include <vector>
#include "catch.hpp"

#include "testing_utils.hpp"
#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "debug_utils.hpp"

typedef std::vector<int> veci;
typedef std::vector<double> vecd;

// using Eigen::MatrixXd;
// using Eigen::VectorXd;

using namespace filter;

using ar::all_eq;
using ar::to_string;

TEST_CASE("min_max_filter", "subseq") {
	SECTION("filter length 0") {
		int r = 0;
		int len = 17;
		int data[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};

		int lTest[len];
		int uTest[len];
		min_max_filter(data, len, r, lTest, uTest);

		CAPTURE(to_string(lTest, len));
		CAPTURE(to_string(data, len));
		REQUIRE(all_eq(lTest, data, len));

		CAPTURE(to_string(uTest, len));
		CAPTURE(to_string(data, len));
		REQUIRE(all_eq(uTest, data, len));
	}
	SECTION("filter length 2") {
		int r = 2;
		int len = 17;
		float x[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};
		float l[] = {0, 0, 0, 0, 0, 1, 2, 1, 0, 0,-2,-3,-3,-3,-3,-3,-2};
		float u[] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, 0, 0};

		float lTest[len];
		float uTest[len];
		min_max_filter(x, len, r, lTest, uTest);

		CAPTURE(to_string(lTest, len));
		CAPTURE(to_string(l, len));
		REQUIRE(all_eq(lTest, l, len));

		CAPTURE(to_string(uTest, len));
		CAPTURE(to_string(u, len));
		REQUIRE(all_eq(uTest, u, len));
	}
}

TEST_CASE("min_filter", "subseq") {
	SECTION("filter length 0") {
		int r = 0;
		int len = 17;
		int data[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};

		int lTest[len];
		min_filter(data, len, r, lTest);

		CAPTURE(to_string(lTest, len));
		CAPTURE(to_string(data, len));
		REQUIRE(all_eq(lTest, data, len));
	}
	SECTION("filter length 2") {
		int r = 2;
		int len = 17;
		float x[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};
		float l[] = {0, 0, 0, 0, 0, 1, 2, 1, 0, 0,-2,-3,-3,-3,-3,-3,-2};

		float lTest[len];
		min_filter(x, len, r, lTest);

		CAPTURE(to_string(lTest, len));
		CAPTURE(to_string(l, len));
		REQUIRE(all_eq(lTest, l, len));
	}
}

TEST_CASE("max_filter", "subseq") {
	SECTION("filter length 0") {
		int r = 0;
		int len = 17;
		int data[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};

		int uTest[len];
		max_filter(data, len, r, uTest);

		CAPTURE(to_string(uTest, len));
		CAPTURE(to_string(data, len));
		REQUIRE(all_eq(uTest, data, len));
	}
	SECTION("filter length 2") {
		int r = 2;
		int len = 17;
		float data[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};
		float u[] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, 0, 0};

		float uTest[len];
		max_filter(data, len, r, uTest);

		CAPTURE(to_string(uTest, len));
		CAPTURE(to_string(u, len));
		REQUIRE(all_eq(uTest, u, len));
	}
}
