//
//  test_subseq.cpp
//  Dig
//
//  Created by DB on 10/24/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

// #include <algorithm>
// #include <math.h>

#include "subseq.hpp"

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

 using namespace subs;

using ar::all_eq;

TEST_CASE("crossCorrs", "subseq") {

	SECTION("1,1") {
		vecd q {3};
		vecd x {2};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(out[0] == 6.);
	}
	SECTION("1,2") {
		vecd q {3};
		vecd x {2, -1};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(out[0] == 6.);
		REQUIRE(out[1] == -3.);
	}
	SECTION("2,2") {
		vecd q {3, 5};
		vecd x {2, -1};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(out[0] == 1.);
	}
	SECTION("2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {1, -3, 20};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("dist_L1", "subseq") {

	SECTION("1,1") {
		vecd q {3};
		vecd x {2};
		vecd ans {1};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("1,2") {
		vecd q {3};
		vecd x {2, -1};
		vecd ans {1, 4};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("2,2") {
		vecd q {3, 5};
		vecd x {2, -1};
		vecd ans {7};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {7, 9, 4};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("dist_sq", "subseq") {

	SECTION("1,1") {
		vecd q {3};
		vecd x {2};
		vecd ans {1};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("1,2") {
		vecd q {3};
		vecd x {2, -1};
		vecd ans {1, 16};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("2,2") {
		vecd q {3, 5};
		vecd x {2, -1};
		vecd ans {37};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {37, 41, 10};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("dist_L2", "subseq") {

	SECTION("1,1") {
		vecd q {3};
		vecd x {2};
		vecd ans {1};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("1,2") {
		vecd q {3};
		vecd x {2, -1};
		vecd ans {1, sqrt(16)};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("2,2") {
		vecd q {3, 5};
		vecd x {2, -1};
		vecd ans {sqrt(37)};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {sqrt(37), sqrt(41), sqrt(10)};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
}