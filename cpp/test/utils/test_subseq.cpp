//
//  test_subseq.cpp
//  Dig
//
//  Created by DB on 3/20/16
//  Copyright (c) 2016 DB. All rights reserved.
//

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
using ar::to_string;
using ar::at_idxs;

TEST_CASE("crossCorrs", "subseq") {

	SECTION("lengths 1,1") {
		vecd q {3};
		vecd x {2};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(out[0] == 6.);
	}
	SECTION("lengths 1,2") {
		vecd q {3};
		vecd x {2, -1};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(out[0] == 6.);
		REQUIRE(out[1] == -3.);
	}
	SECTION("lengths 2,2") {
		vecd q {3, 5};
		vecd x {2, -1};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(out[0] == 1.);
	}
	SECTION("lengths 2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {1, -3, 20};

		auto out = crossCorrs(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4, stride 2") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {1, 20};

		auto out = crossCorrs(q, x, 2);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("dist_L1", "subseq") {
	SECTION("lengths 1,1") {
		vecd q {3};
		vecd x {2};
		vecd ans {1};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 1,2") {
		vecd q {3};
		vecd x {2, -1};
		vecd ans {1, 4};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,2") {
		vecd q {3, 5};
		vecd x {2, -1};
		vecd ans {7};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {7, 9, 4};

		auto out = dists_L1(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4, stride 2") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {7, 4};

		auto out = dists_L1(q, x, 2);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("dist_sq", "subseq") {
	SECTION("lengths 1,1") {
		vecd q {3};
		vecd x {2};
		vecd ans {1};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 1,2") {
		vecd q {3};
		vecd x {2, -1};
		vecd ans {1, 16};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,2") {
		vecd q {3, 5};
		vecd x {2, -1};
		vecd ans {37};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {37, 41, 10};

		auto out = dists_sq(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4, stride 2") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {37, 10};

		auto out = dists_sq(q, x, 2);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("dist_L2", "subseq") {
	SECTION("lengths 1,1") {
		vecd q {3};
		vecd x {2};
		vecd ans {1};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 1,2") {
		vecd q {3};
		vecd x {2, -1};
		vecd ans {1, sqrt(16)};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,2") {
		vecd q {3, 5};
		vecd x {2, -1};
		vecd ans {sqrt(37)};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {sqrt(37), sqrt(41), sqrt(10)};

		auto out = dists_L2(q, x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 2,4, stride 2") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {sqrt(37), sqrt(10)};

		auto out = dists_L2(q, x, 2);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
}

TEST_CASE("first_discrete_deriv", "subseq") {
	SECTION("length 2") {
		vecd x {2, -1};
		vecd ans {-3};

		auto out = first_derivs(x);

		REQUIRE(out.size() == 1);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("lengths 4") {
		vecd x {2, -1, 0, 4};
		vecd ans {-3, 1, 4};

		auto out = first_derivs(x);

		REQUIRE(out.size() == 3);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("length 4, stride 2") {
		vecd q {3, 5};
		vecd x {2, -1, 0, 4};
		vecd ans {-3, 4};

		auto out = first_derivs(x, 2);

		REQUIRE(out.size() == 2);
		REQUIRE(all_eq(out, ans));
	}
}


TEST_CASE("local_maxima", "subseq") {
	SECTION("length 1") {
		vecd v {-1};
		vecd ans {0};
		auto out = local_maxima(v);
		INFO(to_string(out));
		REQUIRE(all_eq(out, ans));
	}
	SECTION("both endpoints") {
		vecd v {1, 0, 1};
		vecd ans {0, 2};
		auto out = local_maxima(v);
		INFO(to_string(out));
		REQUIRE(all_eq(out, ans));
	}
	SECTION("first endpoint") {
		vecd v {1, 0, -2, 1, .8};
		vecd ans {0, 3};
		auto out = local_maxima(v);
		INFO(to_string(out));
		REQUIRE(all_eq(out, ans));
	}
	SECTION("second endpoint") {
		vecd v {-3, 0, -2, 1, .8, 6};
		vecd ans {1, 3, 5};
		auto out = local_maxima(v);
		INFO(to_string(out));
		REQUIRE(all_eq(out, ans));
	}
	SECTION("neither endpoint") {
		vecd v {-9.0, 1, 0, -2, 1, .8};
		vecd ans {1, 4};
		auto out = local_maxima(v);
		INFO(to_string(out));
		REQUIRE(all_eq(out, ans));
	}
	SECTION("dot products known failure excerpt", "subseq") {
		vecd v {1.91, 2.14, 2.29, (2.29), 2.01};
		vecd ans {3};
		auto out = local_maxima(v, 1);
		CAPTURE(to_string(out));
		CAPTURE(to_string(ans));
		CAPTURE(to_string(at_idxs(v, out)));
		CAPTURE(to_string(at_idxs(v, ans)));
		REQUIRE(all_eq(out, ans));
	}
	SECTION("dot products known failure excerpt simplified", "subseq") {
		vecd v {.5, 1, 2.8, 2.9, 2.0};
//		vecd v {0, 1, 2, 2, 1};
//		vecd v {.5, 1, 2.8, 2.9, 2.0}; //still fails
//		vecd v {.5, 1, 2.8, 2.9, 1.99}; //passes
//		vecd v {.5, 1, 2.8, 2.9, 1.91}; //passes
//		vecd v {.5, 1, 2.8, 2.9, 1.9}; //passes
//		vecd v {.5, 1, 2.8, 2.9, 2.0}; //fails
//		vecd v {.5, 1, 2.8, 2.9, 2}; //fails
//		vecd v {.5, 1, 2.9, 2.9, 2}; //fails
//		vecd v {.5, 1, 3.0, 3.0, 2}; //passes
//		vecd v {.5, 1, 3, 3, 2}; // passes
//		vecd v {.5, 1, 2.5, 2.5, 2}; // fails
//		vecd v {.5, 1, 2.29, 2.29, 2.0}; // fails
//		vecd v {.5, 1, 2.29, 2.29, 2.01}; // fails
		vecd ans {3};
		auto out = local_maxima(v, 1);
		CAPTURE(to_string(out));
		CAPTURE(to_string(ans));
		CAPTURE(to_string(at_idxs(v, out)));
		CAPTURE(to_string(at_idxs(v, ans)));
		REQUIRE(all_eq(out, ans));
	}

	SECTION("dot products known failure", "subseq") {
		vecd v {
			(2.9), 2.22, 1.56, 1.33, 1.13, 1.04, 1.04, 1.18, 1.43, 1.82,  // 10
			2.04, (2.13),1.93, 1.66, 1.33, 1.19, 1.07, 1.06, 1.06, 1.12,  // 20
			1.22,  1.42, 1.68, 1.91, 2.14, 2.29,(2.29),2.01, 1.59, 0.986, // 30
			0.503,0.192,0.0542,  0,    0,    0,    0, 0.111,0.26, 0.682,  // 40
			1.17, 1.6,  (1.77), 1.59, 1.16,0.677,0.36,0.101,0.0799,0.0729,// 50
			0.264, 0.503, 0.947, 1.37, 1.88, 2.21, 2.64, 2.97,(3.05),2.84,// 60
			2.26, 1.59, 0.936, 0.606, 0.279, 0.161, 0.059, 0.0303, 0, 0,  // 70
			0, 0, 0, 0, 0, 0, 0.0543, 0.124, (0.305)};					  // 79
		veci ans {0, 11, 26, 42, 58, 78};

		auto out = local_maxima(v, 1);
		CAPTURE(to_string(out));
		CAPTURE(to_string(ans));
		CAPTURE(to_string(at_idxs(v, out)));
		CAPTURE(to_string(at_idxs(v, ans)));
		REQUIRE(all_eq(out, ans));

		// we don't include 42 because there's a larger value within 12 of
		// it, even though there isn't a larger maximum within 12 of it
		veci ans2 {0, 26, 58, 78};
		auto out2 = local_maxima(v, 12);
		CAPTURE(to_string(out2));
		CAPTURE(to_string(at_idxs(v, out2)));
		CAPTURE(to_string(at_idxs(v, ans2)));
		REQUIRE(all_eq(out2, ans2));
	}
}



