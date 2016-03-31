
#include "flock.hpp"

#include <math.h>
#include "Dense"

#include "catch.hpp"
#include "testing_utils.hpp"
#include "array_utils.hpp"
#include "array_utils_eigen.hpp"

// using std::vector;

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

using ar::randwalks;

//#define SKIP_TESTS_FLOCK
#ifndef SKIP_TESTS_FLOCK

TEST_CASE("canCreateObj", "flock") {
	int N = 100;
	int D = 2;

//	MatrixXd X(D, N);
//	X.setRandom();
	MatrixXd X = randwalks(D, N);
	double Lmin = .1;
	double Lmax = .2;

//	FlockLearner ff(X.data(), D, N, Lmin, Lmax);
	FlockLearner ff;
	ff.learn(X.data(), D, N, Lmin, Lmax);

	REQUIRE(ff.getPattern().sum() == ff.getPatternSum());
//	printf((ff.getPattern().sum());

	// REQUIRE(ff.getInstanceStartIdxs());
	// REQUIRE(false);
}

#endif // SKIP_TESTS_FLOCK
