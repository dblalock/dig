//
//  test_tree.cpp
//  Dig
//
//  Created by DB on 10/24/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include <algorithm>
#include <chrono>
#include <math.h>

#include "Dense"

#include "catch.hpp"
#include "testing_utils.hpp"
#include "array_utils.hpp"
#include "debug_utils.hpp"

#include "tree.hpp"

//using namespace std::chrono;
using std::vector;

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

using cputime_t = std::chrono::high_resolution_clock::time_point;
//#define clock std::chrono::high_resolution_clock // because so much typing

cputime_t timeNow() {
	return std::chrono::high_resolution_clock::now();
}

double durationMs(cputime_t t1, cputime_t t0) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

void runRangeTest(int N, int D, double r, depth_t P=16, double binWidth=-1,
				  bool printNeighbors=false, bool printCount=true) {
	r /= sqrt(D);
	
//	double binWidthDivisor = sqrt(D);
//	if (binWidth < 0) {
//		binWidth = r / binWidthDivisor;
//	}
	MatrixXd X(N, D);
	X.setRandom();
	
	// hmm...if we mean-normalize X, redSVD seems to return 0s as projection
	// vects...hopefully this is goes away if X is non-random?
	// -ya, projection vect norms are 1 for first 1 and 0 for all others
	// -although vects themselves appear to be wrong size...
//	VectorXd means = X.rowwise().mean();
//	VectorXd means(D);
//	means.setRandom(); // works
//	means.setConstant(1.); // works
//	RowVectorXd meansAsRow = means.transpose();
//	meansAsRow = (meansAsRow.array() + .001).matrix();
//	X.rowwise() -= meansAsRow;
	
	VectorXd q(D);
	q.setRandom();
//	q = (q.array() + 100).matrix(); // TODO remove
//	q = (q.array() - q.mean()).matrix();

//	INFO("Running range test with N, D, r, P = %ld, %ld, %g, %d", N, D, r, P);
	CAPTURE(N);
	CAPTURE(D);
	CAPTURE(r);
	CAPTURE(P);
	CAPTURE(binWidth);

//	ar::print_with_name(&q[0], q.size(), "q");

	cputime_t t0 = timeNow();
	MatrixXd projectionVects = computeProjectionVects(X, P);
	auto rootPtr = constructIndex(X, projectionVects, binWidth);

	std::cout << "binWidth: " << binWidth << std::endl;

//	std::cout << projectionVects << std::endl; // random nums in [-1, 1]
//	std::cout << "projection vect norms, VVT" << std::endl;
//	std::cout << projectionVects.rowwise().squaredNorm() << std::endl; // all 1s
//	std::cout << projectionVects * projectionVects.transpose() << std::endl;

	cputime_t t1 = timeNow();
	vector<length_t> neighbors = findNeighbors(q, X, *rootPtr, projectionVects, r, binWidth);

	cputime_t t2 = timeNow();

	VectorXd trueDists = squaredDistsToVector(X, q);
	double r2 = r * r;
	vector<length_t> trueNeighbors;
	for (length_t i = 0; i < X.rows(); i++) {
		if (trueDists(i) <= r2) {
			trueNeighbors.push_back(i);
		}
	}

	cputime_t t3 = timeNow();

	auto indexDuration = durationMs(t1, t0);
	auto queryDuration = durationMs(t2, t1);
	auto bruteDuration = durationMs(t3, t2);

	// sanity check neighbors
	REQUIRE(ar::unique(neighbors).size() == neighbors.size());
	REQUIRE(ar::unique(trueNeighbors).size() == trueNeighbors.size());

	// sort true and returned neighbors
	ar::sort(neighbors);
	// ar::sort(neighbors);
	// ar::sort(trueNeighbors);

	// TODO remove
//	ar::print_with_name(neighbors, "neighbors");
//	ar::print_with_name(trueNeighbors, "trueNeighbors");
//	auto true_neighbor_dists = at_idxs(&trueDists[0], trueNeighbors);
//	ar::print_with_name(true_neighbor_dists, "trueDists");

	printf("> found %ld vs %ld neighbors in %g vs %g ms (index %gms)\n",
		   neighbors.size(), trueNeighbors.size(),
		   queryDuration, bruteDuration, indexDuration);

	// compare true and returned neighbors
	REQUIRE(ar::all_eq(neighbors, trueNeighbors));
}

TEST_CASE("notCrashing", "Tree") {
	int nRows = 40;
	int nCols = 10;
	MatrixXd X(nRows, nCols);
	VectorXd q(nCols);
}

TEST_CASE("rangeQueries", "Tree") {
//	srand(123);

	int N = 30;
	int D = 10;
	double r = 10.;
	depth_t P = 4;

//	runRangeTest(N, D, r, P);

	r = 1.;
	N = 10*1000;
	D = 30;
	P = 16;
//	r = 2.;
//	N = 1000;
//	double binWidth = 999.;
	double binWidth = -1; // ok, so almost all 0, but a few nonzero with this
//	double binWidth = .2; // slightly smaller than .31 that above yields
	runRangeTest(N, D, r, P, binWidth);

	for (double r = 1.; r <= 30.; r += 5) {
		runRangeTest(N, D, r, P, binWidth);
	}
}
