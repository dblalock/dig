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
	double binWidthDivisor = sqrt(D);
	if (binWidth < 0) {
		binWidth = r / binWidthDivisor;
	}
	MatrixXd X(N, D);
	X.setRandom();

	VectorXd q(D);
	q.setRandom();

	cputime_t t0 = timeNow();
	MatrixXd projectionVects = computeProjectionVects(X, P);
	auto rootPtr = constructIndex(X, projectionVects, binWidth);

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
	REQUIRE(array_unique(neighbors).size() == neighbors.size());
	REQUIRE(array_unique(trueNeighbors).size() == trueNeighbors.size());

	// sort true and returned neighbors
	std::sort(std::begin(neighbors), std::end(neighbors));
	// array_sort(neighbors);
	// array_sort(trueNeighbors);

	// TODO remove
	array_print_with_name(neighbors, "neighbors");
	array_print_with_name(trueNeighbors, "trueNeighbors");
	
	printf("found %ld vs %ld neighbors in %g vs %g ms (index %gms)\n",
		   neighbors.size(), trueNeighbors.size(),
		   queryDuration, bruteDuration, indexDuration);
	
	// compare true and returned neighbors
	REQUIRE(array_equal(neighbors, trueNeighbors));
}

TEST_CASE("notCrashing", "Tree") {
	int nRows = 40;
	int nCols = 10;
	MatrixXd X(nRows, nCols);
	VectorXd q(nCols);
}

TEST_CASE("rangeQueries", "Tree") {
	int N = 100;
	int D = 10;
	double r = 10.;
	depth_t P=16;

	runRangeTest(N, D, r, P);
}
