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
	r *= sqrt(D);
	
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

	CAPTURE(binWidth); // set by constructIndex
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

	// sort true and returned neighbors
	ar::sort(neighbors);
	// ar::sort(neighbors);
	// ar::sort(trueNeighbors);
	
	if (trueNeighbors.size() < 50) {
		ar::print_with_name(neighbors, "neighbors");
		ar::print_with_name(trueNeighbors, "trueNeighbors");
		//	auto true_neighbor_dists = ar::at_idxs(&trueDists[0], trueNeighbors);
		//	ar::print_with_name(true_neighbor_dists, "trueDists");
	}
	
	// sanity check neighbors
	REQUIRE(ar::unique(neighbors).size() == neighbors.size());
	REQUIRE(ar::unique(trueNeighbors).size() == trueNeighbors.size());

	printf("> found %ld vs %ld neighbors in %g vs %g ms (index %gms)\n",
		   neighbors.size(), trueNeighbors.size(),
		   queryDuration, bruteDuration, indexDuration);

	// compare true and returned neighbors
	REQUIRE(ar::all_eq(neighbors, trueNeighbors));
}

void run1nnTest(int N, int D, depth_t P=16, double binWidth=-1) {
	
	MatrixXd X(N, D);
	X.setRandom();
	VectorXd q(D);
	q.setRandom();
	
	CAPTURE(N);
	CAPTURE(D);
	CAPTURE(P);
	CAPTURE(binWidth);
	
	cputime_t t0 = timeNow();
	MatrixXd projectionVects = computeProjectionVects(X, P);
	auto rootPtr = constructIndex(X, projectionVects, binWidth);
	
	CAPTURE(binWidth); // set by constructIndex
	
	cputime_t t1 = timeNow();
	auto nn = find1nn(q, X, *rootPtr, projectionVects, binWidth);
	
	cputime_t t2 = timeNow();
	
	VectorXd trueDists = squaredDistsToVector(X, q);
	Neighbor trueNN;
	double d_bsf = INFINITY;
	for (length_t i = 0; i < X.rows(); i++) {
		if (trueDists(i) < d_bsf) {
			d_bsf = trueDists(i);
			trueNN = Neighbor{.idx = i, .dist = trueDists(i)};
		}
	}
	
	CAPTURE(ar::to_string(&trueDists[0], trueDists.size()));
	
	cputime_t t3 = timeNow();
	
	auto indexDuration = durationMs(t1, t0);
	auto queryDuration = durationMs(t2, t1);
	auto bruteDuration = durationMs(t3, t2);
	
	printf("%dx%d, %d: %d vs %d (%g vs %g)\t\t%g vs %g ms (index %gms)\n",
		   N, D, P, nn.idx, trueNN.idx, nn.dist, trueNN.dist,
		   queryDuration, bruteDuration, indexDuration);
	
	REQUIRE(nn.idx == trueNN.idx);
	REQUIRE(std::abs(nn.idx - trueNN.idx) < .001);
//	REQUIRE(nn.dist == trueNN.dist);
}

void runKnnTest(int N, int D, depth_t P=16, int k=1, double binWidth=-1) {
	
	MatrixXd X(N, D);
	X.setRandom();
	VectorXd q(D);
	q.setRandom();
	
	CAPTURE(N);
	CAPTURE(D);
	CAPTURE(P);
	CAPTURE(binWidth);
	
	cputime_t t0 = timeNow();
	MatrixXd projectionVects = computeProjectionVects(X, P);
	auto rootPtr = constructIndex(X, projectionVects, binWidth);
	
	CAPTURE(binWidth); // set by constructIndex
	
	cputime_t t1 = timeNow();
	vector<Neighbor> neighbors = findKnn(q, X, k, *rootPtr, projectionVects, binWidth);
//	printf("knn returned %ld neighbors\n", neighbors.size());
//	for (int i = 0; i < neighbors.size(); i++) {
//		printf("%d: %g, ", neighbors[i].idx, neighbors[i].dist);
//	}
//	printf("\n");
	
	cputime_t t2 = timeNow();
	
	// compute true knn by brute force
	VectorXd trueDists = squaredDistsToVector(X, q);
//	std::cout << "trueDists: " << trueDists << "\n"; // TODO remove
	vector<Neighbor> trueNeighbors;
	for (length_t i = 0; i < k; i++) {
		trueNeighbors.push_back(Neighbor{.idx=i, .dist=trueDists(i)});
	}
	std::sort(std::begin(trueNeighbors), std::end(trueNeighbors), // sort first k
			  [](const Neighbor& n1, const Neighbor& n2) {
				  return n1.dist < n2.dist;
			  });
	
//	printf("sorted true neighbors:\n\t");
//	for (int i = 0; i < trueNeighbors.size(); i++) {
//		printf("%d: %g, ", trueNeighbors[i].idx, trueNeighbors[i].dist);
//	}
//	printf("\n");
	
	double d_bsf = trueNeighbors[k-1].dist;
//	printf("initial d_bsf: %g\n", d_bsf);
	for (length_t i = k; i < X.rows(); i++) { // find true knn
		double dist = trueDists(i);
//		printf("%g (%g)\n", dist, d_bsf);
		if (dist < d_bsf) { // found a closer point than current kth nn
			// propagate point down to the appropriate index
			int j = k-1;
			trueNeighbors[j] = Neighbor{.idx=i, .dist=dist};
			while (j > 0 && trueNeighbors[j-1].dist > dist) {
				Neighbor tmp = trueNeighbors[j-1];
				trueNeighbors[j-1] = trueNeighbors[j];
				trueNeighbors[j] = tmp;
				j--;
			}
			d_bsf = trueNeighbors[k-1].dist;
		}
	}
	
	cputime_t t3 = timeNow();
	
	auto indexDuration = durationMs(t1, t0);
	auto queryDuration = durationMs(t2, t1);
	auto bruteDuration = durationMs(t3, t2);
	
	vector<length_t> neighborIdxs = ar::map([](const Neighbor& n) {
		return n.idx;
	}, neighbors);
	vector<length_t> trueNeighborIdxs = ar::map([](const Neighbor& n) {
		return n.idx;
	}, trueNeighbors);
	vector<double> neighborDists = ar::map([](const Neighbor& n) {
		return n.dist;
	}, neighbors);
	vector<double> trueNeighborDists = ar::map([](const Neighbor& n) {
		return n.dist;
	}, trueNeighbors);
	
	if (trueNeighbors.size() < 50) {
		ar::print_with_name(neighborIdxs , "neighbors");
		ar::print_with_name(trueNeighborIdxs, "trueNeighbors");
		ar::print_with_name(neighborDists , "neighborDists");
		ar::print_with_name(trueNeighborDists, "trueNeighborDists");
	}

	// sanity check neighbors
	REQUIRE(ar::unique(neighborIdxs).size() == neighbors.size());
	REQUIRE(ar::unique(trueNeighborIdxs).size() == trueNeighbors.size());

	printf("> found %ld vs %ld neighbors in %g vs %g ms (index %gms)\n",
		   neighbors.size(), trueNeighbors.size(),
		   queryDuration, bruteDuration, indexDuration);
	
	// compare true and returned neighbors
	REQUIRE(ar::all_eq(neighborIdxs, trueNeighborIdxs));
}

TEST_CASE("notCrashing", "Tree") {
	int nRows = 40;
	int nCols = 10;
	MatrixXd X(nRows, nCols);
	VectorXd q(nCols);
}

TEST_CASE("knnQueries", "Tree") {
//	srand(123);
	
	int N = 30;
	int D = 10;
	int K = 1;
	depth_t P = 4;
	
//	runKnnTest(N, D, P, K);
	for (int i = 0; i < 10; i++) {
		runKnnTest(N, D, P, K);
	}
	
	K = 2;
//	runKnnTest(N, D, P, K);
	for (int i = 0; i < 10; i++) {
		runKnnTest(N, D, P, K);
	}
	
	K = 3;
	//	runKnnTest(N, D, P, K);
	for (int i = 0; i < 10; i++) {
		runKnnTest(N, D, P, K);
	}
	
	N = 1000;
	K = 10;
	//	runKnnTest(N, D, P, K);
	for (int i = 0; i < 10; i++) {
		runKnnTest(N, D, P, K);
	}
	
//	N = 10000;
//	K = 5;
//	//	runKnnTest(N, D, P, K);
//	for (int i = 0; i < 10; i++) {
//		runKnnTest(N, D, P, K);
//	}
}

TEST_CASE("1nnQueries", "Tree") {
//	srand(123);
	
	int N = 30;
	int D = 10;
	depth_t P = 4;
	
	run1nnTest(N, D, P);
	
	P = 64;
	for (int N = 100.; N <= 1*1000; N *= 10) {
		for (int D = 10; D <= 100; D *= 10) {
//		for (int D = 5; D <= 125; D *= 5) {
			for (int P = 4; P <= 64; P *= 2) {
				run1nnTest(N, D, (int)fmin(D/2, P));
			}
		}
	}
}

TEST_CASE("rangeQueries", "Tree") {
//	srand(123);

	int N = 30;
	int D = 10;
	double r = 10.;
	depth_t P = 4;

	runRangeTest(N, D, r, P);

	r = 1.;
	N = 1000;
//	N = 10*1000;
//	N = 100*1000;
	D = 30;
//	D = 100;
	P = 16;
//	r = 2.;
//	N = 1000;
//	double binWidth = 999.;
	double binWidth = -1; // ok, so almost all 0, but a few nonzero with this
//	double binWidth = .2; // slightly smaller than .31 that above yields
	runRangeTest(N, D, r, P, binWidth);

	for (double r = .1; r <= 1.; r += .2) {
		runRangeTest(N, D, r, P, binWidth);
	}
	
//	N = 100*1000;
//	for (double r = .1; r <= 1.; r += .2) {
//		runRangeTest(N, D, r, P, binWidth);
//	}
	
//	N = 1000*1000;
//	for (double r = .1; r <= 1.; r += .2) {
//		runRangeTest(N, D, r, P, binWidth);
//	}
	
//	P = 8;
//	for (double r = 1.; r <= 30.; r += 5) {
//		runRangeTest(N, D, r, P, binWidth);
//	}
}
