//  tree.cpp
//
//  Dig
//
//  Created by DB on 1/24/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

// TODO have nodes use a fixed-size array for points
// TODO have insert just pass on the extra element when it needs to get split,
// not append it to the points first
// TODO have points and children be in a union

#include "tree.hpp"
#include <map> // TODO remove
#include <cmath>

#include "redsvd.hpp"
#include "array_utils.hpp" // for rand_ints

#define DEBUG_POINT 2997

// ================================================================
// Typedefs and usings
// ================================================================

typedef Eigen::Matrix<hash_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HashMat;
typedef Eigen::Matrix<hash_t, Eigen::Dynamic, 1> HashVect;

// using std::unordered_map;
using std::map;
using RedSVD::RedSVD;

// ================================================================
// Constants
// ================================================================

const hash_t MAX_HASH_VALUE = 127; // int8_t
const hash_t MIN_HASH_VALUE = -128;
// const hash_t MAX_HASH_VALUE = 32767; // int16_t
// const hash_t MIN_HASH_VALUE = -32768;
const double TARGET_HASH_SPREAD_STDS = 3.0; // +/- 3 std deviations
// const double TARGET_HASH_SPREAD_STDS = 2.0; // +/- this many std deviations
// const double TARGET_HASH_SPREAD_STDS = 1.0; // +/- this many std deviations

const uint16_t MAX_POINTS_PER_LEAF = 8;

// ================================================================
// Structs
// ================================================================

Node::Node() :
	// children(unordered_map<hash_t, Node>()),
	children(map_t()),
	points(vector<length_t>()),
	is_internal(false)
{
	points.reserve(MAX_POINTS_PER_LEAF + 1);
//	points.shrink_to_fit();
}

//Node::Node(Node& other) :
//	children(other.children),
//	points(other.points),
//	is_internal(other.is_internal)
//{}


// ================================================================
#pragma mark Utils
// ================================================================

double squaredL2Dist(VectorXd x, VectorXd y) {
	VectorXd diff = x - y;
	return diff.squaredNorm();
}

// ================================================================
#pragma mark Index construction
// ================================================================

template <typename T, typename U>
U clamp(const T x, const U lower, const U upper) {
	if (x > upper) {
		return upper;
	} else if (x < lower) {
		return lower;
	}
	return static_cast<U>(x);
}

double computeBinWidth(const MatrixXd& positions) {
	// assumes first col of positions corresponds to dominant eigenvect
	ArrayXd firstCol = positions.col(0).array();
	firstCol -= firstCol.mean();
	double SSE = firstCol.matrix().squaredNorm();
	double variance = SSE / firstCol.size();
	double std = sqrt(variance);
	double targetBinsPerStd = MAX_HASH_VALUE / TARGET_HASH_SPREAD_STDS;
	return std / targetBinsPerStd;
}

MatrixXd computeProjectionVects(const MatrixXd& X, depth_t numVects) {
	std::cout << "starting pca...\n";
	RedSVD<MatrixXd> svd(X, numVects);
	std::cout << "finished pca\n";
	return svd.matrixV().transpose(); // cols should be eigenvectors -> rows eigenvectors
	// TODO don't even compute matrixU--comment out the flag that tells it to
}

HashMat binsForPositions(const MatrixXd& positions, double binWidth) {
//	HashMat bins = positions.cast<hash_t>();
	HashMat bins(positions.rows(), positions.cols());
//	for (size_t i = 0; i < positions.size(); i++) {
	for (size_t j = 0; j < bins.cols(); j++) { // column major order
		for (size_t i = 0; i < bins.rows(); i++) {
			int64_t bin = (int64_t) (positions(i, j) / binWidth);
			bins(i, j) = clamp(bin, MIN_HASH_VALUE, MAX_HASH_VALUE);
		}
	}
	return bins;
}

vector<hash_t> binsForVectPositions(const VectorXd& positions, double binWidth) {
	size_t ndims = positions.size();
	vector<hash_t> bins;
	// std::cout << "binsForVectPositions, positions: " << positions << std::endl;
	// std::cout << "binsForVectPositions, ndims: " << ndims << std::endl;
	// std::cout << "binsForVectPositions, binWidth: " << binWidth << std::endl;
	for (size_t i = 0; i < ndims; i++) {
		int64_t bin = (int64_t) (positions(i) / binWidth);
		// printf("projection #%lu: bin = %lld\n", i, bin);
		bins.push_back(clamp(bin, MIN_HASH_VALUE, MAX_HASH_VALUE));
	}
	return bins;
}

VectorXd squaredDistsToVector(const MatrixXd& X, const VectorXd& v) {
	auto diffs = X.rowwise() - v.transpose();
	return diffs.rowwise().squaredNorm();
}
VectorXd distsToVector(const MatrixXd& X, const VectorXd& v) {
	return squaredDistsToVector(X, v).array().sqrt();
}

ArrayXd squaredDistsToVectors(const MatrixXd& X, const MatrixXd& V) {
	auto prods = -2. * (X * V.transpose());
	MatrixXd dists = prods;
	VectorXd rowSquaredNorms = X.rowwise().squaredNorm();
	VectorXd colSquaredNorms = V.rowwise().squaredNorm();
	RowVectorXd colSquaredNormsAsRow = colSquaredNorms.transpose();
	dists.colwise() += rowSquaredNorms;
	dists.rowwise() += colSquaredNormsAsRow;

	// // does the above compute the distances properly? -> yes
	// ArrayXd trueDists = ArrayXd(X.rows(), V.rows());
	// for (int i = 0; i < X.rows(); i++) {
	// 	for (int j = 0; j < V.rows(); j++) {
	// 		VectorXd diff = X.row(i) - V.row(j);
	// 		trueDists(i, j) = diff.squaredNorm();

	// 		auto gap = fabs(trueDists(i, j) - dists(i, j));
	// 		if (gap > .001) {
	// 			printf("WE'RE COMPUTING THE DISTANCES WRONG!!!");
	// 		}
	// 		assert(gap < .001);
	// 	}
	// }

	return dists.array();
}

ArrayXd distsToVectors(const MatrixXd& X, const MatrixXd& V) {
	return squaredDistsToVectors(X, V).sqrt();
}

pair<Node*, depth_t> leafNodeForPointBins(Node* table, const HashVect& bins) {
	depth_t dim = 0;
	// Node* node = &table;
	// printf("----- received table %p\n", table);
	Node* node = table;
	while (node->is_internal) {
		// printf("----- find leaf node loop: examining node %p\n", node);
		hash_t key = bins(dim);
//		node.children.emplace(key); // create key, Node() entry // TODO uncomment
		node->children.emplace(key, Node{}); // creates Node() iff key not present // TODO uncommment
		node = &(node->children[key]); // get ptr to the created node
		dim++;
	}

	return pair<Node*, depth_t>(node, dim);
}

void splitLeafNode(Node* node, depth_t nodeDepth, const HashMat& allBins) {
	//TODO go back to pass by ref after debug

	node->is_internal = true;
	auto points = node->points;

	// std::cout << "splitting node " << node << " with " << points.size() << " points" << std::endl;

	if (ar::contains(points, DEBUG_POINT)) {
		printf("splitting node %p containing point %d\n", node, DEBUG_POINT);
	}

	auto depth = nodeDepth;
	for (length_t point : points) {
		HashVect bins = allBins.row(point); // point is an index
		hash_t key = bins[depth];

//		node.children.emplace(key); // creates Node() iff key not present // TODO uncommment
		if (! node->children.count(key)) { // TODO remove
			node->children.emplace(key, Node{}); // creates Node() iff key not present // TODO uncommment
		}

		Node& child = node->children[key];
		child.points.push_back(point);

//		// TODO remove
//		if (key == DEBUG_POINT) {
//			printf("moved node %d to child %p at depth %d", DEBUG_POINT, &child, depth+1);
//		}
	}
}

void insert(Node* table, const HashVect& bins, length_t id,
	const HashMat& allBins) {

	// printf("========\n");

	auto nodeAndDepth = leafNodeForPointBins(table, bins);
	Node* node = nodeAndDepth.first;
	depth_t depth = nodeAndDepth.second;

	node->points.push_back(id); // append point to list

	// // here's the problem: this is always the only point it has
	// if (id == DEBUG_POINT) { // TODO remove
	// 	printf("inserted %ldth point %d into node %p at depth %d\n",
	// 		   node->points.size(), id, node, depth);
	// 	std::cout << "debug point bins: " << bins << std::endl;
	// }

	auto maxDepth = bins.rows() - 1;
	if (node->points.size() > MAX_POINTS_PER_LEAF && depth < maxDepth) {
		splitLeafNode(node, depth, allBins);
	}
}

unique_ptr<Node> constructIndex(const MatrixXd& X, const MatrixXd& projectionVects,
								double& binWidth) {
	// auto V = computeProjectionVects(X, numProjections);
	MatrixXd positions = X * projectionVects.transpose();

	if (binWidth <= 0.) {
		binWidth = computeBinWidth(positions);
	}


	auto bins = binsForPositions(positions, binWidth);

	// this is right
	std::cout << "X rows, X cols: " << X.rows() << ", " << X.cols() << std::endl;
	std::cout << "V rows, V cols: " << projectionVects.rows() << ", " << projectionVects.cols() << std::endl;
	std::cout << "bins rows, bins cols: " << bins.rows() << ", " << bins.cols() << std::endl;

	std::cout << "sum of bins: " << bins.array().abs().sum() << std::endl;

	// everything is in bin 0 cuz r is huge...
	// std::cout << "positions: " << positions << std::endl;
	// std::cout << "bins: " << bins << std::endl;

	auto root = unique_ptr<Node>(new Node());
	for (length_t i = 0; i < bins.rows(); i++) {
		insert(root.get(), bins.row(i), i, bins);
	}

	return root;
}

// ================================================================
#pragma mark Queries
// ================================================================

// ------------------------------------------------ Main funcs

vector<length_t> findNeighborsForBins(Node& node, const hash_t bins[],
// vector<length_t> findNeighborsForBins(Node* node, const hash_t bins[],
									  double binWidth, double maxDistSq) {

	// printf("node %p, current bin %d\n", &node, bins[0]);

	// ------------------------ leaf node
	if (!node.is_internal) {
		if (ar::contains(node.points, DEBUG_POINT)) {
			printf("leaf node %p contains point %d\n", &node, DEBUG_POINT);
		}
		// printf("found leaf node %p: number of points: %ld\n", &node, node.points.size());
		return node.points; // don't test for actually being in range here
	}

	// ------------------------ internal node
	vector<length_t> neighbors;
	hash_t key = bins[0];
	// hash_t maxBinGap = static_cast<hash_t>(floor(sqrt(maxDistSq) / binWidth));
	// hash_t maxBinOffset = maxBinGap + 1;

	// std::cout << "node number of children: " << node.children.size() << std::endl;

	auto it = node.children.lower_bound(key); // first el with key >= query key
	// if (it == node.children.end()) { // key > all keys already in map
	// 	std::cout << "decrementing it since key was largest: " << key << std::endl;
	// 	--it; // point to last element; has at least one element by construction
	// }

	map_t::reverse_iterator itr(it);
	auto end = node.children.end();
	auto rend = node.children.rend();
	while (it != end || itr != rend) {
		if (it != end) {
			auto binKey = it->first;
			int32_t binDiff = binKey - key; // hash_t could overflow if signed
			binDiff = std::max(0, binDiff - 1);
			auto binDist = binWidth * binDiff;
			binDist *= binDist;
			double distBound = maxDistSq - binDist;

			if (distBound < 0) {
				it = end;
			} else {
				auto child = it->second;
				auto childNeighbors = findNeighborsForBins(child, bins + 1,
														   binWidth, distBound);
				std::move(std::begin(childNeighbors), std::end(childNeighbors),
						  std::back_inserter(neighbors));
				++it;
			}
		}
		if (itr != rend) {
			auto binKey = it->first;
			int32_t binDiff = key - binKey;
			binDiff = std::max(0, binDiff - 1);
			auto binDist = binWidth * binDiff;
			binDist *= binDist;
			double distBound = maxDistSq - binDist;

			if (distBound < 0) {
				itr = rend;
			} else {
				auto child = itr->second;
				auto childNeighbors = findNeighborsForBins(child, bins + 1,
														   binWidth, distBound);
				std::move(std::begin(childNeighbors), std::end(childNeighbors),
						  std::back_inserter(neighbors));
				++itr;
			}
		}
		// std::cout << "ran main loop" << std::endl;
	}

	// // recurse in query's bin
	// if (node.children.count(key)) { // no child node for this offset
	// 	auto child = node.children[key];
	// 	auto childNeighbors = findNeighborsForBins(child, bins + 1,
	// 											   binWidth, maxDistSq);
	// 	std::move(std::begin(childNeighbors), std::end(childNeighbors),
	// 			  std::back_inserter(neighbors));
	// }

	// // recurse in adjacent bins and append results to overall list
	// for (hash_t offsetMag = 1; offsetMag <= maxBinOffset; offsetMag++) {
	// 	hash_t binGap = offsetMag - 1;
	// 	double binDist = binWidth * binGap;
	// 	// double distBound = sqrt(radiusL2*radiusL2 - binDist*binDist);
	// 	binDist *= binDist; //TODO use exact query position within bin
	// 	double distBound = maxDistSq - binDist;

	// 	for (int8_t sign = -1; sign <= 1; sign += 2) { // +/- each offset
	// 		hash_t offset = sign * offsetMag;
	// 		hash_t kk = key + offset;
	// 		if (! node.children.count(kk)) { // no child node for this offset
	// 			continue;
	// 		}
	// 		// recurse and append results to overall list
	// 		auto child = node.children[kk];
	// 		auto childNeighbors = findNeighborsForBins(child, bins + 1,
	// 												   binWidth, distBound);
	// 		std::move(std::begin(childNeighbors), std::end(childNeighbors),
	// 				  std::back_inserter(neighbors));
	// 	}
	// }
	return neighbors;
}

Neighbor find1nnForBins(const VectorXd& q, const MatrixXd& X, Node& node,
	const hash_t bins[], double binWidth, double d_lb, double d_bsf) {
	length_t nn = -1;

	// ------------------------ leaf node
	if (! node.is_internal) { // leaf node
		for (length_t point : node.points) {
			double dist = squaredL2Dist(X.row(point), q);
			if (dist < d_bsf) {
				d_bsf = dist;
				nn = point;
			}
		}
		return Neighbor{.idx = nn, .dist = d_bsf};
	}

	// ------------------------ internal node
	double d_cushion = d_bsf - d_lb;
	hash_t key = bins[0];
	hash_t maxBinGap = hash_t(floor(sqrt(d_cushion) / binWidth));
	hash_t maxBinOffset = maxBinGap + 1;

	for (hash_t offsetMag = 0; offsetMag <= maxBinOffset; offsetMag++) {
		hash_t binGap = std::max(offsetMag - 1, 0);
		double binDist = binWidth * binGap;
		binDist *= binDist;
		double distLowerBound = d_lb + binDist;

		if (distLowerBound > d_bsf) {
			continue;
		}

		for (int8_t sign = -1; sign <= 1; sign += 2) { // +/- each offset
			hash_t offset = sign * offsetMag;
			hash_t kk = key + offset;
			if (! node.children.count(kk)) { // no child node for this offset
				continue;
			}

			auto neighbor = find1nnForBins(q, X, node, bins + 1, binWidth,
											 distLowerBound, d_bsf);
			if (neighbor.dist < d_bsf) {
				d_bsf = neighbor.dist;
				nn = neighbor.idx;
			}
		}
	}
	return Neighbor{.idx = nn, .dist = d_bsf};
}


void findKnnForBins(const VectorXd& q, const MatrixXd& X, uint16_t k,
	Node& node, const hash_t bins[], double binWidth, double d_lb,
	Neighbor neighborsBsf[]) {

	uint16_t lastIdx = k - 1;
	double d_bsf = neighborsBsf[lastIdx].dist;

	// ------------------------ leaf node
	if (! node.is_internal) {
		for (length_t point : node.points) {
			double dist = squaredL2Dist(X.row(point), q);
			if (dist < d_bsf) {
				neighborsBsf[lastIdx].dist = dist;
				neighborsBsf[lastIdx].idx = point;

				// bubble the new neighbor down to the right position
				uint16_t i = lastIdx;
				while (i > 0 && neighborsBsf[i-1].dist > dist) {
					// swap new and previous neighbor
					Neighbor tmp = neighborsBsf[i-1];
					neighborsBsf[i-1] = neighborsBsf[i];
					neighborsBsf[i] = tmp;
					i--;
				}
			}
		}
		return;
	}

	// ------------------------ internal node
	double d_cushion = d_bsf - d_lb;
	hash_t key = bins[0];
	hash_t maxBinGap = hash_t(floor(sqrt(d_cushion) / binWidth));
	hash_t maxBinOffset = maxBinGap + 1;

	for (hash_t offsetMag = 0; offsetMag <= maxBinOffset; offsetMag++) {
		hash_t binGap = std::max(offsetMag - 1, 0);
		double binDist = binWidth * binGap;
		binDist *= binDist;
		double distLowerBound = d_lb + binDist;

		if (distLowerBound > d_bsf) {
			continue;
		}

		for (int8_t sign = -1; sign <= 1; sign += 2) { // +/- each offset
			hash_t offset = sign * offsetMag;
			hash_t kk = key + offset;
			if (! node.children.count(kk)) { // no child node for this offset
				continue;
			}

			auto child = node.children[kk];
			findKnnForBins(q, X, k, child, bins + 1, binWidth, distLowerBound,
						   neighborsBsf);
		}
	}
}

// ------------------------------------------------ Top-level funcs

vector<hash_t> binsForQuery(VectorXd q, MatrixXd projectionVects,
							double binWidth) {
	VectorXd positions = projectionVects * q;
	// std::cout << "query: " << q << std::endl;
	// std::cout << "query positions: " << positions << std::endl;
	return binsForVectPositions(positions, binWidth);
}

vector<length_t> findNeighbors(const VectorXd& q, const MatrixXd& X,
	Node& root, const MatrixXd& projectionVects, double radiusL2,
	double binWidth) {

	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth);
	double maxDistSq = radiusL2 * radiusL2;

	// array_print_with_name(bins, "query bins");
	// printf("initial maxDistSq: %g\n", maxDistSq);
	// printf("initial binWidth: %g\n", binWidth);

	// auto neighbors = findNeighborsForBins(root, &bins[0], binWidth, radiusL2);
	auto neighbors = findNeighborsForBins(root, &bins[0], binWidth, maxDistSq);
	// auto neighbors = findNeighborsForBins(&root, &bins[0], binWidth, maxDistSq);

	std::cout << "num neighbors before filtering: " << neighbors.size() << std::endl;

	// TODO remove
	// array_print_with_name(array_unique(&bins[0]), bins.size(), "uniqBins");
	// array_sort(neighbors);
	// array_print_with_name(neighbors, "neighborsInBuckets");

	// double maxDistSq = radiusL2*radiusL2;
	return ar::filter([&X, &q, maxDistSq](length_t i) { // prune false positives
		// return squaredL2Dist(X.row(i), q) <= maxDistSq;
		return squaredL2Dist(X.row(i), q) <= maxDistSq;
	}, neighbors);
	// return neighbors; // TODO remove
}

Neighbor find1nn(const VectorXd& q, const MatrixXd& X, Node& root,
				 const MatrixXd& projectionVects, double binWidth) {
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth);

	// pick a random point as initial nearest neighbor
	int idx = rand() % X.rows();
	double d_bsf = squaredL2Dist(X.row(idx), q) + .0001; // add const in case this is best
	double d_lb = 0;

	return find1nnForBins(q, X, root, &bins[0], binWidth, d_lb, d_bsf);
}

vector<Neighbor> findKnn(const VectorXd& q, const MatrixXd& X, uint16_t k,
						Node& root, const MatrixXd& projectionVects,
						double binWidth, uint8_t numGuessesMultipleOfK) {

	assert(numGuessesMultipleOfK >= 2); // need k points disjoint from the k best

	// sample a number of points at random
	int numNeighborGuesses = numGuessesMultipleOfK * k;
	auto idxs = ar::rand_ints(0, X.rows(), numNeighborGuesses);
	vector<Neighbor> sampleNeighbors;
	for (int i = 0; i < numNeighborGuesses; i++) {
		double dist = squaredL2Dist(X.row(i), q);
		length_t idx = static_cast<length_t>(idxs[i]);
		sampleNeighbors.push_back(Neighbor{.dist=dist, .idx=idx});
	}

	// sort sampled points by increasing distance from the query
	std::sort(std::begin(sampleNeighbors), std::end(sampleNeighbors),
			  [](const Neighbor& n1, const Neighbor& n2) {
				  return n1.dist < n2.dist;
			  });

	// create vector containing kth to (2k-1)th best neighbors from sample
	vector<Neighbor> trueNeighbors(k);
	auto begin = std::begin(sampleNeighbors);
	std::copy(begin + k, begin + 2*k, std::back_inserter(trueNeighbors));

	// find the true k nearest neighbors using this initial guess
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth);
	double d_lb = 0;
	findKnnForBins(q, X, k, root, &bins[0], binWidth, d_lb, &trueNeighbors[0]);
	return trueNeighbors;
}








