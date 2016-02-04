//
//  tree.cpp
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

// ================================================================
// Typedefs and usings
// ================================================================

typedef Eigen::Matrix<hash_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HashMat;
typedef Eigen::Matrix<hash_t, Eigen::Dynamic, 1> HashVect;

using std::unordered_map;
using RedSVD::RedSVD;

// ================================================================
// Constants
// ================================================================

const hash_t MAX_HASH_VALUE = 32767; // because int16_t
const hash_t MIN_HASH_VALUE = -32768;

const uint16_t MAX_POINTS_PER_LEAF = 8;

// ================================================================
// Structs
// ================================================================

Node::Node() :
	children(unordered_map<hash_t, Node>()),
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

MatrixXd computeProjectionVects(const MatrixXd& X, depth_t numVects) {
	RedSVD<MatrixXd> svd(X, numVects);
	return svd.matrixV().transpose(); // cols should be eigenvectors -> rows eigenvectors
	// TODO don't even compute matrixU--comment out the flag that tells it to
}

HashMat binsForPositions(const MatrixXd& positions, double binWidth) {
//	HashMat bins = positions.cast<hash_t>();
	HashMat bins(positions.rows(), positions.cols());
//	for (size_t i = 0; i < positions.size(); i++) {
	for (size_t j = 0; j < bins.cols(); j++) { // column major order
		for (size_t i = 0; i < bins.rows(); i++) {
			int bin = (int) (positions(i, j) / binWidth);
			bins(i, j) = clamp(bin, MIN_HASH_VALUE, MAX_HASH_VALUE);
		}
	}
	return bins;
}

vector<hash_t> binsForVectPositions(const VectorXd& positions, double binWidth) {
	size_t ndims = positions.size();
	vector<hash_t>bins(ndims);
	for (size_t i; i < ndims; i++) {
		int64_t bin = (int64_t) (positions(i) / binWidth);
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
//	RowVectorXd colSquaredNormsAsRow = colSquaredNorms.transposeInPlace();
	RowVectorXd colSquaredNormsAsRow = colSquaredNorms.transpose();
	dists.colwise() += rowSquaredNorms;
	dists.rowwise() += colSquaredNormsAsRow;

	return dists.array();
}

ArrayXd distsToVectors(const MatrixXd& X, const MatrixXd& V) {
	return squaredDistsToVectors(X, V).sqrt();
}

pair<Node&, depth_t> leafNodeForPointBins(Node& table, const HashVect& bins) {
	depth_t dim = 0;
	Node& node = table;
	while (node.is_internal) {
		auto key = bins(dim);
//		node.children.emplace(key); // create key, Node() entry // TODO uncomment
		node.children.emplace(key, Node{}); // creates Node() iff key not present // TODO uncommment
		node = node.children[key]; // get the created node
		dim++;
	}

	return pair<Node&, depth_t>(node, dim);
}

void splitLeafNode(Node& node, depth_t nodeDepth, const HashMat& allBins) {
	node.is_internal = true;
	auto points = node.points;

	auto depth = nodeDepth;
	for (length_t point : points) {
		HashVect bins = allBins.row(point); // point is an index
		auto key = bins[depth];

//		node.children.emplace(key); // creates Node() iff key not present // TODO uncommment
		node.children.emplace(key, Node{}); // creates Node() iff key not present // TODO uncommment
		Node& child = node.children[key];
		child.points.push_back(point);
	}
}

void insert(Node& table, const HashVect& bins, length_t id,
	const HashMat& allBins) {

	auto nodeAndDepth = leafNodeForPointBins(table, bins);
	auto node = nodeAndDepth.first;
	auto depth = nodeAndDepth.second;

	node.points.push_back(id); // append point to list

	auto maxDepth = bins.rows() - 1;
	if (node.points.size() > MAX_POINTS_PER_LEAF && depth < maxDepth) {
		splitLeafNode(node, depth, allBins);
	}
}

unique_ptr<Node> constructIndex(const MatrixXd& X, const MatrixXd& projectionVects,
								double binWidth) {
//	auto V = computeProjectionVects(X, numProjections);
	MatrixXd positions = X * projectionVects.transpose();
	auto bins = binsForPositions(positions, binWidth);

	auto root = unique_ptr<Node>(new Node());
	for (length_t i = 0; i < bins.rows(); i++) {
		insert(*root, bins.row(i), i, bins);
	}

	return root;
}

// ================================================================
#pragma mark Queries
// ================================================================

// ------------------------------------------------ Main funcs

vector<length_t> findNeighborsForBins(Node& node, const hash_t bins[],
									  double binWidth, double maxDistSq) {

	// ------------------------ leaf node
	if (!node.is_internal) { // leaf node
		return node.points; // don't test for actually being in range here
	}

	// ------------------------ internal node
	vector<length_t> neighbors;
	hash_t key = bins[0];
	hash_t maxBinGap = hash_t(floor(sqrt(maxDistSq / binWidth)));
	hash_t maxBinOffset = maxBinGap + 1;

	for (hash_t offsetMag = 0; offsetMag < maxBinOffset; offsetMag++) {
		hash_t binGap = std::max(offsetMag - 1, 0);
		double binDist = binWidth * binGap;
		binDist *= binDist;
		double distBound = maxDistSq - binDist;

		for (int8_t sign = -1; sign <= 1; sign += 2) { // +/- each offset
			hash_t offset = sign * offsetMag;
			hash_t kk = key + offset;
			if (! node.children.count(kk)) { // no child node for this offset
				continue;
			}

			// recurse and append results to overall list
			auto child = node.children[kk];
			auto childNeighbors = findNeighborsForBins(child, bins + 1,
													   binWidth, distBound);
			std::move(childNeighbors.begin(), childNeighbors.end(),
					  std::back_inserter(neighbors));
		}
	}

	return neighbors;
}

Neighbor find1nnForBins(const VectorXd& q, const MatrixXd& X, Node& node,
									  const hash_t bins[], double binWidth,
									  double d_lb, double d_bsf) {
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
	hash_t maxBinGap = hash_t(floor(sqrt(d_cushion / binWidth)));
	hash_t maxBinOffset = maxBinGap + 1;

	for (hash_t offsetMag = 0; offsetMag < maxBinOffset; offsetMag++) {
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
	hash_t maxBinGap = hash_t(floor(sqrt(d_cushion / binWidth)));
	hash_t maxBinOffset = maxBinGap + 1;

	for (hash_t offsetMag = 0; offsetMag < maxBinOffset; offsetMag++) {
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

// ------------------------------------------------ Top-level funcs

vector<hash_t> binsForQuery(VectorXd q, MatrixXd projectionVects,
							double binWidth) {
	VectorXd positions = projectionVects * q;
	return binsForVectPositions(positions, binWidth);
}

vector<length_t> findNeighbors(const VectorXd& q, const MatrixXd& X,
	Node& root, const MatrixXd& projectionVects, double radiusL2,
	double binWidth) {
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth);
	double maxDistSq = radiusL2 * radiusL2;

	auto neighbors = findNeighborsForBins(root, &bins[0], binWidth, maxDistSq);
	return filter([&X, &q, maxDistSq](length_t i) { // prune false positives
		return squaredL2Dist(X.row(i), q) <= maxDistSq;
	}, neighbors);
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
	auto idxs = rand_ints(0, X.rows(), numNeighborGuesses);
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








