//
//  tree.cpp
//  Dig
//
//  Created by DB on 1/24/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include "tree.hpp"
#include <map> // TODO remove
#include <unordered_map>
#include <vector>
#include <cmath>

#include "Dense"
#include "redsvd.hpp"

using std::pair;
using std::make_pair;
using std::unique_ptr;

using std::vector;
using std::unordered_map;

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::ArrayXi;
//typedef Eigen::MatrixXd<double, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
using RedSVD::RedSVD;

// ================================================================
// Typedefs and usings
// ================================================================

typedef int16_t hash_t;
typedef int32_t length_t; // only handle up to length 2 billion
typedef int16_t depth_t;
typedef int64_t obj_id_t;

typedef Eigen::Matrix<hash_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HashMat;
typedef Eigen::Matrix<hash_t, Eigen::Dynamic, 1> HashVect;

// ================================================================
// Constants
// ================================================================

const hash_t MAX_HASH_VALUE = 32767; // because int16_t
const hash_t MIN_HASH_VALUE = -32768;

const uint16_t MAX_POINTS_PER_LEAF = 8;

// ================================================================
// Data structures
// ================================================================

// ------------------------------------------------ Node
typedef struct Node {
//	std::map<hash_t, struct Node> children;
	unordered_map<hash_t, struct Node> children;
//	std::array<length_t, MAX_POINTS_PER_LEAF> points;
	vector<length_t> points;
	bool is_internal; /// flag indicating that this is not a leaf node
public:
	Node();
//	Node(const Node& other);
} Node;

Node::Node() :
	children(unordered_map<hash_t, Node>()),
	points(vector<length_t>(MAX_POINTS_PER_LEAF)),
	is_internal(false)
{}

//Node::Node(const Node& other) :
//	children(other.children),
//	points(other.points),
//	is_internal(other.is_internal)
//{}



// ------------------------------------------------ Neighbor
typedef struct Neighbor {
	double dist;
	length_t idx;
} Neighbor;


// ================================================================
// Index construction
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

MatrixXd computeProjectionVects(MatrixXd X, depth_t numVects=16) {
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

ArrayXd distsToVectors(const MatrixXd& X, const MatrixXd& V) {
//	auto prods = -2. * (X * V.transpose());
//	auto dists = prods;
//	VectorXd rowSquaredNorms = X.rowwise().squaredNorm();
//	dists.colwise() += rowSquaredNorms;
//	dists.rowwise() += V.rowwise().squaredNorm();
//	
//	return dists.array().sqrt();
	return X.array(); // TODO remove
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

void insert(Node& table, const HashVect& bins, length_t id, const HashMat& allBins) {
	auto nodeAndDepth = leafNodeForPointBins(table, bins);
	auto node = nodeAndDepth.first;
	auto depth = nodeAndDepth.second;
	
	node.points.push_back(id); // append point to list
	
	auto maxDepth = bins.rows() - 1;
	if (node.points.size() > MAX_POINTS_PER_LEAF && depth < maxDepth) {
		splitLeafNode(node, depth, allBins);
	}
}

unique_ptr<Node> constructIndex(MatrixXd X, MatrixXd projectionVects,
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
// Queries
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
			if (not node.children.count(kk)) { // no child node for this offset
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

Neighbor find1nnForBins(VectorXd q, MatrixXd X, Node& node,
									  const hash_t bins[], double binWidth,
									  double d_lb, double d_bsf) {
	length_t nn = -1;
	
	// ------------------------ leaf node
	if (! node.is_internal) { // leaf node
		for (length_t point : node.points) {
			VectorXd diffs = X.row(point) - q;
			double dist = diffs.squaredNorm();
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


void findKnnForBins(VectorXd q, MatrixXd X, uint16_t k, Node& node,
					const hash_t bins[], double binWidth,
					double d_lb, Neighbor neighborsBsf[]) {
	uint16_t lastIdx = k - 1;
	double d_bsf = neighborsBsf[lastIdx].dist;
	
	// ------------------------ leaf node
	if (! node.is_internal) {
		for (length_t point : node.points) {
			VectorXd diffs = q - X.row(point);
			double dist = diffs.squaredNorm();
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

vector<length_t> findNeighbors(VectorXd q, Node& root, MatrixXd projectionVects,
				  double radiusL2, double binWidth) {
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth);
	double maxDistSq = radiusL2 * radiusL2;
	return findNeighborsForBins(root, &bins[0], binWidth, maxDistSq);
}

Neighbor find1nn(VectorXd q, MatrixXd X, Node& root, MatrixXd projectionVects,
				 double binWidth) {
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth);
	
	// pick a random point as initial nearest neighbor
	int idx = rand() % X.rows();
	auto diff = X.row(idx) - q;
	double d_bsf = diff.squaredNorm() + .0001; // add const in case this is best
	double d_lb = 0;
	
	return find1nnForBins(q, X, root, &bins[0], binWidth, d_lb, d_bsf);
}

//vector<Neighbor> findKnn(VectorXd q, MatrixXd X, uint16_t k, Node& root,
//						 MatrixXd projectionVects, double binWidth) {
//	
//}
//







