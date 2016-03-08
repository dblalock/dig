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
#include "eigen_utils.hpp"

//#define DEBUG_POINT 2997


// ================================================================
// Temp # TODO remove
// ================================================================

#include "eigen_utils.hpp"
double swigEigenTest(double* X, int m, int n) {
	auto A = eigenWrap2D_nocopy(X, m, n);
	VectorXd rowSums = A.rowwise().sum();
	return rowSums.squaredNorm();
}


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

// const hash_t MAX_HASH_VALUE = 127; // int8_t
// const hash_t MIN_HASH_VALUE = -128;
const hash_t MAX_HASH_VALUE = 63; // int8_t
const hash_t MIN_HASH_VALUE = 0;
const hash_t HASH_VALUE_OFFSET = 32;

// const hash_t MAX_HASH_VALUE = 32767; // int16_t
// const hash_t MIN_HASH_VALUE = -32768;
// const double TARGET_HASH_SPREAD_STDS = 3.0; // +/- 3 std deviations
const double TARGET_HASH_SPREAD_STDS = 2.0; // +/- this many std deviations
// const double TARGET_HASH_SPREAD_STDS = 1.0; // +/- this many std deviations

// const uint16_t MAX_POINTS_PER_LEAF = 8;
const uint16_t MAX_POINTS_PER_LEAF = 64;
// const uint16_t MAX_POINTS_PER_LEAF = 256;

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
// Classes
// ================================================================

// ------------------------------------------------ BinTree Impl

class BinTree::Impl {
private:
	void initAfterX(depth_t numProjections);

public:
	MatrixXd _X;
	MatrixXd _projectionVects;
	depth_t _numProjections;
	node_ptr_t _root;
	double _binWidth;

	//ctors
	Impl(const MatrixXd& X, depth_t numProjections):
		_X(X)
	{
		initAfterX(numProjections);
	}

	Impl(double* X, int m, int n, int numProjections):
		_X(eigenWrap2D_aligned(X, m, n))
	{
			initAfterX(numProjections);
	}
};

void BinTree::Impl::initAfterX(depth_t numProjections) {
	_projectionVects = computeProjectionVects(_X, numProjections);
	_root = constructIndex(_X, _projectionVects, _binWidth); // sets _binWidth
}

// ------------------------------------------------ BinTree

BinTree::~BinTree() = default; // correct if we could use a unique_ptr

// BinTree::~BinTree() {
// 	delete _ths;
// }

BinTree::BinTree(const MatrixXd& X, depth_t numProjections):
	_ths{new BinTree::Impl{X, numProjections}}
{}

vector<length_t> BinTree::rangeQuery(const VectorXd& q, double radiusL2) {
	return findNeighbors(q, _ths->_X, *(_ths->_root), _ths->_projectionVects,
						 radiusL2, _ths->_binWidth);
}
vector<Neighbor> BinTree::knnQuery(const VectorXd& q, int k) {
	return findKnn(q, _ths->_X, k, *(_ths->_root), _ths->_projectionVects, _ths->_binWidth);
}

// ------------------------ versions for numpy typemaps
BinTree::BinTree(double* X, int m, int n, int numProjections):
	_ths{new BinTree::Impl{X, m, n, numProjections}}
{}

vector<length_t> BinTree::rangeQuery(const double* q, int len, double radiusL2) {
	VectorXd qVect = eigenWrap1D_aligned(q, len);
	return rangeQuery(qVect, radiusL2);
}
vector<Neighbor> BinTree::knnQuery(const double* q, int len, int k) {
	VectorXd qVect = eigenWrap1D_aligned(q, len);
	return knnQuery(qVect, k);
}
// void BinTree::rangeQuery(const double* q, int len, double radiusL2,
// 				int* outVec, int outLen) {
// 	VectorXd qVect = eigenWrap1D_aligned(q, len);
// 	auto neighbors = rangeQuery(qVect, radiusL2);
// 	for (int i = 0; i < neighbors.size(); i++) {
// 		outVec[i] = neighbors[i];
// 	}
// }
// void BinTree::knnQuery(const double* q, int len, int k,
// 			  int* outVec, int outLen) {
// 	VectorXd qVect = eigenWrap1D_aligned(q, len);
// 	auto neighbors = knnQuery(qVect, k);
// 	for (int i = 0; i < neighbors.size(); i++) {
// 		outVec[i] = neighbors[i].idx;
// 	}
// }

// ================================================================
#pragma mark Utils
// ================================================================

double squaredL2Dist(const VectorXd& x, const VectorXd& y) {
	VectorXd diff = x - y;
	return diff.squaredNorm();
}

template <typename T, typename U>
U clamp(const T x, const U lower, const U upper) {
	if (x > upper) {
		return upper;
	} else if (x < lower) {
		return lower;
	}
	return static_cast<U>(x);
}

// ================================================================
#pragma mark Index construction
// ================================================================

inline hash_t hashValForPosition(double position, double binWidth) {
	int64_t quantized = (int64_t) (position / binWidth);
	quantized += HASH_VALUE_OFFSET;
	return clamp(quantized, MIN_HASH_VALUE, MAX_HASH_VALUE);
}

// computes the distance of the position above the lower edge of its bin;
// eg, if the position 7.7 gets mapped to a bin from [5.5, 8.81), we
// return (7.7 - 5.5) = 2.2
inline double distAboveBinCutoff(double position, double binWidth) {
	int64_t quantized = (int64_t) (position / binWidth);
	double dist = position - (binWidth * quantized);

	// deal with case when hash value is clipping
	int64_t offseted = quantized + HASH_VALUE_OFFSET;
	if (offseted < MIN_HASH_VALUE) {
		return 0; // not "above" the boundary for this bin at all
	} else if (offseted > MAX_HASH_VALUE) {
		dist += binWidth * (offseted - MAX_HASH_VALUE);
	} else {
		assert(dist < binWidth); // sanity check
	}
	return dist;
}

double computeBinWidth(const MatrixXd& positions) {
	// assumes first col of positions corresponds to dominant eigenvect
	ArrayXd firstCol = positions.col(0).array();
	firstCol -= firstCol.mean();
	double SSE = firstCol.matrix().squaredNorm();
	double variance = SSE / firstCol.size();
	double std = sqrt(variance);
	double targetBinsPerStd = (MAX_HASH_VALUE - HASH_VALUE_OFFSET) / TARGET_HASH_SPREAD_STDS;
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
			bins(i, j) = hashValForPosition(positions(i, j), binWidth);
			// int64_t bin = (int64_t) (positions(i, j) / binWidth);
			// bins(i, j) = clamp(bin, MIN_HASH_VALUE, MAX_HASH_VALUE);
		}
	}
	return bins;
}

vector<hash_t> binsForVectPositions(const VectorXd& positions, double binWidth,
	double* binDists) {
	size_t ndims = positions.size();
	vector<hash_t> bins;
	// std::cout << "binsForVectPositions, positions: " << positions << std::endl;
	// std::cout << "binsForVectPositions, ndims: " << ndims << std::endl;
	// std::cout << "binsForVectPositions, binWidth: " << binWidth << std::endl;
	for (size_t i = 0; i < ndims; i++) {
		bins.push_back(hashValForPosition(positions(i), binWidth));
		if (binDists) {
			binDists[i] = distAboveBinCutoff(positions(i), binWidth);
		}
		// int64_t bin = (int64_t) (positions(i) / binWidth);
		// printf("projection #%lu: bin = %lld\n", i, bin);
		// bins.push_back(clamp(bin, MIN_HASH_VALUE, MAX_HASH_VALUE));
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

inline void insertEmptyNodePtr(map_t& map, hash_t key) {
	map.put(key, unique_ptr<Node>(new Node{}));
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
//		node->children.emplace(key, Node{}); // creates Node() iff key not present // TODO uncommment
//		node = &(node->children[key]); // get ptr to the created node
		if (!node->children.contains(key)) {
			insertEmptyNodePtr(node->children, key);
		}
		node = node->children.get(key).get();
		dim++;
	}

	return pair<Node*, depth_t>(node, dim);
}

void splitLeafNode(Node* node, depth_t nodeDepth, const HashMat& allBins) {
	//TODO go back to pass by ref after debug

	node->is_internal = true;
	auto points = node->points;

	// reclaim memory from storing these points, since
	// children will now store them
	node->points.clear();
	node->points.shrink_to_fit();

	// std::cout << "splitting node " << node << " with " << points.size() << " points" << std::endl;

	// if (ar::contains(points, DEBUG_POINT)) {
	// 	printf("splitting node %p containing point %d\n", node, DEBUG_POINT);
	// }

	for (length_t point : points) {
		HashVect bins = allBins.row(point); // point is an index
		hash_t key = bins[nodeDepth];

//		node.children.emplace(key); // creates Node() iff key not present // TODO uncommment
//		if (! node->children.count(key)) { // TODO remove
//			node->children.emplace(key, Node{}); // creates Node() iff key not present // TODO uncommment
//		}
		if (!node->children.contains(key)) {
			insertEmptyNodePtr(node->children, key);
		}

		Node* child = node->children.get(key).get();
		child->points.push_back(point);

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

	auto root = unique_ptr<Node>(new Node{});
	for (length_t i = 0; i < bins.rows(); i++) {
		insert(root.get(), bins.row(i), i, bins);
	}

	return root;
}

// ================================================================
#pragma mark Queries
// ================================================================

template<typename P>
inline int32_t prettyPtr(P ptr) {
	return (((int64_t)ptr) << 40) >> 40;
}

// ------------------------------------------------ Main funcs

inline double binDistanceSq(hash_t largerKey, hash_t smallerKey,
						  double binWidth, double distInBin) {
	int32_t binDiff = largerKey - smallerKey; // hash_t could overflow if signed
	int32_t binGap = std::max(0, binDiff - 1);
	auto binDist = binWidth * binGap;
	// binDist += binGap > 0 ? distInBin : 0; // TODO uncomment
	binDist += distInBin; // assumes we passed in 0 if keys were equal to avoid branch
	return binDist * binDist;
}

inline double distanceBoundSq(hash_t largerKey, hash_t smallerKey,
							double binWidth, double maxDistSq, double distInBin) {
	return maxDistSq - binDistanceSq(largerKey, smallerKey, binWidth, distInBin);
}

vector<length_t> findNeighborsForBins(Node* node, const hash_t bins[],
	const double binDists[], double binWidth, double maxDistSq) {

	 // printf("node %p, current bin %d\n", &node, bins[0]);

	// ------------------------ leaf node
	if (!node->is_internal) {
		// if (ar::contains(node->points, DEBUG_POINT)) {
		// 	printf("leaf node %p contains point %d\n", &node, DEBUG_POINT);
		// }
		// printf("found leaf node %p: number of points: %ld\n", node, node->points.size());
		// printf("\tleaf %d contains %lu points: %s\n", prettyPtr(node),
		// 	node->points.size(), ar::to_string(node->points).c_str());
		return node->points; // don't test for actually being in range here
	}

	// ------------------------ internal node
	vector<length_t> neighbors;
	hash_t key = bins[0];
	double distInBinRev = binDists[0];
	double distInBinFwd = binWidth - distInBinRev;

	map_t& map = node->children;
	auto key_fwd = map.firstKeyAtOrAfter(key);
	auto key_rev = map.lastKeyBefore(key);

	// printf("node %p has %d children\n", node, map.size());
	// map._dumpOccupiedArray();
	// std::cout << "\n";

	while ((key_fwd >= 0) || (key_rev >= 0)) {
		// printf("bin %d) key_fwd, key_rev = %d, %d\n", key, key_fwd, key_rev);
		if (key_fwd >= 0) {
			double distInBin = key_fwd > key ? distInBinFwd : 0;
			double distBound = distanceBoundSq(key_fwd, key, binWidth,
											   maxDistSq, distInBin);

			if (distBound < 0) {
				key_fwd = -1;
			} else {
				Node* child = map.get(key_fwd).get();
				 // printf("fwd child: %p\n", child);
				auto childNeighbors = findNeighborsForBins(child, bins + 1,
					binDists + 1, binWidth, distBound);
				std::move(std::begin(childNeighbors), std::end(childNeighbors),
						  std::back_inserter(neighbors));
				key_fwd = map.firstKeyAfter(key_fwd);
			}
		}
		if (key_rev >= 0) {
			double distBound = distanceBoundSq(key, key_rev, binWidth,
											   maxDistSq, distInBinRev);

			if (distBound < 0) {
				key_rev = -1;
			} else {
				Node* child = map.get(key_rev).get();
				// printf("rev child: %p\n", child);
				auto childNeighbors = findNeighborsForBins(child, bins + 1,
					binDists + 1, binWidth, distBound);
				std::move(std::begin(childNeighbors), std::end(childNeighbors),
						  std::back_inserter(neighbors));
				key_rev = map.lastKeyBefore(key_rev);
			}
		}
	}
	return neighbors;
}

Neighbor find1nnForBins(const VectorXd& q, const MatrixXd& X, Node* node,
	const hash_t bins[], const double binDists[], double binWidth,
	double d_lb, double d_bsf) {

	length_t nn = -1;

	// printf("%d: bin = %d\n", prettyPtr(node), bins[0]);

	// ------------------------ leaf node
	if (! node->is_internal) { // leaf node
		// printf("found leaf node %p\n", node);
		for (length_t point : node->points) {
			double dist = squaredL2Dist(X.row(point), q);
			if (dist < d_bsf) {
				// std::cout << "found point " << point << " with lower dist " << dist << "\n";
				d_bsf = dist;
				nn = point;
			}
		}
		return Neighbor{.idx = nn, .dist = d_bsf};
	}

	// ------------------------ internal node
	hash_t key = bins[0];
	double distInBinRev = binDists[0];
	double distInBinFwd = binWidth - distInBinRev;

	map_t& map = node->children;
	auto key_fwd = map.firstKeyAtOrAfter(key);
	auto key_rev = map.lastKeyBefore(key);

	// printf("%d: initial key_fwd, key_rev = %d, %d\n", prettyPtr(node), key_fwd, key_rev);

	while ((key_fwd >= 0) || (key_rev >= 0)) {
		if (key_fwd >= 0) {
			double distInBin = key_fwd > key ? distInBinFwd : 0;
			double distLowerBound = binDistanceSq(key_fwd, key, binWidth,
												  distInBin) + d_lb;
			if (distLowerBound < d_bsf) {
				Node* child = map.get(key_fwd).get();
				auto neighbor = find1nnForBins(q, X, child, bins + 1,
					binDists + 1, binWidth, distLowerBound, d_bsf);
				if (neighbor.dist < d_bsf) {
					d_bsf = neighbor.dist;
					nn = neighbor.idx;
				}
				key_fwd = map.firstKeyAfter(key_fwd);
			} else {
				key_fwd = -1;
			}
		}
		if (key_rev >= 0) {
			double distLowerBound = binDistanceSq(key, key_rev, binWidth,
												  distInBinRev) + d_lb;
			if (distLowerBound < d_bsf) {
				Node* child = map.get(key_rev).get();
				auto neighbor = find1nnForBins(q, X, child, bins + 1,
					binDists + 1, binWidth, distLowerBound, d_bsf);
				if (neighbor.dist < d_bsf) {
					d_bsf = neighbor.dist;
					nn = neighbor.idx;
				}
				key_rev = map.lastKeyBefore(key_rev);
			} else {
				key_rev = -1;
			}
		}
	}
	return Neighbor{.idx = nn, .dist = d_bsf};

// 	hash_t maxBinGap = hash_t(floor(sqrt(d_cushion) / binWidth));
// 	hash_t maxBinOffset = maxBinGap + 1;

// 	for (hash_t offsetMag = 0; offsetMag <= maxBinOffset; offsetMag++) {
// 		hash_t binGap = std::max(offsetMag - 1, 0);
// 		double binDist = binWidth * binGap;
// 		binDist *= binDist;
// 		double distLowerBound = d_lb + binDist;

// 		if (distLowerBound > d_bsf) {
// 			continue;
// 		}

// 		for (int8_t sign = -1; sign <= 1; sign += 2) { // +/- each offset
// 			hash_t offset = sign * offsetMag;
// 			hash_t kk = key + offset;
// //			if (! node.children.count(kk)) { // no child node for this offset
// 			if (! node.children.contains(kk)) { // no child node for this offset
// 				continue;
// 			}

// 			auto neighbor = find1nnForBins(q, X, node, bins + 1, binWidth,
// 											 distLowerBound, d_bsf);
// 			if (neighbor.dist < d_bsf) {
// 				d_bsf = neighbor.dist;
// 				nn = neighbor.idx;
// 			}
// 		}
// 	}
// 	return Neighbor{.idx = nn, .dist = d_bsf};
}


void findKnnForBins(const VectorXd& q, const MatrixXd& X, uint16_t k,
	Node* node, const hash_t bins[], const double binDists[],
	double binWidth, double d_lb, Neighbor neighborsBsf[]) {

	uint16_t lastIdx = k - 1;
	double d_bsf = neighborsBsf[lastIdx].dist;

	// ------------------------ leaf node
	if (! node->is_internal) {
		for (length_t point : node->points) {
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
				d_bsf = neighborsBsf[lastIdx].dist;
			}
		}
		return;
	}

	// ------------------------ internal node

	hash_t key = bins[0];
	double distInBinRev = binDists[0];
	double distInBinFwd = binWidth - distInBinRev;

	map_t& map = node->children;
	auto key_fwd = map.firstKeyAtOrAfter(key);
	auto key_rev = map.lastKeyBefore(key);

	while ((key_fwd >= 0) || (key_rev >= 0)) {
		if (key_fwd >= 0) {
			double distInBin = key_fwd > key ? distInBinFwd : 0;
			double distLowerBound = binDistanceSq(key_fwd, key, binWidth, distInBin) + d_lb;
			if (distLowerBound < d_bsf) {
				Node* child = map.get(key_fwd).get();
				findKnnForBins(q, X, k, child, bins + 1, binDists + 1, binWidth,
					distLowerBound, neighborsBsf);
				key_fwd = map.firstKeyAfter(key_fwd);
			} else {
				key_fwd = -1;
			}
		}
		if (key_rev >= 0) {
			double distLowerBound = binDistanceSq(key, key_rev, binWidth, distInBinRev) + d_lb;
			if (distLowerBound < d_bsf) {
				Node* child = map.get(key_rev).get();
				findKnnForBins(q, X, k, child, bins + 1, binDists + 1, binWidth,
					distLowerBound, neighborsBsf);
				key_rev = map.lastKeyBefore(key_rev);
			} else {
				key_rev = -1;
			}
		}
	}

	// double d_cushion = d_bsf - d_lb;
	// hash_t key = bins[0];
	// hash_t maxBinGap = hash_t(floor(sqrt(d_cushion) / binWidth));
	// hash_t maxBinOffset = maxBinGap + 1;

	// for (hash_t offsetMag = 0; offsetMag <= maxBinOffset; offsetMag++) {
	// 	hash_t binGap = std::max(offsetMag - 1, 0);
	// 	double binDist = binWidth * binGap;
	// 	binDist *= binDist;
	// 	double distLowerBound = d_lb + binDist;

	// 	if (distLowerBound > d_bsf) {
	// 		continue;
	// 	}

	// 	for (int8_t sign = -1; sign <= 1; sign += 2) { // +/- each offset
	// 		hash_t offset = sign * offsetMag;
	// 		hash_t kk = key + offset;
	// 		if (! node->children.contains(kk)) { // no child node for this offset
	// 			continue;
	// 		}

	// 		auto child = node->children.get(kk).get();
	// 		findKnnForBins(q, X, k, child, bins + 1, binWidth, distLowerBound,
	// 					   neighborsBsf);
	// 	}
	// }
}

// ------------------------------------------------ Top-level funcs

vector<hash_t> binsForQuery(VectorXd q, MatrixXd projectionVects,
							double binWidth, double* binDists) {
	VectorXd positions = projectionVects * q;
	// std::cout << "query: " << q << std::endl;
	// std::cout << "query positions: " << positions << std::endl;
	return binsForVectPositions(positions, binWidth, binDists);
}

vector<length_t> findNeighbors(const VectorXd& q, const MatrixXd& X,
	Node& root, const MatrixXd& projectionVects, double radiusL2,
	double binWidth) {

	double binDists[q.size()];
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth, binDists);
	double maxDistSq = radiusL2 * radiusL2;

	// array_print_with_name(bins, "query bins");
	// printf("initial maxDistSq: %g\n", maxDistSq);
	// printf("initial binWidth: %g\n", binWidth);

	// auto neighbors = findNeighborsForBins(root, &bins[0], binWidth, radiusL2);
	auto neighbors = findNeighborsForBins(&root, &bins[0], binDists, binWidth, maxDistSq);
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

	double binDists[q.size()];
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth, binDists);

	// pick a random point as initial nearest neighbor
	int idx = rand() % X.rows();
	double d_bsf = squaredL2Dist(X.row(idx), q) + .0001; // add const in case this is best
	// printf("initial d_bsf: %g\n", d_bsf);
	// d_bsf = 999.; // TODO remove
	double d_lb = 0;

	return find1nnForBins(q, X, &root, &bins[0], binDists, binWidth, d_lb, d_bsf);
}

vector<Neighbor> findKnn(const VectorXd& q, const MatrixXd& X, uint16_t k,
						Node& root, const MatrixXd& projectionVects,
						double binWidth, uint8_t numGuessesMultipleOfK) {
	assert(k > 0);

	if (k == 1) { // knn query code assumes k > 1
		Neighbor n = find1nn(q, X, root, projectionVects, binWidth);
		vector<Neighbor> ret;
		ret.push_back(n);
		printf("just returning 1nn\n");
		return ret;
	}

	assert(numGuessesMultipleOfK >= 2); // need k points disjoint from the k best

	// sample a number of points at random
	int numNeighborGuesses = numGuessesMultipleOfK * k;
	numNeighborGuesses = fmin(numNeighborGuesses, X.rows()); // TODO just brute force it
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
	vector<Neighbor> trueNeighbors;
	auto begin = std::begin(sampleNeighbors);
	std::copy(begin + k, begin + 2*k, std::back_inserter(trueNeighbors));

	printf("knn initial neighbors (%ld): \n\t{", trueNeighbors.size());
	for (int i = 0; i < trueNeighbors.size(); i++) {
		printf("%d: %g, ", trueNeighbors[i].idx, trueNeighbors[i].dist);
	}
	printf("}\n");

	// find the true k nearest neighbors using this initial guess
	double binDists[q.size()];
	vector<hash_t> bins = binsForQuery(q, projectionVects, binWidth, nullptr);
	double d_lb = 0;
	findKnnForBins(q, X, k, &root, &bins[0], binDists, binWidth, d_lb,
		&trueNeighbors[0]);
	return trueNeighbors;
}








