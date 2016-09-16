//
//  neighbors.hpp
//  Dig
//
//  Created by DB on 10/2/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef Dig_Neighbors_hpp
#define Dig_Neighbors_hpp

#include <memory>
#include <vector>

#include "Dense"

//#include "type_defs.h"

//typedef int32_t length_t;

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static constexpr double kMaxDist = std::numeric_limits<double>::max();


double swigEigenTest(double* X, int m, int n);

// ------------------------------------------------ Neighbor
typedef struct Neighbor {
	typedef double dist_t;
	typedef int64_t idx_t;
	
	double dist;
	int64_t idx;
} Neighbor;

// ================================================================
// Classes
// ================================================================

// ------------------------------------------------ BinTree

// TODO these shouldn't be in this header; probably use pimpl idiom
typedef int16_t depth_t;
//struct Node;
//typedef std::unique_ptr<Node> node_ptr_t;

//class BinTree_impl;

class BinTree {
private:
	class Impl;
	std::unique_ptr<Impl> _ths; // pimpl idiom
	// Impl* _ths; // pimpl idiom--no uniq ptr since SWIG can't handle it
	// EDIT: nvm, new SWIG seems to be good with it

public:
	~BinTree();

	// no copying TODO uncomment without breaking SWIG
	BinTree(const BinTree& other) = delete;
	BinTree& operator=(const BinTree&) = delete;
//	BinTree(const BinTree& other);
//	BinTree& operator=(const BinTree&);

	BinTree(const MatrixXd& X, depth_t numProjections=16);

	vector<int32_t> rangeQuery(const VectorXd& q, double radiusL2);
	vector<int32_t> knnQuery(const VectorXd& q, int k);

	// ------------------------ versions for numpy typemaps
	BinTree(double* X, int m, int n, int numProjections=16);

	double getIndexConstructionTimeMs();
	// double getIndexConstructionTimeMs() { return 8.8; }; // TODO remove
	double getQueryTimeMs();

	// vector<length_t> rangeQuery(const double* q, int len, double radiusL2);
	// vector<Neighbor> knnQuery(const double* q, int len, int k);

	// TODO, ya so default args seem to be making SWIG unhappy
//	void rangeQuery(const double* q, int len, double radiusL2,
//								int* outVec, int outLen=-1);
//	void knnQuery(const double* q, int len, int k,
//							  int* outVec, int outLen=-1);
	// void rangeQuery(const double* q, int len, double radiusL2,
	// 				int* outVec, int outLen);
	// void knnQuery(const double* q, int len, int k,
	// 			  int* outVec, int outLen);
};

#endif
