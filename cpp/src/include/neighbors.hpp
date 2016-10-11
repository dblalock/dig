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

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
	Eigen::RowMajor> RowMatrixXd;
// typedef Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic,
// 	Eigen::RowMajor> MatrixXi;

namespace nn {
	using idx_t = int64_t;
}

double swigEigenTest(double* X, int m, int n);

// ------------------------------------------------ Neighbor
typedef struct Neighbor {
	typedef double dist_t;
	typedef nn::idx_t idx_t;

	double dist;
	idx_t idx;
} Neighbor;

// ================================================================
// Classes
// ================================================================

// ------------------------------------------------ MatmulIndex

class MatmulIndex { // TODO also have MatmulIndexF for floats
private:
	class Impl;
	std::unique_ptr<Impl> _ths; // pimpl idiom
public:
	typedef MatmulIndex SelfT;

	// ------------------------ lifecycle
	MatmulIndex(const SelfT& other) = delete;
	SelfT& operator=(const SelfT&) = delete;
	~MatmulIndex();

	MatmulIndex(const MatrixXd& X);
	MatmulIndex(double* X, int m, int n);

	// ------------------------ querying
	vector<int64_t> radius(const VectorXd& q, double radiusL2);
	vector<int64_t> knn(const VectorXd& q, int k);
	MatrixXi radius_batch(const RowMatrixXd& queries, double radiusL2);
	MatrixXi knn_batch(const RowMatrixXd& queries, int k);

	// ------------------------ stats
	double getIndexConstructionTimeMs();
	double getQueryTimeMs();
};


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
