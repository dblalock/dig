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

#include "eigen_utils.hpp"

using std::vector;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::MatrixXi;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
	Eigen::RowMajor> RowMatrixXd;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
Eigen::RowMajor> RowMatrixXf;
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

#define NO_COPYING_AND_DEFAULT_DTOR(NAME) \
	NAME(const NAME & other) = delete; \
	NAME & operator=(const NAME &) = delete; \
	~NAME();

// NAME(const MatrixT & X);
#define DECLARE_INDEX_CTORS_DTOR(NAME, Scalar) \
	NAME(Scalar* X, int m, int n); \
	NO_COPYING_AND_DEFAULT_DTOR(NAME) \

#define DECLARE_INDEX_QUERY_FUNCS(VectorT, RowMatrixT) \
	vector<int64_t> radius(const VectorT & q, double radiusL2); \
	vector<int64_t> knn(const VectorT & q, int k); \
	MatrixXi radius_batch(const RowMatrixT & queries, double radiusL2); \
	MatrixXi knn_batch(const RowMatrixT & queries, int k);

#define DECLARE_INDEX_STATS_FUNCS \
	double getIndexConstructionTimeMs(); \
	double getQueryTimeMs();

#define DECLARE_INDEX_PIMPL(NAME) \
private: \
	class Impl; \
	std::unique_ptr<Impl> _this; \

// NOTE: we can't just compute VectorT, etc, based on Scalar because it
// makes SWIG unhappy; compiles but thinks args have wrong types at runtime
#define DECLARE_INDEX_CLASS(NAME, Scalar, VectorT, RowMatrixT) \
class NAME { \
public: \
	DECLARE_INDEX_CTORS_DTOR(NAME, Scalar) \
	DECLARE_INDEX_QUERY_FUNCS(VectorT, RowMatrixT) \
	DECLARE_INDEX_STATS_FUNCS \
	DECLARE_INDEX_PIMPL(NAME) \
};

DECLARE_INDEX_CLASS(MatmulIndex, double, VectorXd, RowMatrixXd);
DECLARE_INDEX_CLASS(MatmulIndexF, float, VectorXf, RowMatrixXf);

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
	// Impl* _this; // pimpl idiom--no uniq ptr since SWIG can't handle it
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
