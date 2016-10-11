//  neighbors.cpp
//
//  Dig
//
//  Created by DB on 10/10/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include "neighbors.hpp"

#include "debug_utils.hpp"  // TODO remove

#include "eigen_utils.hpp"
#include "nn_index.hpp"
#include "timing_utils.hpp"

using std::vector;
using nn::idx_t;

// ================================================================
// Funcs
// ================================================================

MatrixXi idx_mat_from_nested_neighbor_idxs(
    const vector<vector<idx_t> >& nested_neighbors)
{
	using Index = typename MatrixXi::Scalar;
    size_t max_num_neighbors = ar::max(ar::map([](auto& neighbors) {
        // PRINT_VAR(neighbors.size());
        return neighbors.size();
    }, nested_neighbors));
    size_t num_rows = nested_neighbors.size();

    max_num_neighbors = ar::max(max_num_neighbors, 1); // no neighbors -> 1
    MatrixXi ret(num_rows, max_num_neighbors);
    for(int i = 0; i < num_rows; i++) {
        auto& neighbors = nested_neighbors[i];
        int j = 0;
        for (; j < neighbors.size(); j++) {
            ret(i, j) = static_cast<Index>(neighbors[j]);
        }
        for (; j < max_num_neighbors; j++) { // write -1 past end in each row
            ret(i, j) = -1;
        }
    }
    return ret;
}

// ================================================================
// Classes
// ================================================================

// ------------------------------------------------ IndexImpl

template<class IndexT>
class IndexImpl {
public:
    using Scalar = typename IndexT::Scalar;
    using MatrixT = typename scalar_traits<Scalar>::ColMatrixT;
    using VectorT = typename scalar_traits<Scalar>::ColVectorT;
    using RowMatrixT = typename scalar_traits<Scalar>::RowMatrixT;

    // ------------------------ ctors
    IndexImpl(const MatrixT& X):
        _indexStartTimeMs(timeNow()),
        _index(X),
        _indexTimeMs(durationMs(_indexStartTimeMs, timeNow()))
    {}
    IndexImpl(Scalar* X, int m, int n):
        IndexImpl(eigenWrap2D_aligned(X, m, n)) {}

    // ------------------------ single queries
    vector<int64_t> radius(const VectorT& q, double radiusL2) {
        EasyTimer t(&_queryTimeMs);
        return _index.radius_idxs(q.transpose(), radiusL2);
    }
    vector<int64_t> knn(const VectorT& q, int k) {
        EasyTimer t(&_queryTimeMs);
        return _index.knn_idxs(q.transpose(), k);
    }

    // ------------------------ batch queries
    MatrixXi radius_batch(const RowMatrixT& queries, double radiusL2) {
        EasyTimer t(&_queryTimeMs);
        auto nested_neighbors = _index.radius_batch_idxs(queries, radiusL2);
        return idx_mat_from_nested_neighbor_idxs(nested_neighbors);
    }
    MatrixXi knn_batch(const RowMatrixT& queries, int k) {
        EasyTimer t(&_queryTimeMs);
        auto nested_neighbors = _index.knn_batch_idxs(queries, k);
        return idx_mat_from_nested_neighbor_idxs(nested_neighbors);
    }

protected:
    cputime_t _indexStartTimeMs;
    IndexT _index;
    double _indexTimeMs;
    double _queryTimeMs;
};

// ------------------------------------------------ MatmulIndex::Impl

class MatmulIndex::Impl: public IndexImpl<nn::L2IndexBrute<double> > {
	using Super = IndexImpl<nn::L2IndexBrute<double> >;
    friend class MatmulIndex;
    using Super::Super; // inherit super ctors
	// Impl(const typename Super::MatrixT& X): Super{X} {}
	// Impl(typename Super::Scalar* X, int m, int n): Super(X, m, n) {}
};

// ------------------------------------------------ MatmulIndex

// ------------------------ ctors / dtors

// #define INDEX_CTORS_DTOR(INDEX_NAME) \
// INDEX_NAME ## ::INDEX_NAME(const MatrixXd& X): _this{new INDEX_NAME ## ::Impl{X}} {} \
// INDEX_NAME ## ::INDEX_NAME(double* X, int m, int n): \
//     _this{new INDEX_NAME ## ::Impl{X, m, n}} {} \
// INDEX_NAME ## ::~INDEX_NAME() = default; // needed for swig with unique_ptr

// INDEX_CTORS_DTOR(MatmulIndex)

MatmulIndex::MatmulIndex(const MatrixXd& X): _this{new MatmulIndex::Impl{X}} {}
MatmulIndex::MatmulIndex(double* X, int m, int n):
    _this{new MatmulIndex::Impl{X, m, n}}
{}
MatmulIndex::~MatmulIndex() = default; // needed for swig with unique_ptr


// ------------------------ single queries
vector<int64_t> MatmulIndex::radius(const VectorXd& q, double radiusL2) {
    return _this->radius(q, radiusL2);
}
vector<int64_t> MatmulIndex::knn(const VectorXd& q, int k) {
    return _this->knn(q, k);
}

// ------------------------ batch queries
MatrixXi MatmulIndex::radius_batch(const RowMatrixXd& queries, double radiusL2) {
    return _this->radius_batch(queries, radiusL2);
}
MatrixXi MatmulIndex::knn_batch(const RowMatrixXd& queries, int k) {
    return _this->knn_batch(queries, k);
}

// ------------------------ stats
double MatmulIndex::getIndexConstructionTimeMs() { return _this->_indexTimeMs; }
double MatmulIndex::getQueryTimeMs() { return _this->_queryTimeMs; }
