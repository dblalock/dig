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

// ------------------------------------------------ Index creation macros

// ------------------------ pimpl

#define INDEX_PIMPL(INDEX_NAME, InnerIndexT) \
class INDEX_NAME ::Impl: public IndexImpl<InnerIndexT > { \
    using Super = IndexImpl<InnerIndexT >; \
    friend class INDEX_NAME; \
    using Super::Super; \
};

// ------------------------ ctors / dtors

// INDEX_NAME ::INDEX_NAME(const MatrixT & X): _this{new INDEX_NAME ::Impl{X}} {}
#define INDEX_CTORS_DTOR(INDEX_NAME, Scalar) \
    INDEX_NAME ::INDEX_NAME(Scalar* X, int m, int n): \
        _this{new INDEX_NAME ::Impl{X, m, n}} {} \
    INDEX_NAME ::~INDEX_NAME() = default;
// ^ default dtor needed for swig with unique_ptr

// ------------------------ query funcs

#define INDEX_QUERY_FUNCS(INDEX_NAME, VectorT, RowMatrixT) \
vector<int64_t> INDEX_NAME ::radius(const VectorT & q, double radiusL2) { \
    return _this->radius(q, radiusL2); \
} \
vector<int64_t> INDEX_NAME ::knn(const VectorT & q, int k) { \
    return _this->knn(q, k); \
} \
MatrixXi INDEX_NAME ::radius_batch(const RowMatrixT & queries, double radiusL2) { \
    return _this->radius_batch(queries, radiusL2); \
} \
MatrixXi INDEX_NAME ::knn_batch(const RowMatrixT & queries, int k) { \
    return _this->knn_batch(queries, k); \
}

// ------------------------ stats

#define INDEX_STATS_FUNCS(INDEX_NAME) \
double INDEX_NAME ::getIndexConstructionTimeMs() { return _this->_indexTimeMs; } \
double INDEX_NAME ::getQueryTimeMs() { return _this->_queryTimeMs; }

// ------------------------ top-level macro for convenience

#define DEFINE_INDEX(NAME, Scalar, VectorT, RowMatrixT, InnerIndexT) \
    INDEX_PIMPL(NAME, InnerIndexT) \
    INDEX_CTORS_DTOR(NAME, Scalar) \
    INDEX_QUERY_FUNCS(NAME, VectorT, RowMatrixT) \
    INDEX_STATS_FUNCS(NAME)

// ------------------------------------------------ MatmulIndex

DEFINE_INDEX(MatmulIndex, double, VectorXd, RowMatrixXd, nn::L2IndexBrute<double>)
DEFINE_INDEX(MatmulIndexF, float, VectorXf, RowMatrixXf, nn::L2IndexBrute<float>);


// TODO use reorder preproc once we verify that this is working
// DEFINE_INDEX(AbandonIndex, double, VectorXd, RowMatrixXd, nn::L2IndexAbandon<double>);



//     INDEX_PIMPL(MatmulIndexF, nn::L2IndexBrute<float>)

// // class MatmulIndexF ::Impl: public IndexImpl<nn::L2IndexBrute<float> > { \
// //     using Super = IndexImpl<nn::L2IndexBrute<float> >; \
// //     friend class MatmulIndexF; \
// //     using Super::Super; \
// // };

//     INDEX_CTORS_DTOR(MatmulIndexF, float, MatrixXf)

//     INDEX_QUERY_FUNCS(MatmulIndexF, VectorXf, RowMatrixXf)

// // vector<int64_t> MatmulIndexF ::radius(const VectorXf & q, double radiusL2) { \
// //     return _this->radius(q, radiusL2); \
// // } \
// // vector<int64_t> MatmulIndexF ::knn(const VectorXf & q, int k) { \
// //     return _this->knn(q, k); \
// // } \
// // MatrixXi MatmulIndexF ::radius_batch(const RowMatrixXf & queries, double radiusL2) { \
// //     return _this->radius_batch(queries, radiusL2); \
// // } \
// // MatrixXi MatmulIndexF ::knn_batch(const RowMatrixXf & queries, int k) { \
// //     return _this->knn_batch(queries, k); \
// // }

//     INDEX_STATS_FUNCS(MatmulIndexF)
