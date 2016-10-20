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
// Helper Classes
// ================================================================

// ------------------------------------------------ IndexImpl

template<class IndexT>
class IndexImpl {
public:
    using Scalar = typename IndexT::Scalar;
    using VectorT = typename scalar_traits<Scalar>::ColVectorT;
    using RowMatrixT = typename scalar_traits<Scalar>::RowMatrixT;

    // ------------------------ ctors
    template<class... Args>
    IndexImpl(const RowMatrixT& X, Args&&... args):
        _indexStartTimeMs(timeNow()),
        _index(X, std::forward<Args>(args)...),
        _indexTimeMs(durationMs(_indexStartTimeMs, timeNow()))
    {}
    template<class... Args>
    IndexImpl(Scalar* X, int m, int n, Args&&... args):
        IndexImpl(eigenWrap2D_aligned(X, m, n), std::forward<Args>(args)...) {}

    // ------------------------ single queries
    template<class... Args>
    vector<int64_t> radius(const VectorT& q, double radiusL2, Args&&... args) {
        EasyTimer t(&_queryTimeMs);
        return _index.radius_idxs(q.transpose(), radiusL2,
            std::forward<Args>(args)...);
    }
    template<class... Args>
    vector<int64_t> knn(const VectorT& q, int k, Args&&... args) {
        EasyTimer t(&_queryTimeMs);
        return _index.knn_idxs(q.transpose(), k, std::forward<Args>(args)...);
    }

    // ------------------------ batch queries
    template<class... Args>
    MatrixXi radius_batch(const RowMatrixT& queries, double radiusL2,
        Args&&... args)
    {
        EasyTimer t(&_queryTimeMs);
        auto nested_neighbors = _index.radius_batch_idxs(
            queries, radiusL2, std::forward<Args>(args)...);
        return idx_mat_from_nested_neighbor_idxs(nested_neighbors);
    }
    template<class... Args>
    MatrixXi knn_batch(const RowMatrixT& queries, int k, Args&&... args) {
        EasyTimer t(&_queryTimeMs);
        auto nested_neighbors = _index.knn_batch_idxs(
            queries, k, std::forward<Args>(args)...);
        return idx_mat_from_nested_neighbor_idxs(nested_neighbors);
    }

protected:
    cputime_t _indexStartTimeMs;
    IndexT _index;
    double _indexTimeMs;
    double _queryTimeMs;
};

// ================================================================
// Index creation macros
// ================================================================

// ------------------------ pimpl

#define INDEX_PIMPL(INDEX_NAME, InnerIndexT) \
class INDEX_NAME ::Impl: public IndexImpl<InnerIndexT > { \
    using Super = IndexImpl<InnerIndexT >; \
    friend class INDEX_NAME; \
    using Super::Super; \
};

// ------------------------ ctors / dtors

#define INDEX_CTORS_DTOR(INDEX_NAME, Scalar, MatrixT) \
    INDEX_NAME ::INDEX_NAME(const MatrixT & X): _this{new INDEX_NAME ::Impl{X}} {} \
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
    INDEX_CTORS_DTOR(NAME, Scalar, RowMatrixT) \
    INDEX_QUERY_FUNCS(NAME, VectorT, RowMatrixT) \
    INDEX_STATS_FUNCS(NAME)

// ================================================================
// Index Impls
// ================================================================

// ------------------------------------------------ Matmul, Abandon, Simple

DEFINE_INDEX(MatmulIndex, double, VectorXd, RowMatrixXd, nn::L2IndexBrute<double>)
DEFINE_INDEX(MatmulIndexF, float, VectorXf, RowMatrixXf, nn::L2IndexBrute<float>);

template<class T>
using InnerIndexAbandonT = nn::L2IndexAbandon<T, nn::ReorderPreproc<T> >;
DEFINE_INDEX(AbandonIndex, double, VectorXd, RowMatrixXd, InnerIndexAbandonT<double>);
DEFINE_INDEX(AbandonIndexF, float, VectorXf, RowMatrixXf, InnerIndexAbandonT<float>);

template<class T>
using InnerIndexSimpleT = nn::L2IndexSimple<T>;
DEFINE_INDEX(SimpleIndex, double, VectorXd, RowMatrixXd, InnerIndexSimpleT<double>);
DEFINE_INDEX(SimpleIndexF, float, VectorXf, RowMatrixXf, InnerIndexSimpleT<float>);

// ------------------------------------------------ KmeansIndex

// ------------------------ type aliases

template<class T>
using KnnInnerIndexT = nn::L2IndexSimple<T>;
template<class T>
using KmeansIndexT = nn::L2KmeansIndex<T, KnnInnerIndexT<T> >;
// TODO remove these once using a macro
using Scalar = double;
using VectorT = VectorXd;
using RowMatrixT = RowMatrixXd;

// ------------------------ custom ctors (and dtor impl)

class KmeansIndex::Impl: public IndexImpl<KmeansIndexT<double> > {
    using Super = IndexImpl<KmeansIndexT<double> >;
    using Scalar = Super::Scalar;
    using VectorT = Super::VectorT;
    using RowMatrixT = Super::RowMatrixT;
    friend class KmeansIndex;
    // using Super::Super;

    Impl(const RowMatrixT& X, int k):
        Super(X, k) {}
    Impl(Scalar* X, int m, int n, int k):
        Super(X, m, n, k) {}
};

// ------------------------ macro invocations

// INDEX_PIMPL(KmeansIndex, KmeansIndexT<Scalar>)
// INDEX_CTORS_DTOR(KmeansIndex, Scalar, RowMatrixT)
// INDEX_QUERY_FUNCS(KmeansIndex, VectorT, RowMatrixT)
INDEX_STATS_FUNCS(KmeansIndex)

// ------------------------ ctors
// template<class... Args>
// IndexImpl(const RowMatrixT& X, Args&&... args):
// _indexStartTimeMs(timeNow()),
// _index(X, std::forward<Args>(args)...),
// _indexTimeMs(durationMs(_indexStartTimeMs, timeNow()))
// {}
// template<class... Args>
// IndexImpl(Scalar* X, int m, int n, Args&&... args):
// IndexImpl(eigenWrap2D_aligned(X, m, n), std::forward<Args>(args)...) {}

KmeansIndex ::KmeansIndex(const RowMatrixT & X, int k):
    _this{new KmeansIndex ::Impl{X, k}} {}

KmeansIndex ::KmeansIndex(Scalar* X, int m, int n, int k):
    _this{new KmeansIndex ::Impl{X, m, n, k}} {}

KmeansIndex ::~KmeansIndex() = default;

// ------------------------ custom ctors (and dtor impl)

vector<int64_t> KmeansIndex ::radius(const VectorT & q, double radiusL2,
    float search_frac)
{
	return _this->radius(q, radiusL2, search_frac);
}

vector<int64_t> KmeansIndex ::knn(const VectorT & q, int k, float search_frac) {
	return _this->knn(q, k, search_frac);
}

MatrixXi KmeansIndex ::radius_batch(const RowMatrixT & queries, double radiusL2,
    float search_frac)
{
	return _this->radius_batch(queries, radiusL2, search_frac);
}

MatrixXi KmeansIndex ::knn_batch(const RowMatrixT & queries, int k,
    float search_frac)
{
	return _this->knn_batch(queries, k, search_frac);
}

// DEFINE_INDEX(SimpleIndex, double, VectorXd, RowMatrixXd, InnerIndexSimpleT<double>);
// DEFINE_INDEX(SimpleIndexF, float, VectorXf, RowMatrixXf, InnerIndexSimpleT<float>);
