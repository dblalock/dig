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

    PRINT_VAR(max_num_neighbors);
    PRINT_VAR(num_rows);

    max_num_neighbors = ar::max(max_num_neighbors, 1); // no neighbors -> 1
    PRINT_VAR(max_num_neighbors);
    MatrixXi ret(num_rows, max_num_neighbors);
    for(int i = 0; i < num_rows; i++) {
        auto& neighbors = nested_neighbors[i];
        int j = 0;
        for (; j < neighbors.size(); j++) {
            ret(i, j) = static_cast<Index>(neighbors[j]);
        }
        for (; j < max_num_neighbors; j++) { // write -1 past end in each row
            ret(i, j) = -1;
            PRINT_VAR(ret(i, j));
        }
    }

    std::cout << "final idx mat:\n" << ret << "\n";
    return ret;
}

// ================================================================
// Classes
// ================================================================

// ------------------------------------------------ MatmulIndex::Impl

class MatmulIndex::Impl {
    friend class MatmulIndex;

    cputime_t _indexStartTimeMs;
    nn::L2IndexBrute<double> _index;
    double _indexTimeMs;
    double _queryTimeMs;

    // ctors
    Impl(const MatrixXd& X):
        _indexStartTimeMs(timeNow()),
        _index(X),
        _indexTimeMs(durationMs(_indexStartTimeMs, timeNow()))
    {}
    Impl(double* X, int m, int n): Impl(eigenWrap2D_aligned(X, m, n)) {}
};

// ------------------------------------------------ MatmulIndex

// ------------------------ ctors / dtors
MatmulIndex::MatmulIndex(const MatrixXd& X): _ths{new MatmulIndex::Impl{X}} {}
MatmulIndex::MatmulIndex(double* X, int m, int n):
    _ths{new MatmulIndex::Impl{X, m, n}}
{}

MatmulIndex::~MatmulIndex() = default; // needed for swig with unique_ptr


// ------------------------ single queries
vector<int64_t> MatmulIndex::radius(const VectorXd& q, double radiusL2) {
    auto t0 = timeNow();
    auto ret = _ths->_index.radius_idxs(q.transpose(), radiusL2);
    _ths->_queryTimeMs = durationMs(t0, timeNow());
    return ret;
}
vector<int64_t> MatmulIndex::knn(const VectorXd& q, int k) {
    auto t0 = timeNow();
    auto ret = _ths->_index.knn_idxs(q.transpose(), k);
    _ths->_queryTimeMs = durationMs(t0, timeNow());
    return ret;
}

// ------------------------ batch queries
MatrixXi MatmulIndex::radius_batch(const RowMatrixXd& queries, double radiusL2)
{
    auto t0 = timeNow();
    auto nested_neighbors = _ths->_index.radius_batch_idxs(queries, radiusL2);
    _ths->_queryTimeMs = durationMs(t0, timeNow());
    return idx_mat_from_nested_neighbor_idxs(nested_neighbors);
}
MatrixXi MatmulIndex::knn_batch(const RowMatrixXd& queries, int k) {
    auto t0 = timeNow();
    auto nested_neighbors = _ths->_index.knn_batch_idxs(queries, k);
    _ths->_queryTimeMs = durationMs(t0, timeNow());
    return idx_mat_from_nested_neighbor_idxs(nested_neighbors);
}

// ------------------------ stats
double MatmulIndex::getIndexConstructionTimeMs() { return _ths->_indexTimeMs; }
double MatmulIndex::getQueryTimeMs() { return _ths->_queryTimeMs; }
