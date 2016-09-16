//
//  nn_index.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __NN_INDEX_HPP
#define __NN_INDEX_HPP

#include "assert.h"

#include "nn_search.hpp"
#include "array_utils.hpp"

namespace nn {

// superclass using Curiously Recurring Template Pattern to get compile-time
// polymorphism; i.e., we can call arbitrary child class functions without
// having to declare them anywhere in this class. This is the same pattern
// used by Eigen.
template<class Derived, class RowMatrixT, class dist_t=float>
class IndexBase {
public:
    // ------------------------------------------------ return only idxs

    template<class VectorT>
    vector<idx_t> radius_idxs(const VectorT& query, dist_t radius_sq) {
        auto neighbors = Derived::radius(query, radius_sq);
        return ar::map([](const Neighbor& n) {
            return n.idx;
        }, neighbors);
    }

    template<class VectorT>
    idx_t onenn_idxs(const VectorT& query) {
        auto neighbor = Derived::onenn(query);
        return query.idx;
    }

    template<class VectorT>
    vector<idx_t> knn_idxs(const VectorT& query, size_t k) {
        auto neighbors = Derived::knn(query, k);
        // TODO have neighbor idx be idx_t; we might overflow 32 bits so
        // needs to be 64
        return map([](const Neighbor& n) { return static_cast<idx_t>(n.idx); },
                   neighbors);
    }

    // ------------------------------------------------ batch return only idxs

    vector<vector<idx_t> > radius_batch_idxs(const RowMatrixT& queries,
                                             dist_t radius_sq)
    {
        auto neighbors = Derived::radius_batch(queries, radius_sq);
        return idxs_from_nested_neighbors(neighbors);
    }

    vector<idx_t> onenn_batch_idxs(const RowMatrixT& queries) {
        auto neighbors = Derived::onenn_batch(queries);
        return map([](const Neighbor& n) {
            return n.idx;
        }, neighbors);
    }

    vector<vector<idx_t> > knn_batch_idxs(const RowMatrixT& queries, size_t k) {
        auto neighbors = Derived::knn_batch(queries, k);
        return _idxs_from_nested_neighbors(neighbors);
    }
};

// ------------------------------------------------ L2IndexBrute

template<class RowMatrixT, class dist_t=float>
class L2IndexBrute: public IndexBase<L2IndexBrute<RowMatrixT, dist_t>, RowMatrixT, dist_t> {
public:

    typedef Matrix<typename RowMatrixT::Scalar, Dynamic, 1> ColVectorT;

    explicit L2IndexBrute(const RowMatrixT& data):
        _data(data)
    {
		assert(_data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
    }

    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query, float radius_sq) {
        return nn::radius(_data, query, radius_sq);
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
        return nn::onenn(_data, query);
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, int k) {
        return nn::knn(_data, query, k);
    }

    // ------------------------------------------------ batch of queries

    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
        float radius_sq)
    {
        return radius_batch(_data, queries, radius_sq, _rowNorms);
    }

    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, int k) {
        return knn_batch(_data, queries, k, _rowNorms);
    }

    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        return onenn_batch(_data, queries, _rowNorms);
    }

private:
    RowMatrixT _data;
    ColVectorT _rowNorms;
};


template<class RowMatrixT, class dist_t=float>
class L2IndexAbandon: public IndexBase<L2IndexAbandon<RowMatrixT>, RowMatrixT> {
public:

    typedef Matrix<typename RowMatrixT::Scalar, 1, Dynamic, RowMajor> RowVectT;
    typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

    explicit L2IndexAbandon(const RowMatrixT& data):
        _data(data)
    {
        _colMeans = data.colwise().mean();
        _orderIdxs = RowVectIdxsT(data.cols());
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
//      typename RowMatrixT::Index idx;
//      double dist = (_data.rowwise() - query.transpose()).rowwise().squaredNorm().minCoeff(&idx);
//      return {.dist = dist, .idx = static_cast<int32_t>(idx)}; // TODO fix length_t
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, size_t k) {
//      VectorT dists = squaredDistsToVector(_data, query);
//      return knn_from_dists(dists.data(), dists.size(), k);
    }

    // ------------------------------------------------ batch of queries

    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
                                           dist_t radius_sq)
    {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(radius(queries.row(j).eval(), radius_sq));
        }
        return ret;
    }

    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, size_t k) {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(knn(queries.row(j).eval(), k));
        }
        return ret;
    }

    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        vector<Neighbor> ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(onenn(queries.row(j).eval()));
        }
        return ret;
    }

private:
    RowMatrixT _data;
    RowVectT _colMeans;
    RowVectIdxsT _orderIdxs;
};


    // X TODO similar class called L2Index_abandon that does UCR ED
    // (without znorming)
    //  -prolly also give it batch funcs, but these just wrap the
    //  one-at-a-time func
    //  -also give it ability to store approx top eigenvects for early
    //  abandoning

    // TODO all of the above need insert and delete methods
    //  -alloc a bigger mat than we need and store how many rows are used
    //  -wrap this up in its own class that's subclass of RowMatrixT

    // X TODO abstract superclass for all the KNN-ish classes

    // TODO wrapper for eigen mat prolly needs to also pad ends of rows so
    // that each row is 128-bit aligned for our SIMD instrs
    //  -will prolly need to wrap rows in a Map to explicitly tell it the
    //  alignment though, cuz won't know it's safe at compile-time
    //  -just punt this for now cuz it's really ugly

    // TODO brute force (not early abandoning) stuff needs num_rows to support
    // inserts and deletes

    // TODO tests for all of the above

    // TODO E2LSH impl
    // TODO Selective Hashing Impl on top of it

} // namespace nn
#endif // __NN_INDEX_HPP

