//
//  nn_index.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __NN_INDEX_HPP
#define __NN_INDEX_HPP

#include <memory>
#include "assert.h"

#include "Dense"

#include "nn_search.hpp"
#include "array_utils.hpp"
#include "flat_store.hpp"

namespace nn {


// ================================================================
// IndexBase
// ================================================================


//template<class IndexT> struct index_traits {
//    typedef typename IndexT::Scalar Scalar;
//    typedef typename IndexT::Distance Distance;
//};
// template<class IndexT> struct index_traits {};
// template<> struct<L2IndexBrute> index_traits {
//     typedef
// }

// superclass using Curiously Recurring Template Pattern to get compile-time
// polymorphism; i.e., we can call arbitrary child class functions without
// having to declare them anywhere in this class. This is the same pattern
// used by Eigen.
template<class Derived, class ScalarT, class DistT=float>
class IndexBase {
public:
 //    typedef typename index_traits<Derived>::Distance DistT;
	// typedef typename index_traits<Derived>::Scalar ScalarT;

//    // ------------------------------------------------ insert and erase
//
//	template<class Id>
//	auto insert(const ScalarT* row_start, Id id)
//        -> decltype(Derived::_data.insert(row_start, id))
//    {
//		return Derived::_data.insert(row_start, id);
//    }
//
//	template<class Id>
//	auto erase(Id id) -> decltype(Derived::_data.erase(id)) {
//        return Derived::_data.erase(id);
//    }

    // ------------------------------------------------ batch of queries

    template<class RowMatrixT>
    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
                                           DistT radius_sq)
    {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(radius(queries.row(j).eval(), radius_sq));
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, size_t k) {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(knn(queries.row(j).eval(), k));
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        vector<Neighbor> ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(onenn(queries.row(j).eval()));
        }
        return ret;
    }

    // ------------------------------------------------ return only idxs

    template<class VectorT>
    vector<idx_t> radius_idxs(const VectorT& query, DistT radius_sq) {
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

    template<class RowMatrixT>
    vector<vector<idx_t> > radius_batch_idxs(const RowMatrixT& queries,
                                             DistT radius_sq)
    {
        auto neighbors = Derived::radius_batch(queries, radius_sq);
        return idxs_from_nested_neighbors(neighbors);
    }

    template<class RowMatrixT>
    vector<idx_t> onenn_batch_idxs(const RowMatrixT& queries) {
        auto neighbors = Derived::onenn_batch(queries);
        return map([](const Neighbor& n) {
            return n.idx;
        }, neighbors);
    }

    template<class RowMatrixT>
    vector<vector<idx_t> > knn_batch_idxs(const RowMatrixT& queries, size_t k) {
        auto neighbors = Derived::knn_batch(queries, k);
        return _idxs_from_nested_neighbors(neighbors);
    }
};

// ================================================================
// L2IndexBrute
// ================================================================

// answers neighbor queries using matmuls
template<class T, class DistT=float>
class L2IndexBrute: public IndexBase<L2IndexBrute<T, DistT>, T, DistT> {
public:
    typedef T Scalar;
    typedef DistT Distance;
    typedef Matrix<Scalar, Dynamic, 1> ColVectorT;

    template<class RowMatrixT>
    explicit L2IndexBrute(const RowMatrixT& data):
        _data(data)
    {
		assert(data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
    }

    // // ------------------------------------------------ insert and erase

    // template<class Id>
    // void insert(const ScalarT* row_start, Id id) {
    //     auto changed_capacity = IndexBase::insert(row_start, id);
    //     if (changed_capacity > 0) {
    //         _rowNorms.conservativeResize(changed_capacity);
    //     }
    // }

    // template<class Id>
    // auto erase(Id id) -> decltype(IndexBase::erase(id)) {
    //     auto erase_idx = IndexBase::erase(id);
    //     if (erase_idx >= 0) {
    //         // auto last_idx = _dat - 1;
    //         _rowNorms(erase_idx) = _rowNorms(last_idx);
    //     }
    // }

    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query, DistT radius_sq) {
        return brute::radius(_data.matrix(), query, radius_sq);
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
        return brute::onenn(_data.matrix(), query);
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, int k) {
        return brute::knn(_data.matrix(), query, k);
    }

    // ------------------------------------------------ batch of queries

    template<class RowMatrixT>
    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
        DistT radius_sq)
    {
        return brute::radius_batch(_data.matrix(), queries, radius_sq, _rowNorms);
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, int k) {
        return brute::knn_batch(_data.matrix(), queries, k, _rowNorms);
    }

    template<class RowMatrixT>
    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        return brute::onenn_batch(_data.matrix(), queries, _rowNorms);
    }

private:
    RowStore<Scalar, 0> _data;
    ColVectorT _rowNorms;
};

// ================================================================
// L2IndexAbandon
// ================================================================

// TODO pad query if queryIdPadded == 0; maybe also do this in IndexBase?
template<class T, class DistT=float, int QueryIsPadded=1>
class L2IndexAbandon: public IndexBase<L2IndexAbandon<T, DistT, QueryIsPadded>, T, DistT> {
public:
    typedef T Scalar;
	typedef DistT Distance;
    typedef Matrix<Scalar, 1, Dynamic, RowMajor> RowVectT;
    typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

    template<class RowMatrixT>
    explicit L2IndexAbandon(const RowMatrixT& data):
        _data(data)
    {
        assert(_data.IsRowMajor);
        _colMeans = data.colwise().mean();
        // _orderIdxs = RowVectIdxsT(data.cols());
    }

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query, DistT radius_sq) {
        return abandon::radius(_data, query, radius_sq);
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
        return abandon::onenn(_data, query);
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, int k) {
        return abandon::knn(_data, query, k);
    }

private:
    RowStore<Scalar> _data;
    RowVectT _colMeans;
    // RowVectIdxsT _orderIdxs;
};


// ================================================================
// CascadeIndex
// ================================================================

template<class T>
class CascadeIndex {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixT;

// private:
    // L2IndexAbandon<
};

// template<class T, int dims>
// class PCABound {
// public:

// private:
//     RowStore<

// };

    // X TODO similar class called L2Index_abandon that does UCR ED
    // (without znorming)
    //  -prolly also give it batch funcs, but these just wrap the
    //  one-at-a-time func
    //  -also give it ability to store approx top eigenvects for early
    //  abandoning

    // X TODO all of the above need insert and delete methods
    //  -alloc a bigger mat than we need and store how many rows are used
    //  -wrap this up in its own class that's subclass of RowMatrixT

    // X TODO abstract superclass for all the KNN-ish classes

    // X TODO wrapper for eigen mat prolly needs to also pad ends of rows so
    // that each row is 128-bit aligned for our SIMD instrs
    //  -will prolly need to wrap rows in a Map to explicitly tell it the
    //  alignment though, cuz won't know it's safe at compile-time
    //  -just punt this for now cuz it's really ugly

    // X TODO brute force (not early abandoning) stuff needs num_rows to support
    // inserts and deletes

    // TODO tests for all of the above

    // TODO E2LSH impl
    // TODO Selective Hashing Impl on top of it

} // namespace nn
#endif // __NN_INDEX_HPP

