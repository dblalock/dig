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

static const int8_t kAlignBytes = 32;

template<class IdT=idx_t>
class FlatIndex {
public:
    typedef IdT ID;

    FlatIndex(size_t len):
        _ids(ar::range(static_cast<ID>(0), static_cast<ID>(len)))
    {}

    idx_t rows() const { return static_cast<idx_t>(_ids.size()); }

protected:
    std::vector<ID> _ids;
};

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
class L2IndexBrute:
    public IndexBase<L2IndexBrute<T, DistT>, T, DistT>,
    public FlatIndex<> {
public:
    typedef T Scalar;
	typedef FlatIndex IDsStore;
    // typedef typename IDsStore::ID ID;
    typedef DistT Distance;
    typedef idx_t Index;
    // typedef DynamicRowArray<Scalar, 0> StorageT;
    // typedef typename StorageT::MatrixT::Index Index;
    typedef Matrix<Scalar, Dynamic, 1> ColVectorT;

    template<class RowMatrixT>
    explicit L2IndexBrute(const RowMatrixT& data):
        IDsStore(data.rows()),
        _data(data)
    {
		assert(data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
        //_ids = ar::range(static_cast<ID>(0), static_cast<ID>(data.rows()));
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
        return brute::radius(_matrix(), query, radius_sq);
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
        return brute::onenn(_matrix(), query);
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, int k) {
        return brute::knn(_matrix(), query, k);
    }

    // ------------------------------------------------ batch of queries

    template<class RowMatrixT>
    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
        DistT radius_sq)
    {
        return brute::radius_batch(_matrix(), queries, radius_sq, _rowNorms);
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, int k) {
        return brute::knn_batch(_matrix(), queries, k, _rowNorms);
    }

    template<class RowMatrixT>
    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        return brute::onenn_batch(_matrix(), queries, _rowNorms);
    }

    // ------------------------------------------------ accessors
	Index rows() const { return _ids.size(); }

private:
    ColVectorT _rowNorms; // TODO raw T[] + map
    DynamicRowArray<Scalar, 0> _data;

	auto _matrix() -> decltype(_data.matrix(rows())) {
		return _data.matrix(rows());
	}
};

// ================================================================
// L2IndexAbandon
// ================================================================

// TODO pad query if queryIdPadded == 0; maybe also do this in IndexBase?
template<class T, class DistT=float, int QueryIsPadded=1>
class L2IndexAbandon:
    public IndexBase<L2IndexAbandon<T, DistT, QueryIsPadded>, T, DistT>,
    public FlatIndex<> {
public:
    typedef T Scalar;
	typedef DistT Distance;
	typedef idx_t Index;
    typedef Matrix<Scalar, 1, Dynamic, RowMajor> RowVectT;
    typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

    template<class RowMatrixT>
    explicit L2IndexAbandon(const RowMatrixT& data):
		FlatIndex<>(data.rows()),
        _data(data)
    {
        assert(_data.IsRowMajor);
    }

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query, DistT radius_sq) {
        return abandon::radius(_data, query, radius_sq, rows());
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
        return abandon::onenn(_data, query, kMaxDist, rows());
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, int k) {
        return abandon::knn(_data, query, k, kMaxDist, rows());
    }

	// ------------------------------------------------ accessors
	Index rows() const { return _ids.size(); }

private:
    DynamicRowArray<Scalar, kAlignBytes> _data;
    // RowVectT _colMeans;
};

// ================================================================
// CascadeIdIndex
// ================================================================

// only holds ids of points, not points themselves
template<class T, int Projections=16, class DistT=float>
// class CascadeIdIndex: IndexBase<CascadeIdIndex<T, Projections, DistT>, T, DistT> {
class CascadeIdIndex {
public:
    typedef T Scalar;
    typedef float ProjectionScalar;
    typedef int32_t Index;
    typedef idx_t ID;
    enum { NumProjections = Projections };

    CascadeIdIndex(Index initial_rows):
        _ids(initial_rows),
        _projections(initial_rows),
        _nrows(0),
        _capacity(initial_rows)
    {
        assert(initial_rows > 0);
    }

    // ------------------------ single query

    // Note: doesn't return distances; just indices
    template<class VectorT, class RowMatrixT>
    vector<Index> radius(const RowMatrixT& data, const VectorT& query,
        DistT radius_sq)
    {
        return abandon::radius<Index>(data, query, radius_sq, rows());
    }

    // ------------------------ insert and erase

    void insert(ID id, ProjectionScalar* projections) {
        if (++_nrows > _capacity) {
            Index new_capacity = std::max(_nrows + 1, _capacity * 1.5);
            _ids.resize(_capacity, new_capacity);
            _projections.resize(_capacity, new_capacity);
            _capacity = new_capacity;
        }
        auto at_idx = _nrows - 1;
        _ids.insert<false>(id, at_idx);
        _projections.insert<false>(projections, at_idx);
    }

    bool erase_at(Index at_idx) {
        assert(at_idx >= 0);
        assert(_nrows > 0);
        assert(at_idx < _nrows);
        _nrows--;
        if (at_idx == (_nrows + 1)) { return true; } // erase last row
        // overwrite erased row with last row
        _ids.copy_row(_nrows, at_idx);
        _projections.copy_row(_nrows, at_idx);
    }

    bool erase(ID id) {
        auto begin = _ids.data();
        auto it = ar::find(begin, begin + _nrows, id);
        if (it == end) {
            return false;
        }
        auto idx = static_cast<Index>(it - begin);
        return erase_at(idx);
    }

    // ------------------------ accessors
    Index rows() const { return _nrows; }
    Index capacity() const { return _capacity; }

private:
    FixedRowArray<idx_t, 1, 0> _ids; // 1 col, 0 byte align         // 8B
    FixedRowArray<ProjectionScalar, NumProjections, kAlignBytes> _projections; // 8B
    // DynamicRowArray<Scalar, kAlignBytes>                         // 16B
    Index _nrows;                                                 // 4B
    Index _capacity;                                              // 4B
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

