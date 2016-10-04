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

#include "cluster.hpp"
#include "nn_search.hpp"
#include "array_utils.hpp"
#include "flat_store.hpp"

#include "debug_utils.hpp"

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

    // ------------------------------------------------ single queries

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
	// typedef FlatIndex IDsStore;
    // typedef typename IDsStore::ID ID;
    typedef DistT Distance;
    typedef idx_t Index;
    // typedef DynamicRowArray<Scalar, 0> StorageT;
    // typedef typename StorageT::MatrixT::Index Index;
    typedef Matrix<Scalar, Dynamic, 1> ColVectorT;

    template<class RowMatrixT>
    explicit L2IndexBrute(const RowMatrixT& data):
        FlatIndex(data.rows()),
        _data(data)
    {
		assert(data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
        //_ids = ar::range(static_cast<ID>(0), static_cast<ID>(data.rows()));
    }

    // // ------------------------------------------------ accessors
    // Index rows() const { return _ids.size(); }

    // ------------------------------------------------ insert and erase

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

private:
    ColVectorT _rowNorms; // TODO raw T[] + map
    // FixedRowArray<Scalar, 1, 0> _rowNorms;
    DynamicRowArray<Scalar, 0> _data;

    // auto _row_norms() const { // TODO use this once _rowNorms is RowArray
    //     return _rowNorms.matrix(rows());
    // }

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

	// // ------------------------------------------------ accessors
	// Index rows() const { return _ids.size(); }

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
    FixedRowArray<idx_t, 1, 0, true> _ids; // 1 col, 0 byte align   // 8B
    FixedRowArray<ProjectionScalar, NumProjections, kAlignBytes, true> _projections; // 8B
    // DynamicRowArray<Scalar, kAlignBytes>                         // 16B
    Index _nrows;                                                   // 4B
    Index _capacity;                                                // 4B
};

// ================================================================
// Preprocessing
// ================================================================

// template<class T>
struct IdentityPreproc {
    template<class RowMatrixT> IdentityPreproc(const RowMatrixT& X) {}

    template<class VectorT>
    void preprocess(const VectorT& query, typename VectorT::Scalar* out) {}
    template<class VectorT>
    VectorT& preprocess(VectorT& query) { return query; }

    template<class RowMatrixT, class RowMatrixT2>
    void preprocess_batch(const RowMatrixT& queries, RowMatrixT2& out) {}
    template<class RowMatrixT>
    RowMatrixT& preprocess_batch(RowMatrixT& queries) { return queries; }

    template<class RowMatrixT>
    RowMatrixT& preprocess_data(RowMatrixT& X) { return X; }
};

template<class ScalarT, int AlignBytes=kAlignBytes>
struct ReorderPreproc { // TODO could avoid passing ScalarT template arg
    using Scalar = ScalarT;
    using Index = int32_t;
    enum { max_pad_elements = AlignBytes / sizeof(Scalar) };

    template<class RowMatrixT> ReorderPreproc(const RowMatrixT& X):
            _order(nn::order_col_variance(X))
    {
        assert(_length() > max_pad_elements);
    }
        // _order(_aligned_length(X.cols())) {
        //     auto col_order = nn::order_col_variance(X);
        //     assert(col_order.size() <= _order.size());

        //     first X.cols() entries are sorted; zero-padding after that
        //     just remains ordered at the end
        //     for (int i = 0; i < col_order.size(); i++) {
        //         _order[i] = col_order[i];
        //     }
        //     for (int i = 0; i < _order.size(); i++) {
        //         _order[i] = i;
        //     }
        // }

    // ------------------------ one query
    // assumes that out already has appropriate padding at the end
    template<class VectorT> void preprocess(const VectorT& query,
        typename VectorT::Scalar* out) const
    {
        // PRINT_VAR(_length());
        // auto query_str = ar::to_string(query.data(), query.size());
        // PRINT_VAR(query_str);

        assert(_length() == _aligned_length(query.size()));
        reorder_query(query, _order, out); // TODO uncomment after debug
        // for (int i = 0; i < query.size(); i++) {
        //     out[i] = query.data()[i];
        // }
        // auto outpt_str = ar::to_string(out, _length());
        // PRINT_VAR(outpt_str);
    }
    template<class VectorT> VectorT preprocess(const VectorT& query) const {
        VectorT out(_length());
        // out.segment(query.size(), _length() - query.size()) = 0;
        out.setZero();
        preprocess(query, out.data());
		return out;
    }

    // ------------------------ batch of queries
    template<class RowMatrixT>
    void preprocess_batch(const RowMatrixT& queries, RowMatrixT& out) const {
        assert(queries.IsRowMajor);
        assert(out.IsRowMajor);
        assert(queries.rows() <= out.rows());
        assert(queries.cols() <= out.cols());
        for (Index i = 0; i < queries.rows(); i++) {
            preprocess(queries.row(i), out.row(i).data());
        }
    }
    template<class RowMatrixT>
    RowMatrixT preprocess_batch(const RowMatrixT& queries) const {
        assert(false); // verify we're not actually calling this yet
        RowMatrixT out(queries.rows(), _length());
        // out.topRightCorner(queries.rows(), max_pad_elements).setZero(); // TODO uncomment
        out.setZero();
        preprocess_batch(queries, out);
        return out;
    }

    // ------------------------ data mat
    template<class RowMatrixT> RowMatrixT preprocess_data(
		const RowMatrixT& X) const
	{
        assert(X.IsRowMajor);
		assert(X.cols() <= _length());
		assert(X.cols() > 0);

  //       // TODO rm after debug
  //       RowMatrixT out(X.rows(), _length());
		// // out.setZero();
  //       out.topRightCorner(X.rows(), max_pad_elements).setZero();
  //       out.topLeftCorner(X.rows(), X.cols()) = X;
  //       for (int i = 0; i < X.rows(); i++) {
  //           for(long j = X.cols(); j < _length(); j++) {
  //               if (out(i, j) != 0) {
  //                   DEBUGF("%d, %ld", i, j);
  //                   PRINT_VAR(out(i, j));
		// 			assert(false);
  //               }
  //           }
  //       }
  //       return out;

        // create mat to return and ensure final cols are zeroed
        RowMatrixT out(X.rows(), _length());
        out.topRightCorner(X.rows(), max_pad_elements).setZero();

        out.setZero(); // TODO rm after debug

        // preprocess it the same way as queries
        preprocess_batch(X, out);
        return out;
    }
private:
    Index _aligned_length(int64_t n) const {
        return static_cast<Index>(aligned_length<Scalar, AlignBytes>(n));
    }
    Index _length() const { return _aligned_length(_order.size()); }
    std::vector<Index> _order;
//    std::unique_ptr<Scalar[]> ;
};


// ================================================================
// NNIndex
// ================================================================

#define QUERY_METHOD(NAME, INVOCATION) \
    template<class VectorT, class... Args> \
    auto NAME(const VectorT& query, Args&&... args) \
        -> decltype(INVOCATION (query, std::forward<Args>(args)...)) \
    { \
        auto use_query = _preproc.preprocess(query); \
        return INVOCATION(use_query, std::forward<Args...>(args)...); \
    }

#define QUERY_BATCH_METHOD(NAME, INVOCATION) \
    template<class RowMatrixT, class... Args> \
    auto NAME(const RowMatrixT& queries, Args&&... args) \
        -> decltype(INVOCATION (queries, std::forward<Args>(args)...)) \
    { \
        auto use_queries = _preproc.preprocess_batch(queries); \
        return INVOCATION(use_queries, std::forward<Args...>(args)...); \
    }

template<class InnerIndex, class Preprocessor=IdentityPreproc>
class NNIndex {
private:
    Preprocessor _preproc;
	InnerIndex _index;
public:
    using Scalar = typename InnerIndex::Scalar;
    using Index = typename InnerIndex::Index;

	template<class RowMatrixT>
	NNIndex(const RowMatrixT& X):
		_preproc{X}, // TODO warns about preproc uninitialized if () not {} ??
		_index(_preproc.preprocess_data(X))
	{}

  //   template<class RowMatrixT, class... Args>
  //   NNIndex(const RowMatrixT& X, Args&&... args):
		// _preproc{X}, // TODO warns about preproc uninitialized if () not {} ??
  //       _index(_preproc.preprocess_data(X), std::forward<Args>(args)...)
  //   {}

    QUERY_METHOD(radius, _index.radius);
    QUERY_METHOD(onenn, _index.onenn);
    QUERY_METHOD(knn, _index.knn);
    QUERY_BATCH_METHOD(radius_batch, _index.radius_batch);
    QUERY_BATCH_METHOD(onenn_batch, _index.onenn_batch);
    QUERY_BATCH_METHOD(knn_batch, _index.knn_batch);
};

    // TODO E2LSH impl
    // TODO Selective Hashing Impl on top of it

} // namespace nn
#endif // __NN_INDEX_HPP

