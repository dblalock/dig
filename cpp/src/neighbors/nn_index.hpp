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
#include "eigen_utils.hpp"
#include "flat_store.hpp"

// #include "debug_utils.hpp"

namespace nn {

static const int8_t kAlignBytes = 32;

// ================================================================
// FlatIdStore / Neighbor postprocessing
// ================================================================

struct IdentityIdStore {
    template<class T> T postprocess(const T& neighbors) { return neighbors; }
};

template<class IdT=idx_t>
class FlatIdStore {
public:
    typedef IdT ID;

    FlatIdStore(size_t len):
        _ids(ar::range(static_cast<ID>(0), static_cast<ID>(len)))
    {}

    FlatIdStore() = default;
    FlatIdStore(const FlatIdStore& rhs) = delete;
    // FlatIdStore(FlatIdStore&& rhs): _ids(std::move(rhs._ids)) {
    //     PRINT("move ctor")
    //     PRINT_VAR(_ids.size());
    // }

    idx_t rows() const { return static_cast<idx_t>(_ids.size()); }

    // TODO keep this protected and add it as optional arg to ctor
    std::vector<ID>& ids() { return _ids; }

    // ------------------------ neighbor postprocessing
    // ie, convert indices within storage to point IDs

    Neighbor postprocess(Neighbor n) {
        return Neighbor{.dist = n.dist, .idx = _ids[n.idx]};
    }
    std::vector<Neighbor> postprocess(std::vector<Neighbor> neighbors) {
        return ar::map([this](const Neighbor n) {
			return this->postprocess(n);
        }, neighbors);
    }
    std::vector<std::vector<Neighbor> > postprocess(
        std::vector<std::vector<Neighbor> > nested_neighbors) {
        return ar::map([this](const std::vector<Neighbor> neighbors) {
            return this->postprocess(neighbors);
        }, nested_neighbors);
    }

protected:
    std::vector<ID> _ids;
};

// ================================================================
// Preprocessing
// ================================================================

struct IdentityPreproc {
    // ------------------------ ctors
    template<class RowMatrixT> IdentityPreproc(const RowMatrixT& X) {}
    IdentityPreproc() = default;

    // ------------------------ identity funcs
    template<class VectorT>
    void preprocess(const VectorT& query, typename VectorT::Scalar* out) const {}
    template<class VectorT>
    VectorT& preprocess(VectorT& query) const { return query; }

    template<class RowMatrixT, class RowMatrixT2>
    void preprocess_batch(const RowMatrixT& queries, RowMatrixT2& out) const {}
    template<class RowMatrixT>
    RowMatrixT& preprocess_batch(RowMatrixT& queries) const { return queries; }

    template<class RowMatrixT>
    RowMatrixT& preprocess_data(RowMatrixT& X) const { return X; }
};

template<class ScalarT, int AlignBytes=kAlignBytes>
struct ReorderPreproc { // TODO could avoid passing ScalarT template arg
    // ------------------------ consts
    using Scalar = ScalarT;
    using Index = int32_t;
    enum { max_pad_elements = AlignBytes / sizeof(Scalar) };

    // ------------------------ ctors
    ReorderPreproc() = default;

    template<class RowMatrixT> ReorderPreproc(const RowMatrixT& X):
            _order(nn::order_col_variance(X))
    {
        assert(_length() > max_pad_elements);
    }

    // ------------------------ one query
    // assumes that out already has appropriate padding at the end
    template<class VectorT> void preprocess(const VectorT& query,
        typename VectorT::Scalar* out) const
    {
        assert(_length() == _aligned_length(query.size()));
        reorder_query(query, _order, out);
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

        // create mat to return and ensure final cols are zeroed
        RowMatrixT out(X.rows(), _length());
        out.topRightCorner(X.rows(), max_pad_elements).setZero();

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
//
// The way this class works is that it defines public methods for everything a
// Nearest Neighbor index should be able to do (except insert/delete atm)
// and calls protected funcs, prefixed by an "_", for the actual logic; the
// public funcs preprocess the query (e.g., to reorder or normalize it) and
// postprocess the returned neighbors (e.g., to convert from row indices
// to point IDs). The protected methods should be oblivious to pre- and post-
// processing. There are default impls of some protected methods (the batch
// queries). Also, don't override the *_idxs() methods because then you won't
// get the pre- and post-processing; you shouldn't need to touch them anyway.
//
// In short, subclasses need to implement:
//
//  -_radius(query, d_max)
//  -_onenn(query)
//  -_knn(query, k)
//
//  -preprocess(query)
//  -preprocess_batch(query)
//
//  -postprocess(Neighbor)
//  -postprocess(vector<Neighbor>)
//  -postprocess(vector<vector<Neighbor>>)
//
// and can choose to override:
//
//  -_radius_batch(queries, d_max)
//  -_onenn_batch(queries)
//  -_knn_batch(queries, k)
//
template<class Derived, class ScalarT>
class IndexBase {
public:
    using DistT = ScalarT;

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

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query,
                                           DistT d_max)
    {
        auto neighbors = _derived()->_radius(_derived()->preprocess(query), d_max);
        return _derived()->postprocess(neighbors);
    }

    template<class VectorT>
    Neighbor onenn(const VectorT& query) {
        auto neighbors = _derived()->_onenn(_derived()->preprocess(query));
        return _derived()->postprocess(neighbors);
    }

    template<class VectorT>
    vector<Neighbor> knn(const VectorT& query, int k) {
        auto neighbors = _derived()->_knn(_derived()->preprocess(query), k);
        return _derived()->postprocess(neighbors);
    }

    // ------------------------------------------------ batch of queries
    // TODO allow vector of d_max values for each of these

    template<class RowMatrixT>
    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
                                           DistT d_max)
    {
        auto& queries_proc = Derived::preprocess_batch(queries);
        auto neighbors = Derived::_radius_batch(queries_proc, d_max);
        return Derived::postprocess(neighbors);
    }

    template<class RowMatrixT>
    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        auto& queries_proc = Derived::preprocess_batch(queries);
        auto neighbors = Derived::_onenn_batch(queries_proc);
        return Derived::postprocess(neighbors);
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, size_t k) {
        auto& queries_proc = Derived::preprocess_batch(queries);
        auto neighbors = Derived::_knn_batch(queries_proc, k);
        return Derived::postprocess(neighbors);
    }

    // ------------------------------------------------ return only idxs

    template<class VectorT>
    vector<idx_t> radius_idxs(const VectorT& query, DistT d_max) {
        auto neighbors = Derived::radius(query, d_max);
        return ar::map([](const Neighbor& n) {
            return n.idx;
        }, neighbors);
    }

    template<class VectorT>
    idx_t onenn_idxs(const VectorT& query, DistT d_max=kMaxDist) {
        auto neighbor = Derived::onenn(query);
        return query.idx;
    }

    template<class VectorT>
    vector<idx_t> knn_idxs(const VectorT& query, size_t k,
        DistT d_max=kMaxDist)
    {
        auto neighbors = Derived::knn(query, k);
        return map([](const Neighbor& n) { return static_cast<idx_t>(n.idx); },
                   neighbors);
    }

    // ------------------------------------------------ batch return only idxs
    // TODO allow vector of d_max values for each of these

    template<class RowMatrixT>
    vector<vector<idx_t> > radius_batch_idxs(const RowMatrixT& queries,
                                             DistT d_max)
    {
        auto neighbors = Derived::radius_batch(queries, d_max);
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

protected: // default impls for derived

    // ------------------------------------------------ batch of queries
    // TODO allow vector of d_max values for each of these

    template<class RowMatrixT>
    vector<vector<Neighbor> > _radius_batch(const RowMatrixT& queries,
                                           DistT d_max)
    {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(_derived()->_radius(queries.row(j).eval(), d_max));
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > _knn_batch(const RowMatrixT& queries, size_t k) {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(_derived()->_knn(queries.row(j).eval(), k));
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<Neighbor> _onenn_batch(const RowMatrixT& queries) {
        vector<Neighbor> ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(_derived()->_onenn(queries.row(j).eval()));
        }
        return ret;
    }
private:
    // inline const Derived* _derived() const {
    //     return static_cast<const Derived*>(this);
    // }
    inline Derived* _derived() {
        return static_cast<Derived*>(this);
    }
};

// ================================================================
// L2IndexBrute
// ================================================================

// answers neighbor queries using matmuls
template<class T, class PreprocT=IdentityPreproc>
class L2IndexBrute:
    public IndexBase<L2IndexBrute<T, PreprocT>, T>,
    public PreprocT,
    public FlatIdStore<> {
    friend class IndexBase<L2IndexBrute<T, PreprocT>, T>;
public:
    typedef T Scalar;
    typedef T DistT;
    typedef idx_t Index;
    typedef Matrix<Scalar, Dynamic, 1> ColVectorT;

    // ------------------------------------------------ ctors

    L2IndexBrute() = default;

    template<class RowMatrixT>
    explicit L2IndexBrute(const RowMatrixT& data):
        PreprocT(data),
        FlatIdStore(data.rows()),
        _data(PreprocT::preprocess_data(data))
    {
		assert(data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
    }

 //    L2IndexBrute(L2IndexBrute&& rhs) noexcept:
 //        FlatIdStore(std::move(rhs)),
 //        _data(std::move(rhs._data))
 //    {
	// 	PRINT_VAR(_ids.size());
	// }

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

protected:
    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> _radius(const VectorT& query, DistT d_max) {
        return brute::radius(_matrix(), query, d_max);
    }

    template<class VectorT>
    Neighbor _onenn(const VectorT& query, DistT d_max=kMaxDist) {
        return brute::onenn(_matrix(), query);
    }

    template<class VectorT>
    vector<Neighbor> _knn(const VectorT& query, int k, DistT d_max=kMaxDist) {
        return brute::knn(_matrix(), query, k);
    }

    // ------------------------------------------------ batch of queries

    template<class RowMatrixT>
    vector<vector<Neighbor> > _radius_batch(const RowMatrixT& queries,
        DistT d_max)
    {
        return brute::radius_batch(_matrix(), queries, d_max, _rowNorms);
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > _knn_batch(const RowMatrixT& queries, int k) {
        return brute::knn_batch(_matrix(), queries, k, _rowNorms);
    }

    template<class RowMatrixT>
    vector<Neighbor> _onenn_batch(const RowMatrixT& queries) {
        return brute::onenn_batch(_matrix(), queries, _rowNorms);
    }

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

template<class T, class PreprocT=IdentityPreproc>
class L2IndexAbandon:
    public IndexBase<L2IndexAbandon<T, PreprocT>, T>,
    public PreprocT,
    public FlatIdStore<> {
    friend class IndexBase<L2IndexAbandon<T, PreprocT>, T>;
public:
    typedef T Scalar;
	typedef T DistT;
	typedef idx_t Index;
    typedef Matrix<Scalar, 1, Dynamic, RowMajor> RowVectT;
    typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

    // ------------------------------------------------ ctors

	L2IndexAbandon() = default;

    template<class RowMatrixT>
    explicit L2IndexAbandon(const RowMatrixT& data):
        PreprocT(data),
		FlatIdStore(data.rows()),
		_data(PreprocT::preprocess_data(data))
        // _data(data)
    {
        assert(_data.IsRowMajor);
    }
    // L2IndexAbandon(L2IndexAbandon&& rhs) noexcept:
    //     FlatIdStore(std::move(rhs)),
    //     _data(std::move(rhs._data))
    // {}

protected:
    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> _radius(const VectorT& query, DistT d_max) {
        return abandon::radius(_data, query, d_max, rows());
    }

    template<class VectorT>
    Neighbor _onenn(const VectorT& query, DistT d_max=kMaxDist) {
        return abandon::onenn(_data, query, d_max, rows());
    }

    template<class VectorT>
    vector<Neighbor> _knn(const VectorT& query, int k, DistT d_max=kMaxDist) {
        return abandon::knn(_data, query, k, d_max, rows());
    }

private:
    DynamicRowArray<Scalar, kAlignBytes> _data;
};


// ================================================================
// KmeansIndex
// ================================================================

template<class T, class InnerIndex=L2IndexAbandon<T>,
    class PreprocT=IdentityPreproc>
class L2KmeansIndex:
    public IndexBase<L2KmeansIndex<T, InnerIndex, PreprocT>, T>,
	public PreprocT,
    public IdentityIdStore {
    friend class IndexBase<L2KmeansIndex<T, InnerIndex, PreprocT>, T>;
public:
    using Scalar = T;
    using DistT = T;
    using Index = idx_t;
	using CentroidIndex = int32_t;
	using RowMatrixType = RowMatrix<T>;

	L2KmeansIndex() = delete;
	L2KmeansIndex(const L2KmeansIndex& rhs) = delete;
	L2KmeansIndex(L2KmeansIndex&& rhs) = delete;

    template<class RowMatrixT>
    L2KmeansIndex(const RowMatrixT& X, int k):
        PreprocT(X),
        _order(k),
        _indexes(new InnerIndex[k]),
        _centroid_dists(k),
        _idxs_for_centroids(k),
        _queries_storage(32, PreprocT::preprocess_data(X).cols()), // 32 query capacity
        _num_centroids(k)
    {
        // _indexes.reserve(k);
        auto X_ = PreprocT::preprocess_data(X);

		auto centroids_assignments = cluster::kmeans(X_, k);
        _centroids = centroids_assignments.first;
        auto assigs = centroids_assignments.second;

        // create vect of which vects are associated with each centroid
        // using assig_idx_t = assigs::value_type
        // using idx_vect_t = std::vector<assig_idx_t>;
        using idx_vect_t = std::vector<Index>;
        std::vector<idx_vect_t> idxs_for_centroids(k);
        for (int i = 0; i < assigs.size(); i++) {
            auto idx = assigs[i];
            idxs_for_centroids[idx].push_back(i);
        }
        auto lengths = ar::map([](const idx_vect_t& v) {
            return v.size();
        }, idxs_for_centroids);
        auto max_len = ar::max(lengths);

        // create an index for each centroid
        RowMatrixT storage(max_len, X_.cols());
        for (int i = 0; i < k; i++) {
            // copy appropriate rows to temp storage so they're contiguous
            auto& idxs = idxs_for_centroids[i];
			for (int row_idx = 0; row_idx < idxs.size(); row_idx++) {
				auto idx = idxs[row_idx];
				storage.row(row_idx) = X_.row(idx);
			}
            // create an InnerIndex instance with these rows
            auto nrows = idxs.size();
            new (&_indexes[i]) InnerIndex(storage.topRows(nrows));

            _indexes[i].ids() = idxs; // TODO rm direct set of idxs
        }
    }

    // ------------------------------------------------ accessors

 //    CentroidIndex _num_centroids const {
	// 	return static_cast<CentroidIndex>(_centroids.size());
	// }

    // ------------------------------------------------ single query

protected:
    template<class VectorT>
    vector<Neighbor> _radius(const VectorT& query, DistT d_max,
        CentroidIndex centroids_limit=-1)
    {
        _update_order_for_query(query, centroids_limit); // update _order

        vector<Neighbor> ret;
        for (int i = 0; i < _order.size(); i++) {
            auto& index = _indexes[_order[i]];
            auto neighbors = index.radius(query, d_max);

            // // convert to orig idxs TODO rm once idxs do this themselves
            // for (auto& n : neighbors) {
            //     n.idx = index.ids()[n.idx];
            // }

            ar::concat_inplace(ret, neighbors);
        }
        return ret;
    }

    template<class VectorT>
    Neighbor _onenn(const VectorT& query, CentroidIndex centroids_limit=-1,
        DistT d_max=kMaxDist)
    {
        // PRINT_VAR(_num_centroids);
        _update_order_for_query(query, centroids_limit); // update _order

        Neighbor ret{.dist = d_max, .idx = kInvalidIdx};
        // PRINT_VAR(_order.size());
        for (int i = 0; i < _order.size(); i++) {
            auto& index = _indexes[_order[i]];
            if (index.rows() < 1) { continue; }

            Neighbor n = index.onenn(query);
            if (n.dist < ret.dist) {
                ret = n;
                // ret.dist = n.dist;
                // ret.idx = index.ids()[n.idx];
            }
        }
        return ret;
    }

    template<class VectorT>
    vector<Neighbor> _knn(const VectorT& query, int k, CentroidIndex centroids_limit=-1,
        DistT d_max=kMaxDist)
    {
        _update_order_for_query(query, centroids_limit); // update _order

        vector<Neighbor> ret(k, {.dist = d_max, .idx = kInvalidIdx});
        for (int i = 0; i < _order.size(); i++) {
            auto& index = _indexes[_order[i]];
            if (index.rows() < 1) continue;

            auto neighbors = index.knn(query, k);

            // // convert to orig idxs TODO rm once idxs do this themselves
            // for (auto& n : neighbors) {
            //     n.idx = index.ids()[n.idx];
            // }

            d_max = maybe_insert_neighbors(ret, neighbors);
        }
		return ret;
    }

    // ------------------------------------------------ batch of queries

    template<class RowMatrixT>
    vector<vector<Neighbor> > _radius_batch(const RowMatrixT& queries,
        DistT d_max, CentroidIndex centroids_limit=-1)
    {
        _update_idxs_for_centroids(queries, centroids_limit);
        vector<vector<Neighbor> > ret(queries.rows());

        for (int kk = 0; kk < _num_centroids; kk++) { // for each centroid
            auto& query_idxs = _idxs_for_centroids[kk];
            auto nrows = query_idxs.size();
            _update_query_storate(query_idxs, queries);

            // get the neighbors for each query around this centroid, and
            // append them to the master list of neighbors for each query
            auto nested_neighbors = _indexes[kk].radius_batch(
                _queries_storage, d_max);
            assert(nrows == nested_neighbors.size());
            for (int qq = 0; qq < nrows; qq++) {
                auto query_idx = query_idxs[qq];
                ar::concat_inplace(ret[query_idx], nested_neighbors[qq]);
            }
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<Neighbor> _onenn_batch(const RowMatrixT& queries,
        CentroidIndex centroids_limit=-1)
    {
        _update_idxs_for_centroids(queries, centroids_limit);
        vector<Neighbor> ret(queries.rows());
        for (int kk = 0; kk < _num_centroids; kk++) {
            auto& query_idxs = _idxs_for_centroids[kk];
            auto nrows = query_idxs.size();
            _update_query_storate(query_idxs, queries);

            auto neighbors = _indexes[kk].onenn_batch(_queries_storage);
            assert(nrows == neighbors.size());
            for (int qq = 0; qq < nrows; qq++) {
                auto query_idx = query_idxs[qq];
                ret[query_idx].push_back(neighbors[qq]);
            }
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > _knn_batch(const RowMatrixT& queries, int k,
        CentroidIndex centroids_limit=-1)
    {
        _update_idxs_for_centroids(queries, centroids_limit);
        vector<vector<Neighbor> > ret(queries.rows());

        for (int kk = 0; kk < _num_centroids; kk++) { // for each centroid
            auto& query_idxs = _idxs_for_centroids[kk];
            auto nrows = query_idxs.size();
            _update_query_storate(query_idxs, queries);

            auto nested_neighbors = _indexes[kk].knn_batch(_queries_storage, k);
            assert(nrows == nested_neighbors.size());
            for (int qq = 0; qq < nrows; qq++) {
                auto query_idx = query_idxs[qq];
                maybe_insert_neighbors(ret[query_idx], nested_neighbors[qq]);
            }
        }
        return ret;
    }

private:
    RowMatrixType _centroids;
    // std::vector<InnerIndex> _indexes;
    std::unique_ptr<InnerIndex[]> _indexes;
    // the fields below are storage to avoid doing allocs for each query
    // XXX: shared storage makes this class not remotely thread safe
    // std::vector<CentroidIndex> _order;
    std::vector<CentroidIndex> _order;
    ColVector<Scalar> _centroid_dists;
    std::vector<std::vector<CentroidIndex> > _idxs_for_centroids;
    ColMatrix<Scalar> _queries_storage;
    CentroidIndex _num_centroids;

    CentroidIndex _clamp_centroid_limit(CentroidIndex centroids_limit) {
        if (centroids_limit < 1) { return _num_centroids; }
        return ar::min(_num_centroids, centroids_limit);
    }

    template<class VectorT>
    void _update_order_for_query(const VectorT& query, CentroidIndex centroids_limit=-1) {
        _centroid_dists = dist::squared_dists_to_vector(_centroids, query);
        ar::argsort(_centroid_dists.data(), _num_centroids, _order.data());
        centroids_limit = _clamp_centroid_limit(centroids_limit);
        _order.resize(centroids_limit);
    }

    template<class RowMatrixT>
    void _update_idxs_for_centroids(const RowMatrixT& queries,
        CentroidIndex centroids_limit=-1)
    {
        centroids_limit = _clamp_centroid_limit(centroids_limit);
        _order.resize(centroids_limit);
		auto k = _num_centroids;

        for (int kk = 0; kk < k; kk++) {
            _idxs_for_centroids[kk].clear();
        }

        auto dists = dist::squared_dists_to_vectors<Eigen::ColMajor>(
            _centroids, queries);

        for (int j = 0; j < queries.size; j++) { // for each query
            Scalar* dists_col = dists.col(j).data();
            ar::argsort(dists_col, k, _order);
            for (auto idx : _order) { // for each centroid for this query
                _idxs_for_centroids[idx].push_back(j);
            }
        }
    }

    template<class RowMatrixT>
    void _update_query_storate(const vector<Index>& query_idxs,
        const RowMatrixT& queries)
    {
        assert(_queries_storage.cols() == queries.cols());

        auto nrows = query_idxs.size();
        if (nrows > _queries_storage.rows()) {
            _queries_storage.resize(nrows, queries.cols());
        }
        for (int i = 0; i < nrows; i++) {
            _queries_storage.row(i) = queries.row(query_idxs[i]);
        }
    }
};

// ================================================================
// CascadeIdIndex
// ================================================================

// only holds ids of points, not points themselves
template<class T, int Projections=16>
class CascadeIdIndex {
public:
    typedef T Scalar;
    typedef T DistT;
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
        DistT d_max)
    {
        return abandon::radius<Index>(data, query, d_max, rows());
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
// NNIndex
// ================================================================

// #define QUERY_METHOD(NAME, INVOCATION) \
//     template<class VectorT, class... Args> \
//     auto NAME(const VectorT& query, Args&&... args) \
//         -> decltype(INVOCATION (query, std::forward<Args>(args)...)) \
//     { \
//         auto use_query = _preproc.preprocess(query); \
//         return INVOCATION(use_query, std::forward<Args...>(args)...); \
//     }

// #define QUERY_BATCH_METHOD(NAME, INVOCATION) \
//     template<class RowMatrixT, class... Args> \
//     auto NAME(const RowMatrixT& queries, Args&&... args) \
//         -> decltype(INVOCATION (queries, std::forward<Args>(args)...)) \
//     { \
//         auto use_queries = _preproc.preprocess_batch(queries); \
//         return INVOCATION(use_queries, std::forward<Args...>(args)...); \
//     }

// template<class InnerIndex, class Preprocessor=IdentityPreproc>
// class NNIndex {
// private:
//     Preprocessor _preproc;
// 	InnerIndex _index;
// public:
//     using Scalar = typename InnerIndex::Scalar;
//     using Index = typename InnerIndex::Index;

//     template<class RowMatrixT, class... Args>
//     NNIndex(const RowMatrixT& X, Args&&... args):
// 		_preproc{X}, // TODO warns about preproc uninitialized if () not {} ??
//         _index(_preproc.preprocess_data(X), std::forward<Args>(args)...)
//     {}

//     QUERY_METHOD(radius, _index.radius);
//     QUERY_METHOD(onenn, _index.onenn);
//     QUERY_METHOD(knn, _index.knn);
//     QUERY_BATCH_METHOD(radius_batch, _index.radius_batch);
//     QUERY_BATCH_METHOD(onenn_batch, _index.onenn_batch);
//     QUERY_BATCH_METHOD(knn_batch, _index.knn_batch);
// };

    // TODO E2LSH impl
    // TODO Selective Hashing Impl on top of it

} // namespace nn
#endif // __NN_INDEX_HPP

