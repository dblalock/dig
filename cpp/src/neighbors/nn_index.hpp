//
//  nn_index.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __NN_INDEX_HPP
#define __NN_INDEX_HPP

#include <experimental/optional>
#include <memory>
#include "assert.h"

#include "Dense"

#include "cluster.hpp"
#include "nn_search.hpp"
#include "nn_utils.hpp"
#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "flat_store.hpp"
#include "preproc.hpp"

// #include "debug_utils.hpp"

template<class T> using optional = std::experimental::optional<T>; // will be std::optional in c++17
static constexpr auto nullopt = std::experimental::nullopt;
// auto empty() { return nullopt_t(); }

namespace nn {

static const int8_t kAlignBytes = 32;

// ================================================================
// IndexConfig
// ================================================================

// template<class RowMatrixT>
// class IndexConfig {
// public:
// 	virtual const RowMatrixT& get().data    const;
// 	virtual int get().num_clusters          const;
// 	virtual int get().default_search_frac   const;

// 	virtual ~IndexConfig() {};
// };

// class IndexConfig: public IndexConfig<RowMatrixT> {
template<class RowMatrixT>
class IndexConfig {
public:
    using IdT = idx_t;
    using IdVectorT = vector<IdT>;

    const RowMatrixT& data;
    int num_clusters;
    float default_search_frac;
	optional<IdVectorT> ids;

    // ------------------------ ctors
    IndexConfig() = default;
    IndexConfig(const IndexConfig& rhs) = default;
    ~IndexConfig() = default;
	IndexConfig(const RowMatrixT& data_, int num_clusters_=-1,
		        float default_search_frac_=-1, optional<IdVectorT> ids_=nullopt):
		data(data_),
        num_clusters(num_clusters_),
		default_search_frac(default_search_frac_),
        ids(ids_)
	{}

    // ------------------------ methods
    template<class RowMatrixT2>
    IndexConfig<RowMatrixT2> create_child_config(const RowMatrixT2& data) const {
        IndexConfig<RowMatrixT2> ret(data);
        ret.num_clusters = num_clusters;
        ret.default_search_frac = default_search_frac;
        ret.ids = ids;
        return ret;
    }

    // all classes' get() returns an IndexConfig; this way we don't have
    // to write accessors for each member var we add
    // IndexConfig get() const { return IndexConfig(*this); }
    const IndexConfig& get() const { return *this; }
    IndexConfig& get() { return *this; }
};

//template<class IndexConfigT, class RowMatrixT2>
//std::unique_ptr<IndexConfig> create_child_config(const RowMatrixT2& data,
//	vector<idx_t>* ids_=nullptr)
//{
//	return std::make_unique<IndexConfig<RowMatrixT2> >(data, num_clusters,
//        default_search_frac, ids_);
//}

template<class RowMatrixT, int NumLevels=4>
//class HierarchicalIndexConfig: public IndexConfig<RowMatrixT> {
class HierarchicalIndexConfig {
public:
    int level;
    std::array<IndexConfig<RowMatrixT>, NumLevels> configs;

    // ------------------------ ctors
    HierarchicalIndexConfig() = default;
    HierarchicalIndexConfig(const HierarchicalIndexConfig& rhs) = default;
    ~HierarchicalIndexConfig() = default;

    // ------------------------ methods
    template<class RowMatrixT2>
    HierarchicalIndexConfig<RowMatrixT2> create_child_config(
        const RowMatrixT2& data) const
    {
        assert(level + 1 < NumLevels);
        return IndexConfig<RowMatrixT2>{data, level + 1, configs};
    }

    // setter returns a copy for chaining in initializer lists
    HierarchicalIndexConfig set_level(int new_level) {
        assert(new_level < NumLevels);
        HierarchicalIndexConfig cfg(*this);
        cfg.level = new_level;
        return cfg;
    }
    HierarchicalIndexConfig increment_level() { return set_level(level + 1); }

    const IndexConfig<RowMatrixT>& get() const { return configs[level]; }
    IndexConfig<RowMatrixT>& get() { return configs[level]; }
};


// ================================================================
// FlatIdStore / Neighbor postprocessing
// ================================================================

struct IdentityIdStore {
    template<class... Args> IdentityIdStore(Args&&... args) {};
    template<class T> T postprocess(const T& neighbors) { return neighbors; }
};

template<class IdT=idx_t>
class FlatIdStore {
public:
    typedef IdT ID;

    FlatIdStore(const std::vector<ID> ids): _ids(ids) {}
    FlatIdStore(size_t len):
        // _ids(ar::range(static_cast<ID>(0), static_cast<ID>(len)))
        FlatIdStore(ar::range(static_cast<ID>(0), static_cast<ID>(len))) {}
    FlatIdStore(const optional<std::vector<ID> > ids, size_t len):
        FlatIdStore(len)
	{
		if (ids) {
            _ids = *ids;
        }
		// else { PRINT("not using ids cuz they were null"); }
	}

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
        // PRINT_VAR(_ids.size());
        // DEBUGF("%lld (%g) -> %lld", n.idx, n.dist, _ids[n.idx]);
        // PRINT_VAR(n.dist);

		return Neighbor{_ids[n.idx], n.dist};
    }
    std::vector<Neighbor> postprocess(std::vector<Neighbor> neighbors) {
        auto idxs = ar::map([](auto& n) { return n.idx; }, neighbors);
        auto dists = ar::map([](auto& n) { return n.dist; }, neighbors);
        // PRINT_VAR(ar::to_string(idxs));
        // PRINT_VAR(ar::to_string(dists));
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
// IndexBase
// ================================================================

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

    // // ------------------------------------------------ factory funcs

    // template<class ConfigT>
    // static inline Derived construct(const ConfigT& cfg) {
    //     return Derived{cfg.get().data};
    // }

    // ------------------------------------------------ single queries

    template<class VectorT, class... Args>
    vector<Neighbor> radius(const VectorT& query, DistT d_max, Args&&... args)
    {
        auto neighbors = _derived()->_radius(_derived()->preprocess(query),
            d_max, std::forward<Args>(args)...);
        return _derived()->postprocess(neighbors);
    }

    template<class VectorT, class... Args>
    Neighbor onenn(const VectorT& query, Args&&... args) {
        auto neighbors = _derived()->_onenn(_derived()->preprocess(query),
            std::forward<Args>(args)...);
        return _derived()->postprocess(neighbors);
    }

    template<class VectorT, class... Args>
    vector<Neighbor> knn(const VectorT& query, int k, Args&&... args) {
        auto neighbors = _derived()->_knn(_derived()->preprocess(query), k,
            std::forward<Args>(args)...);
        return _derived()->postprocess(neighbors);
    }

    // ------------------------------------------------ batch of queries
    // TODO allow vector of d_max values for each of these

    template<class RowMatrixT, class... Args>
    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
        DistT d_max, Args&&... args)
    {
        const auto& queries_proc = _derived()->preprocess_batch(queries);
        auto neighbors = _derived()->_radius_batch(queries_proc, d_max,
            std::forward<Args>(args)...);
        return _derived()->postprocess(neighbors);
    }

    template<class RowMatrixT, class... Args>
    vector<Neighbor> onenn_batch(const RowMatrixT& queries, Args&&... args) {
        const auto& queries_proc = Derived::preprocess_batch(queries);
        auto neighbors = Derived::_onenn_batch(queries_proc,
            std::forward<Args>(args)...);
        return Derived::postprocess(neighbors);
    }

    template<class RowMatrixT, class... Args>
    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, int k,
        Args&&... args)
    {
        const auto& queries_proc = _derived()->preprocess_batch(queries);
        auto neighbors = _derived()->_knn_batch(queries_proc, k,
            std::forward<Args>(args)...);
        return _derived()->postprocess(neighbors);
    }

    // ------------------------------------------------ return only idxs

    template<class VectorT, class... Args>
    vector<idx_t> radius_idxs(const VectorT& query, DistT d_max, Args&&... args)
    {
        auto neighbors = Derived::radius(query, d_max,
            std::forward<Args>(args)...);
        return ar::map([](const Neighbor& n) {
            return n.idx;
        }, neighbors);
    }

    template<class VectorT, class... Args>
    idx_t onenn_idxs(const VectorT& query, Args&&... args) {
        auto neighbor = Derived::onenn(query, std::forward<Args>(args)...);
        return neighbor.idx;
    }

    template<class VectorT, class... Args>
    vector<idx_t> knn_idxs(const VectorT& query, int k, Args&&... args) {
        auto neighbors = Derived::knn(query, k, std::forward<Args>(args)...);
        return map([](const Neighbor& n) { return static_cast<idx_t>(n.idx); },
                   neighbors);
    }

    // ------------------------------------------------ batch return only idxs
    // TODO allow vector of d_max values for each of these

    template<class RowMatrixT, class... Args>
    vector<vector<idx_t> > radius_batch_idxs(const RowMatrixT& queries,
        DistT d_max, Args&&... args)
    {
        auto neighbors = Derived::radius_batch(queries, d_max, std::forward<Args>(args)...);
        return idxs_from_nested_neighbors(neighbors);
    }

    template<class RowMatrixT, class... Args>
    vector<idx_t> onenn_batch_idxs(const RowMatrixT& queries, Args&&... args) {
        auto neighbors = Derived::onenn_batch(queries,
			std::forward<Args>(args)...);
        return map([](const Neighbor& n) {
            return n.idx;
        }, neighbors);
    }

    template<class RowMatrixT, class... Args>
    vector<vector<idx_t> > knn_batch_idxs(const RowMatrixT& queries, int k,
        Args&&... args)
    {
        auto neighbors = Derived::knn_batch(queries, k, std::forward<Args>(args)...);
        return idxs_from_nested_neighbors(neighbors);
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
            ret.emplace_back(_derived()->_radius(queries.row(j), d_max));
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<vector<Neighbor> > _knn_batch(const RowMatrixT& queries, int k) {
        vector<vector<Neighbor> > ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(_derived()->_knn(queries.row(j), k));
        }
        return ret;
    }

    template<class RowMatrixT>
    vector<Neighbor> _onenn_batch(const RowMatrixT& queries) {
        vector<Neighbor> ret;
        for (idx_t j = 0; j < queries.rows(); j++) {
            ret.emplace_back(_derived()->_onenn(queries.row(j)));
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

	template<class RowMatrixT, REQ_HAS_ATTR(RowMatrixT, rows())>
	explicit L2IndexBrute(const RowMatrixT& data,
					      const optional<std::vector<idx_t> > ids=nullopt):
        PreprocT(data),
        FlatIdStore(ids, data.rows()),
        _data(PreprocT::preprocess_data(data))
    {
		assert(data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
    }

    template<class ConfigT, REQ_HAS_ATTR(ConfigT, get().data)>
    explicit L2IndexBrute(const ConfigT& cfg):
        L2IndexBrute(cfg.get().data, cfg.get().ids) {}
	// template<class RowMatrixT>
 //    explicit L2IndexBrute(const IndexConfig<RowMatrixT>& cfg):
	// 	L2IndexBrute{cfg.get().data} {}

protected:
    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> _radius(const VectorT& query, DistT d_max) {
        return brute::radius(_matrix(), query, d_max, _rowNorms);
    }

    template<class VectorT>
    Neighbor _onenn(const VectorT& query, DistT d_max=kMaxDist) {
        return brute::onenn(_matrix(), query, _rowNorms);
    }

    template<class VectorT>
    vector<Neighbor> _knn(const VectorT& query, int k, DistT d_max=kMaxDist) {
        return brute::knn(_matrix(), query, k, _rowNorms);
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

    template<class RowMatrixT, REQ_HAS_ATTR(RowMatrixT, rows())>
    explicit L2IndexAbandon(const RowMatrixT& data,
                            const optional<std::vector<idx_t> > ids=nullopt):
        PreprocT(data),
		FlatIdStore(ids, data.rows()),
		_data(PreprocT::preprocess_data(data))
    {
        static_assert(RowMatrixT::IsRowMajor, "Data matrix must be row-major");
        assert(_data.cols() ==
			(aligned_length<Scalar, kAlignBytes>(data.cols())) );
    }
    template<class ConfigT, REQ_HAS_ATTR(ConfigT, get().data)>
    explicit L2IndexAbandon(const ConfigT& cfg):
        L2IndexAbandon(cfg.get().data, cfg.get().ids) {}

    // template<class RowMatrixT>
    // explicit L2IndexAbandon(const IndexConfig<RowMatrixT>& cfg):
    //     L2IndexAbandon{cfg.get().data} {}
    // template<class ConfigT, REQ_HAS_ATTR(ConfigT, get().data)>
    // expL2IndexAbandon build(const ConfigT& cfg) {
    //     return L2IndexAbandon{cfg.get().data};
    // }
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
// L2IndexSimple
// ================================================================

// simple index that searches thru points like L2AbandonIndex, but without
// the abandoning; this lets us measure the effect of the abandoning
template<class T, class PreprocT=IdentityPreproc>
class L2IndexSimple:
    public IndexBase<L2IndexSimple<T, PreprocT>, T>,
    public PreprocT,
    public FlatIdStore<> {
    friend class IndexBase<L2IndexSimple<T, PreprocT>, T>;
public:
    typedef T Scalar;
    typedef T DistT;
    typedef idx_t Index;
    typedef Matrix<Scalar, 1, Dynamic, RowMajor> RowVectT;
    typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

    // ------------------------------------------------ ctors

    L2IndexSimple() = default;

    template<class RowMatrixT, REQ_HAS_ATTR(RowMatrixT, rows())>
    explicit L2IndexSimple(const RowMatrixT& data,
                           const optional<std::vector<idx_t> > ids=nullopt):
        PreprocT(data),
        FlatIdStore(ids, data.rows()),
        _data(PreprocT::preprocess_data(data))
    {
        static_assert(RowMatrixT::IsRowMajor, "Data matrix must be row-major");
        // PRINT("L2IndexSimple ids: ")
        // PRINT_VAR(ar::to_string(_ids));
    }
    template<class ConfigT, REQ_HAS_ATTR(ConfigT, get().data)>
    explicit L2IndexSimple(const ConfigT& cfg):
        L2IndexSimple(cfg.get().data, cfg.get().ids) {}

    // template<class RowMatrixT>
    // explicit L2IndexSimple(const IndexConfig<RowMatrixT>& cfg):
    //     L2IndexSimple{cfg.get().data} {}

protected:
    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> _radius(const VectorT& query, DistT d_max) {
        return simple::radius(_data, query, d_max, rows());
        // return abandon::radius(_data, query, d_max, rows());
    }

    template<class VectorT>
    Neighbor _onenn(const VectorT& query, DistT d_max=kMaxDist) {
        return simple::onenn(_data, query, rows());
        // return abandon::onenn(_data, query, d_max, rows());
    }

    template<class VectorT>
    vector<Neighbor> _knn(const VectorT& query, int k, DistT d_max=kMaxDist) {
        return simple::knn(_data, query, k, rows()); // 5ms
        // return abandon::knn(_data, query, k, d_max, rows()); // 12ms...wat?
    }

    // SELF: the problem is that _data.cols() is 16 cuz of alignment, while the
    // query has the original number of cols passed in (which is 10).
    //     -abandon version always needs rows aligned
    //     -simple version (this class) doesnt really need rows

private:
    DynamicRowArray<Scalar, PreprocT::align_bytes> _data;
};

// ================================================================
// KmeansIndex
// ================================================================

template<class RowMatrixT> class IndexConfig; // forward declare this // TODO move defn to top

template<class T, class InnerIndex=L2IndexAbandon<T>,
    class PreprocT=IdentityPreproc>
class L2KmeansIndex:
    public IndexBase<L2KmeansIndex<T, InnerIndex, PreprocT>, T>,
	public PreprocT,
   public IdentityIdStore {
    // public FlatIdStore<> {
    friend class IndexBase<L2KmeansIndex<T, InnerIndex, PreprocT>, T>;
public:
    using Scalar = T;
    using DistT = T;
    using Index = idx_t;
	using CentroidIndex = int32_t;
	using RowMatrixType = RowMatrix<T>;

	L2KmeansIndex() = default;
	L2KmeansIndex(const L2KmeansIndex& rhs) = delete;
	L2KmeansIndex(L2KmeansIndex&& rhs) = delete;

	template<template <class...> class ConfigT, class RowMatrixT_>
    L2KmeansIndex(const ConfigT<RowMatrixT_>& cfg):
        PreprocT(cfg.get().data),
		// FlatIdStore(cfg.get().ids, cfg.get().data.rows()),
        _order(cfg.get().num_clusters),
        _indexes(new InnerIndex[cfg.get().num_clusters]), // TODO InnerIndex better sync up with cfg
        _centroid_dists(cfg.get().num_clusters),
        _idxs_for_centroids(cfg.get().num_clusters),
        _queries_storage(32, PreprocT::preprocess_data(cfg.get().data).cols()), // 32 query capacity
        _num_rows(cfg.get().data.rows()),
        _num_centroids(cfg.get().num_clusters),
        _default_search_frac(cfg.get().default_search_frac)
    {
        using RowMatrixT = typename mat_traits<RowMatrixT_>::RowMatrixT;

        auto X_ = PreprocT::preprocess_data(cfg.get().data);
        auto k = cfg.get().num_clusters;

        auto num_iters = 10; // TODO increase to 16 after prototyping
		auto centroids_assignments = cluster::kmeans(X_, k, num_iters);
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

        // PRINT("kmeans ids: ")
        // PRINT_VAR(ar::to_string(_ids));

        // create an index for each centroid
        RowMatrixT storage(max_len, X_.cols());
        for (int i = 0; i < k; i++) {
            // copy appropriate rows to temp storage so they're contiguous
            auto& idxs = idxs_for_centroids[i];
            auto nrows = idxs.size();
            for (int row_idx = 0; row_idx < nrows; row_idx++) {
                auto idx = idxs[row_idx];
                storage.row(row_idx) = X_.row(idx);
            }
            // create an InnerIndex instance with these rows
            auto inner_mat = storage.topRows(nrows);
            auto inner_cfg = cfg.create_child_config(inner_mat);
            inner_cfg.get().default_search_frac = _default_search_frac;
            inner_cfg.get().ids = cfg.ids ? ar::at_idxs(*cfg.ids, idxs) : idxs;

       //      if (cfg.ids) {
       //          inner_cfg.ids = ar::at_idxs(*cfg.ids, idxs);
    			// auto inner_ids = ar::at_idxs(*cfg.ids, idxs);
       //          // PRINT_VAR(ar::to_string(inner_ids));
       //          inner_cfg.ids = inner_ids;
       //      } else {
       //          inner_cfg.ids = idxs;
       //      }
            // InnerIndex foo(inner_cfg); // TODO rm
            new (&_indexes[i]) InnerIndex(inner_cfg);
//             new (&_indexes[i]) InnerIndex(storage.topRows(nrows));
//            _indexes[i].ids() = idxs; // TODO rm direct set of idxs
        }
    }

    template<class RowMatrixT>
    L2KmeansIndex(const RowMatrixT& X, int k, float default_search_frac=-1):
		L2KmeansIndex(IndexConfig<RowMatrixT>{X, k, default_search_frac}) {}
//         PreprocT(X),
//         _order(k),
//         _indexes(new InnerIndex[k]),
//         _centroid_dists(k),
//         _idxs_for_centroids(k),
//         _queries_storage(32, PreprocT::preprocess_data(X).cols()), // 32 query capacity
//         _num_centroids(k),
//         _default_search_frac(default_search_frac)
//    {}

//    template<class ConfigT>
//    static inline L2KmeansIndex construct(const ConfigT& cfg) {
//        return L2KmeansIndex{cfg.get().data, cfg.get().num_clusters,};
//    }

    // ------------------------------------------------ accessors

    idx_t rows() const { return _num_rows; }

 //    CentroidIndex _num_centroids const {
	// 	return static_cast<CentroidIndex>(_centroids.size());
	// }

    // ------------------------------------------------ single query

protected:
    template<class VectorT>
    vector<Neighbor> _radius(const VectorT& query, DistT d_max,
        float search_frac=-1)
    {
        _update_order_for_query(query, search_frac); // update _order

        vector<Neighbor> ret;
        for (int i = 0; i < _order.size(); i++) {
            auto& index = _indexes[_order[i]];
            if (index.rows() < 1) { continue; }
            auto neighbors = index.radius(query, d_max);

            ar::concat_inplace(ret, neighbors);
        }
        sort_neighbors_ascending_idx(ret);
        return ret;
    }

    template<class VectorT>
    Neighbor _onenn(const VectorT& query, DistT d_max=kMaxDist,
        float search_frac=-1)
    {
        _update_order_for_query(query, search_frac); // update _order

        Neighbor ret{kInvalidIdx, d_max};
        for (int i = 0; i < _order.size(); i++) {
            auto& index = _indexes[_order[i]];
            if (index.rows() < 1) { continue; }

            Neighbor n = index.onenn(query);
            if (n.dist < ret.dist) {
                ret = n;
            }
        }
        return ret;
    }

    template<class VectorT>
    vector<Neighbor> _knn(const VectorT& query, int k, DistT d_max=kMaxDist,
        float search_frac=-1)
    {
        _update_order_for_query(query, search_frac); // update _order

        // PRINT_VAR(_order.size());

        vector<Neighbor> ret(k, Neighbor{kInvalidIdx, d_max});
        for (int i = 0; i < _order.size(); i++) {
            auto& index = _indexes[_order[i]];
            if (index.rows() < 1) continue;

            auto neighbors = index.knn(query, k);

            d_max = maybe_insert_neighbors(ret, neighbors);
        }
		return ret;
    }

    // ------------------------------------------------ batch of queries

    template<class RowMatrixT>
    vector<vector<Neighbor> > _radius_batch(const RowMatrixT& queries,
        DistT d_max, float search_frac=-1)
    {
        _update_idxs_for_centroids(queries, search_frac);
        vector<vector<Neighbor> > ret(queries.rows());

        for (int kk = 0; kk < _num_centroids; kk++) { // for each centroid
            auto& query_idxs = _idxs_for_centroids[kk];
            auto nrows = query_idxs.size();
            _update_query_storage(query_idxs, queries);

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
        float search_frac=-1)
    {
        _update_idxs_for_centroids(queries, search_frac);
        vector<Neighbor> ret(queries.rows());
        for (int kk = 0; kk < _num_centroids; kk++) {
            auto& query_idxs = _idxs_for_centroids[kk];
            auto nrows = query_idxs.size();
            _update_query_storage(query_idxs, queries);

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
        float search_frac=-1)
    {
        _update_idxs_for_centroids(queries, search_frac);
        vector<vector<Neighbor> > ret(queries.rows());

        for (int kk = 0; kk < _num_centroids; kk++) { // for each centroid
            auto& query_idxs = _idxs_for_centroids[kk];
            auto nrows = query_idxs.size();
            _update_query_storage(query_idxs, queries);

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
    // the fields below are storage to avoid doing allocs for each query
    // XXX: shared storage makes this class not remotely thread safe
    // std::vector<CentroidIndex> _order;
    std::vector<CentroidIndex> _order;
    std::unique_ptr<InnerIndex[]> _indexes;
    ColVector<Scalar> _centroid_dists;
    std::vector<std::vector<CentroidIndex> > _idxs_for_centroids;
    ColMatrix<Scalar> _queries_storage;
    idx_t _num_rows;
    CentroidIndex _num_centroids;
    float _default_search_frac;

    CentroidIndex _clamp_centroid_limit(CentroidIndex centroids_limit) {
        if (centroids_limit < 1) {
            int default_num_centroids = static_cast<CentroidIndex>(
                _default_search_frac * _num_centroids);
            if (default_num_centroids > 0) {
                return default_num_centroids;
            }
            return _num_centroids;
        }
        return ar::min(_num_centroids, centroids_limit);
    }
    CentroidIndex _clamp_centroid_limit(float search_frac) {
        // if search_frac is small and positive, don't round down to 0
        auto limit = search_frac * _num_centroids;
        if (search_frac > 0 && limit == 0) { return 1; }
        // otherwise, just use that fraction of the centroids and clamp
        return _clamp_centroid_limit(static_cast<CentroidIndex>(limit));
    }

    template<class VectorT>
    void _update_order_for_query(const VectorT& query, float search_frac=-1) {
        _centroid_dists = dist::squared_dists_to_vector(_centroids, query);
        ar::argsort(_centroid_dists.data(), _num_centroids, _order.data());
        auto centroids_limit = _clamp_centroid_limit(search_frac);
        _order.resize(centroids_limit);
    }

    template<class RowMatrixT>
    void _update_idxs_for_centroids(const RowMatrixT& queries,
        float search_frac=-1)
    {
        auto centroids_limit = _clamp_centroid_limit(search_frac);
        _order.resize(centroids_limit);
		auto k = _num_centroids;

        for (int kk = 0; kk < k; kk++) {
            _idxs_for_centroids[kk].clear();
        }

        auto dists = dist::squared_dists_to_vectors<Eigen::ColMajor>(
            _centroids, queries);

        for (int j = 0; j < queries.size(); j++) { // for each query
            Scalar* dists_col = dists.col(j).data();
            ar::argsort(dists_col, k, _order.data());
            for (auto idx : _order) { // for each centroid for this query
                _idxs_for_centroids[idx].push_back(j);
            }
        }
    }

    template<class RowMatrixT, class idx_t=Index>
    void _update_query_storage(const vector<idx_t>& query_idxs,
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


// // ================================================================
// // Traits / factory func
// // ================================================================

// enum IndexKeys {
//     L2IndexBruteK,
//     L2IndexAbandonK,
//     L2IndexSimpleK,
//     L2KmeansIndexK // TODO make the class L2IndexKmeans for consistency
// };

// template<int IndexK, class ScalarT, class PreprocT=IdentityPreproc>
// struct index_traits { };

// template<class ScalarT, class PreprocT>
// struct index_traits<L2IndexBruteK, ScalarT, PreprocT> {
//     using Scalar = ScalarT;
//     using Preproc = PreprocT;
//     using IndexT = L2IndexBrute<Scalar, PreprocT>;
// };

// template<class ScalarT, class PreprocT>
// struct index_traits<L2KmeansIndexK, ScalarT, PreprocT> {
//     using Scalar = ScalarT;
//     using Preproc = PreprocT;
//     static const int InnerIndexK = L2IndexSimpleK;
// 	using KmeansInnerIndexT = typename index_traits<InnerIndexK, Scalar>::IndexT;
//     using KmeansIndexT = L2KmeansIndex<Scalar, KmeansInnerIndexT>;
//     using IndexT = L2KmeansIndex<Scalar, PreprocT>;
// };

// template<int IndexK, class ScalarT,
//     class PreprocT=typename index_traits<IndexK, ScalarT>::Preproc,
//     class ConfigT=void>
// auto build_index(const ConfigT& cfg)
// 	-> typename index_traits<IndexK, ScalarT, PreprocT>::IndexT
// {
// 	using IndexT = typename index_traits<IndexK, ScalarT, PreprocT>::IndexT;
// 	return IndexT(cfg);
// }

} // namespace nn
#endif // __NN_INDEX_HPP

