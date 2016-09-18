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

namespace nn {

// ================================================================
// RowStore
// ================================================================

template<int AlignBytes, class IntT>
inline IntT aligned_length(IntT ncols) {
    int16_t align_elements = AlignBytes / sizeof(IntT);
    int16_t remainder = ncols % align_elements;
    if (remainder > 0) {
        ncols += align_elements - remainder;
    }
    return ncols;
}



// helper struct to get Map<> MapOptions based on alignment in bytes
template<int AlignBytes>
struct _AlignHelper { enum { AlignmentType = Eigen::Unaligned }; };
template<> struct _AlignHelper<16> { enum { AlignmentType = Eigen::Aligned }; };
template<> struct _AlignHelper<32> { enum { AlignmentType = Eigen::Aligned }; };

// stores vectors as (optionally) aligned rows in a matrix; lets us access a
// vector and know that it's 32-byte aligned, which enables using
// 32-byte SIMD instructions (as found in 256-bit align for AVX 2)
template<class T, int AlignBytes=32, class IdT=int64_t>
class RowStore {
public:
	typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index Index;
    typedef T Scalar;
    typedef IdT Id;
    typedef int8_t PadLengthT;
    // typedef Eigen::Map<Eigen::Matrix<Scalar, 1, -1>, Eigen::Aligned> RowT;
    typedef Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
        _AlignHelper<AlignBytes>::AlignmentType> RowT;
        // Eigen::Stride<Eigen::Dynamic, 1> > RowT;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixT;
    // typedef Eigen::Map<RowMatrixT> DataMatrixT;

    static const bool IsRowMajor = true;

    // ------------------------ ctors

    RowStore(Index capacity, Index ncols):
        _pad_width(static_cast<PadLengthT>(aligned_length<AlignBytes>(ncols) - ncols)),
        _data(capacity, aligned_length<AlignBytes>(ncols))
//        _ncols(aligned_length<AlignBytes>(ncols)),
        // _capacity(capacity),
//        _data(new Scalar[capacity * _ncols]())
	{
		assert(capacity > 0);
		if (_pad_width > 0) {
			_data.topRightCorner(_data.rows(), _pad_width).setZero();
		}
	}

	RowStore(const RowMatrixT& X):
		RowStore(X.rows(), X.cols())
	{
		_data.topLeftCorner(X.rows(), X.cols()) = X;
		_ids = ar::range(static_cast<Id>(0), static_cast<Id>(X.rows()));
	}

    // template<class RowMatrixT>
    // RowStore(const RowMatrixT&& X) noexcept : _data(X) {}
        // _ncols(aligned_length<AlignBytes>(X.cols())),
        // _capacity(X.rows()),
        // _data(new Scalar[_capacity * _ncols]())
    // {
        // assert(_capacity > 0);
        // for (size_t i = 0; i < X.rows(); i++) {
        //     insert(X.row(i).data(), i);
        // }
    // }

    // ------------------------ accessors
    Index rows() const { return _ids.size(); }
    Index cols() const { return _data.cols(); }
    Index size() const { return rows() * cols(); }
	Index capacity_rows() const { return _data.rows(); }
	Index capacity() const { return capacity_rows() * cols(); }
	Index padding() const { return _pad_width; }

    const std::vector<Id>& row_ids() const { return _ids; }

    Scalar* row_ptr(Index i) { return _data.row(i).data(); }
    const Scalar* row_ptr(Index i) const { return _data.row(i).data(); }

    // aligned Eigen::Map to get explicit vectorization; this should
    // be const, but const and maps are tricky to use together
    RowT row(Index i) const {
		RowT r(row_ptr(i), cols());
		return r;
//		return Eigen::Map<Eigen::Matrix<Scalar, 1, -1, Eigen::RowMajor>, Eigen::Aligned>(row_ptr(i));
    }

    auto matrix() -> decltype(std::declval<RowMatrixT>().topLeftCorner(2, 2)) {
        return _data.topLeftCorner(rows(), cols());
    }
	auto inner_matrix() -> decltype(std::declval<RowMatrixT>().topLeftCorner(2, 2)) {
        return _data.topLeftCorner(rows(), cols() - _pad_width);
    }


    // ------------------------ insert and delete
    void insert(const Scalar* row_start, Id id) {
        _ids.push_back(id);
		auto cap = capacity_rows();
        if (rows() > cap) { // resize if needed
            // Index new_size = std::max(capacity + 1, _capacity * 1.5) * _ncols;
            // std::unique_ptr<Scalar[]> new_data(new Scalar[new_capacity]);
            // std::copy(_data, _data + size(), new_data.get());
            // _capacity = new_capacity;
            // _data.swap(new_data);
            Index new_capacity = std::max(cap + 1, cap * 1.5);
            _data.conservativeResize(new_capacity, Eigen::NoChange);
        }
//		_data.row(rows()) =
//        Scalar* row_start_ptr = _data + (_ncols * rows());
		Scalar* row_start_ptr = row_ptr(rows());
        std::copy(row_start, row_start + (cols() - _pad_width), row_start_ptr);
		if (AlignBytes > 0 && _pad_width > 0) { // check alignbytes to short-circuit
			for (Index i = 0; i < _pad_width; i++) {
				*(row_start_ptr + cols() + i) = 0;
			}
		}
    }

    bool erase(Id id) {
        auto idx = ar::find(_ids, id);
        if (idx < 0) {
            return false;
        }
        // move data at the end to overwrite the removed row so that
        // everything remains contiguous
        // auto last_idx = _ids.size() - 1;
        // _ids[idx] = _ids[last_idx];
        // auto last_row_start = row_ptr(last_idx);
        // Scalar* replace_row_ptr = row_ptr(idx);
        // std::copy(last_row_start, last_row_start + _ncols, replace_row_ptr);

        _ids[idx] = _ids.back();
        _ids.pop_back(); // reduce size
        return true;
    }

private:
    // int32_t _ncols; // 4B
    // int32_t _capacity; // 4B
    std::vector<Id> _ids; // 24B; size() = number of entries
    RowMatrixT _data; // 24B; rows, cols = capacity, vector length
    PadLengthT _pad_width; // 1B
    // std::unique_ptr<Scalar[]> _data; // 8B
};

// ================================================================
// IndexBase
// ================================================================

// superclass using Curiously Recurring Template Pattern to get compile-time
// polymorphism; i.e., we can call arbitrary child class functions without
// having to declare them anywhere in this class. This is the same pattern
// used by Eigen.
template<class Derived, class RowMatrixT, class dist_t=float>
class IndexBase {
public:

    // ------------------------------------------------ insert and erase

	template<class Id>
	void insert(const typename RowMatrixT::Scalar* row_start, Id id) {
		Derived::_data.insert(row_start, id);
    }

	template<class Id>
	bool erase(Id id) { return Derived::_data.erase(id); }

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

// ================================================================
// L2IndexBrute
// ================================================================

// answers neighbor queries using matmuls
template<class RowMatrixT, class dist_t=float>
class L2IndexBrute: public IndexBase<L2IndexBrute<RowMatrixT, dist_t>, RowMatrixT, dist_t> {
public:
    typedef typename RowMatrixT::Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, 1> ColVectorT;
//	typedef Eigen::Map<Matrix<typename RowMatrixT::Scalar, Dynamic, Dynamic, RowMajor> > DataMatrixT;

    explicit L2IndexBrute(const RowMatrixT& data):
        _data(data)
    {
		assert(data.IsRowMajor);
        _rowNorms = data.rowwise().squaredNorm();
    }

//    DataMatrixT data const {
//        return Map<Matrix<typename RowMatrixT::Scalar, Dynamic, Dynamic, RowMajor> >(_data)
//    }

    // ------------------------------------------------ single query

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query, dist_t radius_sq) {
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

    vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
        dist_t radius_sq)
    {
        return brute::radius_batch(_data.matrix(), queries, radius_sq, _rowNorms);
    }

    vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, int k) {
        return brute::knn_batch(_data.matrix(), queries, k, _rowNorms);
    }

    vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
        return brute::onenn_batch(_data.matrix(), queries, _rowNorms);
    }

private:
    // RowMatrixT _data;
    RowStore<typename RowMatrixT::Scalar, 0> _data;
    ColVectorT _rowNorms;
};

// ================================================================
// L2IndexAbandon
// ================================================================

template<class RowMatrixT, class dist_t=float>
class L2IndexAbandon: public IndexBase<L2IndexAbandon<RowMatrixT>, RowMatrixT> {
public:
    typedef typename RowMatrixT::Scalar Scalar;
    typedef Matrix<Scalar, 1, Dynamic, RowMajor> RowVectT;
    typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

    explicit L2IndexAbandon(const RowMatrixT& data):
        _data(data)
    {
        assert(_data.IsRowMajor);
        _colMeans = data.colwise().mean();
        // _orderIdxs = RowVectIdxsT(data.cols());
    }

    template<class VectorT>
    vector<Neighbor> radius(const VectorT& query, dist_t radius_sq) {
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
    RowStore<typename RowMatrixT::Scalar> _data;
    RowVectT _colMeans;
    // RowVectIdxsT _orderIdxs;
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

