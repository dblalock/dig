//
//  nn_index.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __FLAT_STORE_HPP
#define __FLAT_STORE_HPP

#include <memory>
#include "assert.h"

#include "Dense"

#include "array_utils.hpp"
#include "memory.hpp"
#include "nn_utils.hpp"

namespace nn {

// ================================================================
// RowArray
// ================================================================

// ------------------------------------------------ traits

// template<class T, int AlignBytes> class DynamicRowArray;
// template<class T, int NumCols, int AlignBytes> class FixedRowArray;
struct DynamicRowArrayKey;
struct FixedRowArrayKey;

template<class TypeKey, class T, int NumCols=-2>
struct row_array_traits {};

template<class T, int NumCols>
struct row_array_traits<DynamicRowArrayKey, T, NumCols> {
	typedef idx_t RowIndex;
    typedef int32_t ColIndex;
    typedef Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> RowT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
};

template<class T, int NumCols>
struct row_array_traits<FixedRowArrayKey, T, NumCols> {
	typedef idx_t RowIndex;
    typedef int32_t ColIndex;
    typedef Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> RowT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
};

// ------------------------------------------------ BaseRowArray

// BaseRowArray just has a pointer to its data and knows how wide each row
// is by calling Derived::cols(). It can hand a pointer to the start of a
// row, insert into a row, and copy one of its rows into another one; it
// performs no bounds checks of any kind (and doesn't even know how long
// its data is)
//
// Also aligns each row to AlignBytes boundaries, assuming that Derived::cols()
// is guaranteed to yield a number of objects that's an even multiple of
// AlignBytes / sizeof(T)
template<class Derived, class T, class DerivedKey, int AlignBytes=kDefaultAlignBytes,
    bool OwnPtr=true>
class BaseRowArray {
public:
	// typedef SELF_TYPE SelfT;
	// using SelfT = std::remove_reference<decltype(*this)>::type;
	typedef T Scalar;
    // typedef idx_t Index;
    typedef row_array_traits<DerivedKey, T> DerivedTraits;
    // typedef typename row_array_traits<DerivedKey, T>::MatrixT MatrixT;
    // typedef typename row_array_traits<DerivedKey, T>::RowT RowT;
	typedef typename DerivedTraits::RowIndex RowIndex;
    typedef typename DerivedTraits::ColIndex ColIndex;
    typedef typename DerivedTraits::MatrixT MatrixT;
    typedef typename DerivedTraits::RowT RowT;
    // typedef Eigen::Map<MatrixT, _AlignHelper<AlignBytes>::AlignmentType> MatrixMapT;
	typedef Eigen::Map<MatrixT, Eigen::Aligned> MatrixMapT; // always align whole mat
    typedef Eigen::Map<RowT, _AlignHelper<AlignBytes>::AlignmentType> RowMapT;


    static constexpr bool IsRowMajor = true;

    // ------------------------ ctors

    // init data with new pointer that we own
    template<METHOD_REQ(OwnPtr)>
    BaseRowArray(size_t initial_rows, size_t ncols):
        _data(_aligned_alloc(initial_rows * _aligned_length(ncols)))
    {
        assert(initial_rows >= 0);
    }

    BaseRowArray() = default;
	// move ctor
    // BaseRowArray(BaseRowArray&& rhs) noexcept : _data(std::move(rhs._data)) {
    //     rhs._data = nullptr;
    // }


	// template<class RowMatrixT>
	template<class RowMatrixT, REQUIRE_IS_NOT_A(BaseRowArray, RowMatrixT)>
    BaseRowArray(const RowMatrixT& X):
        BaseRowArray(X.rows(), X.cols())
    {
		static_assert(RowMatrixT::IsRowMajor, "Data matrix X must be row-major");
        assert(X.IsRowMajor);
        auto ncols = _aligned_length(X.cols());
        MatrixMapT mat(_data, X.rows(), ncols);
        mat.topLeftCorner(X.rows(), X.cols()) = X;
    }

    // init data ptr directly with existing pointer that we don't own
    template<class RowMatrixT, METHOD_REQ(!OwnPtr)>
    BaseRowArray(const RowMatrixT& X):
        _data(X.data())
    {}
	template<METHOD_REQ(!OwnPtr)>
	BaseRowArray(const Scalar* ptr):
		_data(ptr)
	{}

    ~BaseRowArray() {
        if (OwnPtr && _data) {
            aligned_free<Scalar, AlignBytes>(_data);
        }
    }

    // ------------------------ accessors

    Scalar* data() const { return _data; }

    // template<class Index>
    Scalar* row_ptr(const RowIndex i) const {
        assert(i >= 0);
        assert(_data);
		return _data + (i * _cols());
    }
    // template<class Index>
    // const Scalar* row_ptr(const Index i) const { return row_ptr(i); }

    // template<class Index>
    RowMapT row(const RowIndex i) const {
        return RowMapT{row_ptr(i), _cols()};
    }

    // template<class Index>
    MatrixMapT matrix(const RowIndex nrows) const {
        assert(_data);
        return MatrixMapT{_data, nrows, _cols()};
    }

    // ------------------------ resize, insert, delete
    // template<class Index>
    void resize(RowIndex old_num_rows, RowIndex new_num_rows) {
        auto ncols = _cols();
        assert (_aligned_length(ncols) == ncols);
        assert(new_num_rows > old_num_rows); // for now, should only get bigger
        if (_data == nullptr) { assert(old_num_rows == 0); }

        int64_t old_capacity = old_num_rows * ncols;
        int64_t new_capacity = new_num_rows * ncols;
        int64_t len = std::min(old_capacity, new_capacity);
        Scalar* new_data = _aligned_alloc(new_capacity);
        if (_data) {
            std::copy(_data, _data + len, new_data);
            aligned_free(_data);
        }
        _data = new_data;
    }

    // template<class Index, bool NeedsZeroing=true>
    template<bool NeedsZeroing=true>
    void insert(const Scalar* x, ColIndex len, RowIndex at_idx) {
        auto row_start = row_ptr(at_idx);
        // zero past end of inserted data (and possibly a bit before)
        if (AlignBytes > 0 && NeedsZeroing) {
            auto ncols = _cols();
            assert(AlignBytes > (sizeof(T) * ncols));
            auto row_end = row_start + ncols;
            auto zero_start = row_end - (AlignBytes / sizeof(T));
            for (auto ptr = zero_start; ptr < row_end; ++ptr) {
                *ptr = 0;
            }
        }
        std::copy(x, x + len, row_start);
    }

    void copy_row(RowIndex from_idx, RowIndex to_idx) {
        assert(from_idx != to_idx);
		insert<false>(row_ptr(from_idx), _cols(), to_idx);
    }

protected:
    Scalar* _aligned_alloc(size_t n) {
        // note: always use aligned alloc here, even if rows don't need to
        // be aligned
        return aligned_alloc<Scalar, kDefaultAlignBytes>(n);
    }
    ColIndex _aligned_length(size_t ncols) {
        return static_cast<ColIndex>(aligned_length<T, AlignBytes>(ncols));
    }

private:
    Scalar* _data;

    inline const Derived* _derived() const { return static_cast<const Derived*>(this); }
    // inline auto _cols() -> decltype(Derived::cols()) { return Derived::cols(); }
    // inline ColIndex _cols() const { return Derived::cols(); }
    inline ColIndex _cols() const { return _derived()->cols(); }
};

// ------------------------------------------------ DynamicRowArray

// BaseRowArray subclass with a width set at construction time
template<class T, int AlignBytes=kDefaultAlignBytes, bool OwnPtr=true>
class DynamicRowArray :
	public BaseRowArray<DynamicRowArray<T, AlignBytes, OwnPtr>,
		T, DynamicRowArrayKey, AlignBytes, OwnPtr> {
public:
    // typedef int32_t ColIndex;
    typedef typename row_array_traits<DynamicRowArrayKey, T>::ColIndex ColIndex;
	typedef BaseRowArray<DynamicRowArray<T, AlignBytes, OwnPtr>, T, DynamicRowArrayKey, AlignBytes, OwnPtr> Super;

    // ------------------------ ctors
    DynamicRowArray() = default;

    DynamicRowArray(size_t initial_rows, size_t ncols):
		Super(initial_rows, Super::_aligned_length(ncols)),
		_ncols(Super::_aligned_length(ncols))
    {
        assert(initial_rows >= 0);
    }

    template<class RowMatrixT>
    DynamicRowArray(const RowMatrixT& X):
        Super(X),
		_ncols(Super::_aligned_length(X.cols()))
    {}

    // no copying, only moving
    DynamicRowArray(const DynamicRowArray& other) = delete;
    // DynamicRowArray(const DynamicRowArray&& other) = delete;
    DynamicRowArray(DynamicRowArray&& rhs) noexcept:
        Super(std::move(rhs)),
        _ncols(rhs._ncols)
    {}

    // ------------------------ accessors
    ColIndex cols() const { assert(_ncols > 0); return _ncols; }

private:
    ColIndex _ncols;
};

// ------------------------------------------------ FixedRowArray

// BaseRowArray subclass that has a fixed width, and so doesn't need to
// store it
template<class T, int NumCols, int AlignBytes=kDefaultAlignBytes, bool OwnPtr=true>
class FixedRowArray :
	public BaseRowArray<FixedRowArray<T, NumCols, AlignBytes, OwnPtr>,
		T, FixedRowArrayKey, AlignBytes, OwnPtr> {
public:
	typedef T Scalar;
	typedef row_array_traits<FixedRowArrayKey, T> Traits;
	typedef typename Traits::RowIndex RowIndex;
	typedef typename Traits::ColIndex ColIndex;
    typedef BaseRowArray<FixedRowArray<T, NumCols, AlignBytes, OwnPtr>,
		T, FixedRowArrayKey, AlignBytes, OwnPtr> Super;
    enum { Cols = NumCols };

    static_assert(NumCols > 0, "NumCols must be > 0");

    // ------------------------ ctors

    FixedRowArray() = default;

    FixedRowArray(RowIndex initial_rows):
        Super(initial_rows, NumCols)
    {
        // NOTE: wouldn't need this restriction if we had BaseRowArray
        // always align what it got from derived()->cols()
        static_assert(NumCols == (aligned_length<T, AlignBytes>(NumCols)),
            "Specified NumCols will not yield aligned rows!");
    }

    FixedRowArray(const FixedRowArray& other) = delete;
    // FixedRowArray(const FixedRowArray&& other) = delete;
    FixedRowArray(FixedRowArray&& rhs) noexcept: Super(std::move(rhs)) {}

    // ------------------------ accessors
    inline constexpr ColIndex cols() const { return NumCols; }

    // ------------------------ insert
    template<bool NeedsZeroing=true>
    void insert(const Scalar* x, RowIndex at_idx) {
        Super::insert(x, NumCols, at_idx);
    }
};

// ================================================================
// RowStore
// ================================================================
// This code is unused but was working

// stores vectors as (optionally) aligned rows in a matrix; lets us access a
// vector and know that it's 32-byte aligned, which enables using
// 32-byte SIMD instructions (as found in 256-bit AVX 2)
//
// This code is somewhat redundant with RowArray code, but refactoring
// to combine them is not a priority atm. Also, I don't think anything is
// actually using this code anymore.
template<class T, int AlignBytes=kDefaultAlignBytes, class IdT=int64_t>
class RowStore {
public:
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index Index;
    typedef T Scalar;
    typedef IdT Id;
    typedef int8_t PadLengthT;
    typedef Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
        _AlignHelper<AlignBytes>::AlignmentType> RowT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixT;

    static constexpr bool IsRowMajor = true;

    // ------------------------ ctors

    RowStore(Index capacity, Index ncols):
        _pad_width(static_cast<PadLengthT>(aligned_length<Scalar, AlignBytes>(ncols) - ncols)),
        _data(capacity, aligned_length<Scalar, AlignBytes>(ncols))
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

    // aligned Eigen::Map to get explicit vectorization
    RowT row(Index i) const {
        RowT r(row_ptr(i), cols());
        return r;
    }

    auto matrix() -> decltype(std::declval<RowMatrixT>().topLeftCorner(2, 2)) {
        return _data.topLeftCorner(rows(), cols());
    }
    auto inner_matrix() -> decltype(std::declval<RowMatrixT>().topLeftCorner(2, 2)) {
        return _data.topLeftCorner(rows(), cols() - _pad_width);
    }

    // ------------------------ insert and delete

    // insert returns 0 if no resize, else new capacity
    Index insert(const Scalar* row_start, Id id) {
        _ids.push_back(id);
        auto cap = capacity_rows();
        Index ret = 0;
        if (rows() > cap) { // resize if needed
            Index new_capacity = std::max(cap + 1, cap * 1.5);
            _data.conservativeResize(new_capacity, Eigen::NoChange);
            ret = new_capacity;
        }
        Scalar* row_start_ptr = row_ptr(rows());
        std::copy(row_start, row_start + (cols() - _pad_width), row_start_ptr);
        if (AlignBytes > 0 && _pad_width > 0) { // check alignbytes to short-circuit
            for (Index i = 0; i < _pad_width; i++) {
                *(row_start_ptr + cols() + i) = 0;
            }
        }
        return ret;
    }

    Index erase(Id id) {
        auto idx = ar::find(_ids, id);
        if (idx < 0) {
            return -1;
        }
        _ids[idx] = _ids.back();
        _ids.pop_back(); // reduce size
        return idx;
    }

private:
    std::vector<Id> _ids; // 24B; size() = number of entries
    RowMatrixT _data; // 24B; rows, cols = capacity, vector length
    PadLengthT _pad_width; // 1B
};

} // namespace nn

#endif // __FLAT_STORE_HPP
