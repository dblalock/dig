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

// #include "nn_search.hpp"
#include "array_utils.hpp"

namespace nn {

// ================================================================
// Utils
// ================================================================

template<class T, int AlignBytes, class IntT>
inline IntT aligned_length(IntT ncols) {
    int16_t align_elements = AlignBytes / sizeof(T);
    int16_t remainder = ncols % align_elements;
    if (remainder > 0) {
        ncols += align_elements - remainder;
    }
    return ncols;
}

// helper struct to get Eigen::Map<> MapOptions based on alignment in bytes
template<int AlignBytes>
struct _AlignHelper { enum { AlignmentType = Eigen::Unaligned }; };
template<> struct _AlignHelper<16> { enum { AlignmentType = Eigen::Aligned }; };
template<> struct _AlignHelper<32> { enum { AlignmentType = Eigen::Aligned }; };

template<class T, int AlignBytes>
static inline T* aligned_alloc(size_t n) {
    static_assert(AlignBytes == 0 || AlignBytes == 32,
				  "Only AlignBytes values of 0 and 32 are supported!");
    if (AlignBytes == 32) {
        return Eigen::aligned_allocator<T>{}.allocate(n);
    } else {
        return new T[n];
    }
}
template<class T, int AlignBytes>
static inline void aligned_free(T* p) {
    static_assert(AlignBytes == 0 || AlignBytes == 32,
				  "Only AlignBytes values of 0 and 32 are supported!");
    if (AlignBytes == 32) {
        Eigen::aligned_allocator<T>{}.deallocate(p, 0); // 0 unused
    } else {
        delete[] p;
    }
}

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

// struct row_array_traits<DynamicRowArray<T, AlignBytes>, T, NumCols, AlignBytes> {
//     typedef int32_t ColIndex;
//     typedef Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> RowT;
//     typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
// };

// template<template<class..., int...> class Derived, class T=char, int NumCols=-2, int... ints>
// struct row_array_traits {};

// template<template<class..., int...> class Derived, class T, int NumCols, int... ints>
// struct row_array_traits<DynamicRowArray, T, NumCols> {
//     typedef int32_t ColIndex;
//     typedef Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> RowT;
//     typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
// };

// // set default param values here so don't have to pass them in
// template<class Derived, class T=char, int NumCols=-2, int AlignBytes=-2>
// struct row_array_traits {};

// // template <class T, int NumCols, int AlignBytes, class... Args>
// // struct row_array_traits<DynamicRowArray<Args...>, T, NumCols, AlignBytes> {
// template<class T, int NumCols, int AlignBytes>
// struct row_array_traits<DynamicRowArray<T, AlignBytes>, T, NumCols, AlignBytes> {
//     typedef int32_t ColIndex;
//     typedef Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> RowT;
//     typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
// };

// // template <class T, int NumCols, int AlignBytes, class... Args>
// // struct row_array_traits<FixedRowArray<T, NumCols, AlignBytes>, T, NumCols, AlignBytes, Args...> {
// template<class T, int NumCols, int AlignBytes>
// struct row_array_traits<FixedRowArray<T, NumCols, AlignBytes>, T, NumCols, AlignBytes> {
//     typedef int32_t ColIndex;
//     typedef Eigen::Matrix<T, 1, NumCols, Eigen::RowMajor> RowT;
//     typedef Eigen::Matrix<T, Eigen::Dynamic, NumCols, Eigen::RowMajor> MatrixT;
// };

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
template<class Derived, class T, class DerivedKey, int AlignBytes=32>
class BaseRowArray {
public:
	typedef T Scalar;
    // typedef idx_t Index;
    typedef row_array_traits<DerivedKey, T> DerivedTraits;
    // typedef typename row_array_traits<DerivedKey, T>::MatrixT MatrixT;
    // typedef typename row_array_traits<DerivedKey, T>::RowT RowT;
	typedef typename DerivedTraits::RowIndex RowIndex;
    typedef typename DerivedTraits::ColIndex ColIndex;
    typedef typename DerivedTraits::MatrixT MatrixT;
    typedef typename DerivedTraits::RowT RowT;
	typedef Eigen::Map<MatrixT, _AlignHelper<AlignBytes>::AlignmentType> MatrixMapT;
    typedef Eigen::Map<RowT, _AlignHelper<AlignBytes>::AlignmentType> RowMapT;

    static const bool IsRowMajor = true;

    // ------------------------ ctors
    BaseRowArray(size_t initial_rows, size_t ncols):
        _data(_aligned_alloc(initial_rows * _aligned_length(ncols)))
    {
        assert(initial_rows >= 0);
    }

    template<class RowMatrixT>
    BaseRowArray(const RowMatrixT& X):
        BaseRowArray(X.rows(), X.cols())
    {
        assert(X.IsRowMajor);
        auto ncols = _aligned_length(X.cols());
        MatrixMapT mat(_data, X.rows(), ncols);
        mat.topLeftCorner(X.rows(), X.cols()) = X;
    }

    ~BaseRowArray() { aligned_free<Scalar, AlignBytes>(_data); }

    // ------------------------ accessors

    Scalar* data() const { return _data; }

    // template<class Index>
    Scalar* row_ptr(const RowIndex i) const {
        assert(i >= 0);
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
        return MatrixMapT{_data, nrows, _cols()};
    }

    // ------------------------ resize, insert, delete
    // template<class Index>
    void resize(RowIndex old_num_rows, RowIndex new_num_rows) {
        auto ncols = _cols();
        assert (_aligned_length(ncols) == ncols);
        assert(new_num_rows > old_num_rows); // for now, should only get bigger

        int64_t old_capacity = old_num_rows * ncols;
        int64_t new_capacity = new_num_rows * ncols;
        int64_t len = std::min(old_capacity, new_capacity);
        Scalar* new_data = _aligned_alloc(new_capacity);
        std::copy(_data, _data + len, new_data);
        aligned_free(_data);
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

    // template<class Index>
    void copy_row(RowIndex from_idx, RowIndex to_idx) {
        assert(from_idx != to_idx);
		insert<false>(row_ptr(from_idx), _cols(), to_idx);
    }

protected:
    Scalar* _aligned_alloc(size_t n) {
        return aligned_alloc<Scalar, AlignBytes>(n);
    }
    ColIndex _aligned_length(size_t ncols) {
        return static_cast<ColIndex>(aligned_length<T, AlignBytes>(ncols));
    }

private:
    Scalar* _data;

    inline const Derived* _derived() const { return static_cast<const Derived*>(this); }
    // inline auto _cols() -> decltype(Derived::cols()) { return Derived::cols(); }
    inline ColIndex _cols() const { return _derived()->cols(); }
};

// ------------------------------------------------ DynamicRowArray

// BaseRowArray subclass with a width set at construction time
template<class T, int AlignBytes=32>
class DynamicRowArray : public BaseRowArray<DynamicRowArray<T, AlignBytes>, T, DynamicRowArrayKey, AlignBytes> {
public:
    // typedef int32_t ColIndex;
    typedef typename row_array_traits<DynamicRowArrayKey, T>::ColIndex ColIndex;
	typedef BaseRowArray<DynamicRowArray<T, AlignBytes>, T, DynamicRowArrayKey, AlignBytes> Super;

    // ------------------------ ctors
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

    // no copying
    DynamicRowArray(const DynamicRowArray& other) = delete;
    DynamicRowArray(const DynamicRowArray&& other) = delete;

    // ------------------------ accessors
    ColIndex cols() const { return _ncols; }

private:
    ColIndex _ncols;
};

// ------------------------------------------------ FixedRowArray

// BaseRowArray subclass that has a fixed width, and so doesn't need to
// store it
template<class T, int NumCols, int AlignBytes=32>
class FixedRowArray : public BaseRowArray<FixedRowArray<T, NumCols, AlignBytes>, T, FixedRowArrayKey, AlignBytes> {
public:
	typedef T Scalar;
	typedef row_array_traits<FixedRowArrayKey, T> Traits;
	typedef typename Traits::RowIndex RowIndex;
	typedef typename Traits::ColIndex ColIndex;
    typedef BaseRowArray<FixedRowArray<T, NumCols, AlignBytes>, T, FixedRowArrayKey, AlignBytes> Super;
    enum { Cols = NumCols };

    // ------------------------ ctors
    FixedRowArray(RowIndex initial_rows):
        Super(initial_rows, NumCols)
    {
        // NOTE: wouldn't need this restriction if we had BaseRowArray
        // always align what it got from derived()->cols()
        size_t aligned_ncols = aligned_length<T, AlignBytes>(NumCols);
        assert(aligned_ncols == NumCols);
		// constexpr size_t aligned_ncols = aligned_length<T, AlignBytes>(NumCols);
   //      static_assert(aligned_ncols == NumCols,
			// "NumCols * sizeof(T) must be a mulitple of AlignBytes");
    }

    FixedRowArray(const FixedRowArray& other) = delete;
    FixedRowArray(const FixedRowArray&& other) = delete;

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

// stores vectors as (optionally) aligned rows in a matrix; lets us access a
// vector and know that it's 32-byte aligned, which enables using
// 32-byte SIMD instructions (as found in 256-bit AVX 2)
//
// this code is somewhat redundant with RowArray code, but refactoring
// to combine them is not a priority atm
template<class T, int AlignBytes=32, class IdT=int64_t>
class RowStore {
public:
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index Index;
    typedef T Scalar;
    typedef IdT Id;
    typedef int8_t PadLengthT;
    typedef Eigen::Map<const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>,
        _AlignHelper<AlignBytes>::AlignmentType> RowT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixT;

    static const bool IsRowMajor = true;

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
