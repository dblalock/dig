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

#include "nn_search.hpp"
#include "array_utils.hpp"

namespace nn {





// SELF: pick up by having the nn_index classes use (Fixed)RowArrays




// ================================================================
// Utils
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

template<class T>
static inline T* aligned_alloc(size_t n) {
    Eigen::aligned_allocator<T>{}.allocate(n);
}
template<class T>
static inline void aligned_free(T* p) {
    Eigen::aligned_allocator<T>{}.free(p, 0); // 0 unused
}


// ================================================================
// RowArray
// ================================================================
// TODO maybe don't store _ncols because we have 4B dead space with it

// ------------------------------------------------ traits

// class RowArray;
// class FixedRowArray;

// template<class Derived, class Args...>
// struct row_array_traits;

// template <class Args...>
// struct row_array_traits<RowArray> {
//     typedef typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
// };

// template <class T, int NumCols, class Args...>
// struct row_array_traits<FixedRowArray> {
//     typedef typedef Eigen::Matrix<T, Eigen::Dynamic, NumCols, Eigen::RowMajor> MatrixT;
// };

// ------------------------------------------------ BaseRowArray

template<class Derived, class T, int AlignBytes=32>
class BaseRowArray {
	typedef T Scalar;
	// note: using a dynamic matrix here
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
	typedef Eigen::Map<MatrixT> MatrixMapT;

    static const bool IsRowMajor = true;

    // ------------------------ ctors
    BaseRowArray(size_t initial_rows, size_t ncols):
        _data(aligned_alloc<Scalar>(initial_rows * aligned_length<AlignBytes>(ncols)))
    {
        assert(initial_rows >= 0);
    }

    template<class RowMatrixT>
    BaseRowArray(const RowMatrixT& X):
        BaseRowArray(X.rows(), X.cols())
    {
        assert(X.IsRowMajor);
        auto ncols = aligned_length(X.cols());
        MatrixMapT mat(_data, X.rows(), ncols);
        mat.topLeftCorner(X.rows(), X.cols()) = X;
    }

    ~BaseRowArray() { aligned_free(_data); }

    // ------------------------ accessors
    template<class Index>
    Scalar* row_ptr(const Index i) {
        assert(i >= 0);
		return _data + (i * Derived::cols());
    }
    template<class Index>
    const Scalar* row_ptr(const Index i) { return row_ptr(i); }

    // ------------------------ resize, insert, delete
    template<class Index>
    void resize(Index old_num_rows, Index new_num_rows) {
        auto ncols = Derived::cols();
        int64_t old_capacity = old_num_rows * ncols;
        int64_t new_capacity = new_num_rows * ncols;
        // std::unique_ptr<Scalar[]> new_data(new Scalar[new_capacity]);
        // std::copy(_data, _data + old_capacity, new_data.get());
        // _data.swap(new_data);
        Scalar* new_data = aligned_alloc<Scalar>(new_capacity);
        std::copy(_data, _data + old_capacity, new_data);
        aligned_free(_data);
        _data = new_data;
    }

    template<class Index, bool NeedsZeroing=true>
    void insert(const Scalar* x, Index len, Index at_idx) {
        auto row_start = row_ptr(at_idx);
        // zero past end of inserted data (and possibly a bit before)
        if (AlignBytes > 0 && NeedsZeroing) {
            auto ncols = Derived::cols();
            assert(AlignBytes > (sizeof(T) * ncols));
            auto row_end = row_start + ncols;
            auto zero_start = row_end - (AlignBytes / sizeof(T));
            for (auto ptr = zero_start; ptr < row_end; ++ptr) {
                *ptr = 0;
            }
        }
        std::copy(x, x + len, row_start);
    }

    template<class Index>
    void copy_row(Index from_idx, Index to_idx) {
        assert(from_idx != to_idx);
		insert<Index, false>(row_ptr(from_idx), Derived::cols(), to_idx);
    }

protected:
    Scalar* _data;
};

// ------------------------------------------------ RowArray

template<class T, int AlignBytes=32>
class RowArray : public BaseRowArray<RowArray<T, AlignBytes>, T, AlignBytes> {
    typedef T Scalar;
    typedef int32_t ColIndex;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixT;
    typedef Eigen::Map<MatrixT> MatrixMapT;
	typedef BaseRowArray<RowArray<T, AlignBytes>, T, AlignBytes> Super;

    // ------------------------ ctors
    template<class Index>
    RowArray(Index initial_rows, Index ncols):
		Super(initial_rows, aligned_length<AlignBytes>(ncols)),
        _ncols(aligned_length<AlignBytes>(ncols))
    {
        assert(initial_rows >= 0);
    }

    // no copying
    RowArray(const RowArray& other) = delete;
    RowArray(const RowArray&& other) = delete;

    // ------------------------ accessors
    ColIndex cols() const { return _ncols; }

    template<class Index>
    MatrixMapT matrix(Index nrows) {
		MatrixMapT mat(Super::_data, nrows, _ncols);
		return mat;
    }

private:
    ColIndex _ncols;
};

// ------------------------------------------------ FixedRowArray

template<class T, int NumCols, int AlignBytes=32>
class FixedRowArray : public BaseRowArray<FixedRowArray<T, NumCols, AlignBytes>, T, AlignBytes> {
    typedef T Scalar;
	typedef int32_t ColIndex;
    typedef Eigen::Matrix<T, Eigen::Dynamic, NumCols, Eigen::RowMajor> MatrixT;
    typedef Eigen::Map<MatrixT> MatrixMapT;
    typedef BaseRowArray<FixedRowArray<T, NumCols, AlignBytes>, T, AlignBytes> Super;

    // ------------------------ ctors
    FixedRowArray(const FixedRowArray& other) = delete;
    FixedRowArray(const FixedRowArray&& other) = delete;

    // ------------------------ accessors
    inline constexpr ColIndex cols() const { return NumCols; }

    template<class Index>
    MatrixMapT matrix(Index nrows) {
        MatrixMapT mat(Super::_data, nrows, NumCols);
        return mat;
    }
};

// ================================================================
// RowStore
// ================================================================

// stores vectors as (optionally) aligned rows in a matrix; lets us access a
// vector and know that it's 32-byte aligned, which enables using
// 32-byte SIMD instructions (as found in 256-bit AVX 2)
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
            // Index new_size = std::max(capacity + 1, _capacity * 1.5) * _ncols;
            // std::unique_ptr<Scalar[]> new_data(new Scalar[new_capacity]);
            // std::copy(_data, _data + size(), new_data.get());
            // _capacity = new_capacity;
            // _data.swap(new_data);
            Index new_capacity = std::max(cap + 1, cap * 1.5);
            _data.conservativeResize(new_capacity, Eigen::NoChange);
            ret = new_capacity;
        }
//      _data.row(rows()) =
//        Scalar* row_start_ptr = _data + (_ncols * rows());
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
        // move data at the end to overwrite the removed row so that
        // everything remains contiguous
        // auto last_idx = _ids.size() - 1;
        // _ids[idx] = _ids[last_idx];
        // auto last_row_start = row_ptr(last_idx);
        // Scalar* replace_row_ptr = row_ptr(idx);
        // std::copy(last_row_start, last_row_start + _ncols, replace_row_ptr);

        _ids[idx] = _ids.back();
        _ids.pop_back(); // reduce size
        return idx;
    }

private:
    // int32_t _ncols; // 4B
    // int32_t _capacity; // 4B
    std::vector<Id> _ids; // 24B; size() = number of entries
    RowMatrixT _data; // 24B; rows, cols = capacity, vector length
    PadLengthT _pad_width; // 1B
    // std::unique_ptr<Scalar[]> _data; // 8B
};

} // namespace nn

#endif // __FLAT_STORE_HPP
