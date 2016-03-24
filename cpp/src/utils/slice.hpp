//
//  slice.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __SLICE_HPP
#define __SLICE_HPP

#include <type_traits>

#include "macros.hpp"
#include "array_utils.hpp"
// #include "restrict.h"

using std::remove_reference;

namespace ar {

// ================================================================ Slice Obj

// TODO worth finishing this, but need to finish algo port for now;
// stuff we want in here:
	// -static type trait for whether it's rowmajor or not
	// -ability to get ptr to start of row or col
	// -ability to get slice for row or col
	// -ability to modify the data ptr, so it works with sliding windows
	// -mirror Eigen::Dense / Eigen::EigenBase API so below funcs for eigen
	//	types will just magically work with it also
	// -prolly have copy ctor defaulted, but operator= copy the data
	//		-and work with arbitrary containers or raw arrays
	// -possibly rowEndSkip and colEndSkip so you can take the middle cols
	//  in a row-major slice or vice versa
	// 		-actually, think this is subsumed by having an array of strides
	//		-just means it has a huge stride in one direction
	// -prolly have explicit Slice1, Slice2, Slice3 etc for different orders
	//		-with one base class that covers shared functionality
	//		-alternative is to template on order, but I think we'll end up with
	// 		 a bunch of variadic constructors and stuff
	// -prolly only have IsRowMajor for 1D and 2D slices; needs to be rowmajor
	//  for higher order than this

// class Slice2<bool IsRowMajor=1, T> {
// public:
// 	T* data;
// 	length_t size;
// 	length_t strides[ORDER];
// 	const bool IsRowMajor = IsRowMajor; // initial caps to parallel Eigen

// 	// typedefs to parallel those in Eigen
// 	typedef T type;
// 	typedef length_t Index;

// 	inline T* data() const { return data; }
// 	inline const T* data() const { return data; }
// 	inline length_t size() const { return size; }
// 	inline length_t stride() const { return strides[0]; }
// 	inline bool isRowMajor() const { return IsRowMajor; }

// 	inline T* rowPtr(length_t i) {
// 		if (IsRowMajor)
// 		return data *
// 	}
// };

// ================================================================
// slicing
// ================================================================
// TODO restrict keyword here, or compile with -fno-strict-aliasing

// template<class data_t, class idx_t>
// static inline data_t* row_start_idx_rowmajor(data_t* basePtr, length_t m,
// 	length_t n, idx_t rowIdx)
// {
// 	return basePtr + (rowIdx * n);
// }
// template<class data_t, class idx_t>
// static inline data_t* row_start_idx_rowmajor(data_t* basePtr, length_t m,
// 	length_t n, idx_t rowIdx)
// {
// 	return basePtr + (rowIdx * n);
// }

// ================================ slice args checking

template<class idx_t>
static inline bool check_idxs_valid(const idx_t* idxs, length_t idxsLen,
	length_t maxIdx)
{
	if (idxsLen <= 0) {
		return false;
	}
	assert(idxs);

	for (length_t j = 0; j < idxsLen; j++) {
		auto idx = idxs[j];
		assert(0 <= idx);
		assert(idx <= maxIdx);
	}
	return true;
}

template<class data_t, class idx_t1, class idx_t2>
static inline void slice_check_args(const data_t* in, length_t m, length_t n,
	const idx_t1* rowIdxs, length_t rowIdxsLen,
	const idx_t2* colIdxs, length_t colIdxsLen, data_t* out)
{
	assert(m > 0);
	assert(n > 0);
	assert(in);
	assert(out);

	auto slicingRows = check_idxs_valid(rowIdxs, rowIdxsLen, m-1);
	auto slicingCols = check_idxs_valid(colIdxs, colIdxsLen, n-1);
	assert(slicingRows || slicingCols);
}

// ================================ copy from src to dest, raw ptrs
//
// TODO replace all of these with calls to copy_strided2

// ------------------------ rows, rowmajor

template<class data_t, class idx_t, REQUIRE_INT(idx_t)>
static void copy_rows_rowmajor(const data_t *RESTRICT in, length_t m,
	length_t n, const idx_t* idxs, length_t idxsLen, data_t *RESTRICT out)
{
	slice_check_args(in, m, n, idxs, idxsLen, nullptr, 0, out);

	for (length_t i = 0; i < idxsLen; i++) {
		auto idx = idxs[i];
		auto inRowPtr = in + (n * idx);
		auto outRowPtr = out + (n * idx);
		ar::copy(inRowPtr, outRowPtr, n);
	}
}
template<class data_t, class idx_t, REQUIRE_INT(idx_t)>
static inline unique_ptr<data_t[]> copy_rows_rowmajor(const data_t* in,
	length_t m, length_t n, const idx_t* idxs, length_t idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_rowmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ cols, rowmajor

template<class data_t, class idx_t>
static void copy_cols_rowmajor(const data_t *RESTRICT in, length_t m, length_t n,
	const idx_t* idxs, length_t idxsLen, data_t *RESTRICT out)
{
	slice_check_args(in, m, n, nullptr, 0, idxs, idxsLen, out);

	for (length_t i = 0; i < m; i++) {
		auto inRowPtr = in + (n * i);
		auto outRowPtr = out + (n * i);
		for (length_t j = 0; j < idxsLen; j++) {
			auto idx = idxs[j];
			auto inPtr = inRowPtr + j;
			auto outPtr = outRowPtr + j;
			*outPtr = *inPtr;
		}
	}
}
template<class data_t, class idx_t>
static inline unique_ptr<data_t[]> copy_cols_rowmajor(const data_t* in,
	length_t m, length_t n, const idx_t* idxs, length_t idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_cols_rowmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ cols, colmajor

template<class data_t, class idx_t>
static void copy_cols_colmajor(const data_t* in, length_t m, length_t n,
	const idx_t* idxs, length_t idxsLen, data_t* out)
{
	copy_rows_rowmajor(in, n, m, idxs, idxsLen, out);
}
template<class data_t, class idx_t>
static inline unique_ptr<data_t[]> copy_cols_colmajor(const data_t* in,
	length_t m, length_t n, const idx_t* idxs, length_t idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_cols_colmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ rows, colmajor

template<class data_t, class idx_t>
static void copy_rows_colmajor(const data_t* in, length_t m, length_t n,
	const idx_t* idxs, length_t idxsLen, data_t* out)
{
	copy_cols_rowmajor(in, n, m, idxs, idxsLen, out);
}
template<class data_t, class idx_t>
static inline unique_ptr<data_t[]> copy_rows_colmajor(const data_t* in,
	length_t m, length_t n, const idx_t* idxs, length_t idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_colmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ rows and cols, rowmajor

template<class data_t, class idx_t1, class idx_t2,
	REQUIRE_INT(idx_t1), REQUIRE_INT(idx_t2)>
static void copy_rows_cols_rowmajor(const data_t *RESTRICT in, length_t m, length_t n,
	const idx_t1* rowIdxs, length_t rowIdxsLen,
	const idx_t2* colIdxs, length_t colIdxsLen, data_t *RESTRICT out)
{
	slice_check_args(in, m, n, rowIdxs, rowIdxsLen, colIdxs, colIdxsLen, out);

	for (length_t i = 0; i < rowIdxsLen; i++) {
		auto rowIdx = rowIdxs[i];
		auto inRowPtr = in + (n * rowIdx);
		auto outRowPtr = out + (n * rowIdx);
		for (length_t j = 0; j < colIdxsLen; j++) {
			auto colIdx = rowIdxs[i];
			auto inPtr = inRowPtr + colIdx;
			auto outPtr = outRowPtr + colIdx;
			*outPtr = *inPtr;
		}
	}
}
template<class data_t, class idx_t1, class idx_t2,
	REQUIRE_INT(idx_t1), REQUIRE_INT(idx_t2)>
static inline unique_ptr<data_t[]> copy_rows_cols_rowmajor(
	const data_t* in, length_t m, length_t n,
	const idx_t1* rowIdxs, length_t rowIdxsLen,
	const idx_t2* colIdxs, length_t colIdxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_cols_rowmajor(in, m, n, rowIdxs, rowIdxsLen,
		colIdxs, colIdxsLen, ret);
	return ret;
}

// ------------------------ rows and cols, colmajor

template<class data_t, class idx_t1, class idx_t2,
	REQUIRE_INT(idx_t1), REQUIRE_INT(idx_t2)>
static inline void copy_rows_cols_colmajor(const data_t* in, length_t m, length_t n,
	const idx_t1* rowIdxs, length_t rowIdxsLen,
	const idx_t2* colIdxs, length_t colIdxsLen, data_t* out)
{
	copy_rows_cols_rowmajor(in, n, m, colIdxs, colIdxsLen,
		rowIdxs, rowIdxsLen, out);
}
template<class data_t, class idx_t1, class idx_t2>
static inline unique_ptr<data_t[]> copy_rows_cols_rowmajor(
	const data_t* in, length_t m, length_t n,
	const idx_t1* rowIdxs, length_t rowIdxsLen,
	const idx_t2* colIdxs, length_t colIdxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_cols_rowmajor(in, m, n, rowIdxs, rowIdxsLen,
		colIdxs, colIdxsLen, ret);
	return ret;
}

// ================================================================ Major conversion

// ------------------------ all data, rowmajor to colmajor

template<class data_t>
static void copy_rowmajor_to_colmajor(const data_t *RESTRICT in, length_t m,
	length_t n, data_t *RESTRICT out,
	length_t rowStrideIn=1, length_t colStrideIn=1,
	length_t rowStrideOut=1, length_t colStrideOut=1)
{
	rowStrideIn = max(rowStrideIn, m);
	colStrideIn = max(colStrideIn, 1);
	rowStrideOut = max(rowStrideOut, 1);
	colStrideOut = max(colStrideOut, n);

	copy_strided2(in, m, n, out, rowStrideIn, colStrideIn,
		rowStrideOut, colStrideOut);
}
template<class data_t>
static void copy_colmajor_to_rowmajor(const data_t *RESTRICT in, length_t m,
	length_t n, data_t *RESTRICT out,
	length_t rowStrideIn=1, length_t colStrideIn=1,
	length_t rowStrideOut=1, length_t colStrideOut=1)
{
	rowStrideIn = max(rowStrideIn, 1);
	colStrideIn = max(colStrideIn, n);
	rowStrideOut = max(rowStrideOut, m);
	colStrideOut = max(colStrideOut, 1);

	copy_strided2(in, m, n, out, rowStrideIn, colStrideIn,
		rowStrideOut, colStrideOut);
}

// what params mean:
// 	-rowStride = &x[i+1, j] - &x[i, j] = dist to same col in next row
// 	-colStride = &x[i, j+1] - &x[i, j] = dist to same row in next col
template<class data_t>
static void copy_strided2(const data_t *RESTRICT in, length_t m, length_t n,
	data_t *RESTRICT out, length_t rowStrideIn=1, length_t colStrideIn=1,
	length_t rowStrideOut=1, length_t colStrideOut=1)
{
	auto inStrideDiff = rowStrideIn - colStrideIn;
	auto outStrideDiff = rowStrideOut - colStrideOut;
	auto strideDiff = inStrideDiff + outStrideDiff;
	bool rowMajor = strideDiff >= 0; // rowStrides bigger than col strides

	if (rowMajor) {
		for (length_t i = 0; i < m; i++) {
			for (length_t j = 0; j < n; j++) {
				auto inPtr = in + (i * rowStrideIn) + (j * colStrideIn);
				auto outPtr = out + (i * rowStrideOut) + (j * colStrideOut);
				*outPtr = *inPtr;
			}
		}
	} else { // same as above, but loop order reversed
		for (length_t j = 0; j < n; j++) {
			for (length_t i = 0; i < m; i++) {
				auto inPtr = in + (i * rowStrideIn) + (j * colStrideIn);
				auto outPtr = out + (i * rowStrideOut) + (j * colStrideOut);
				*outPtr = *inPtr;
			}
		}
	}
}

template<class data_t, class idx_t1, class idx_t2,
	REQUIRE_INT(idx_t1), REQUIRE_INT(idx_t2)>
static void copy_strided2(const data_t* in, length_t m, length_t n,
	data_t *RESTRICT out,
	const idx_t1* rowIdxs, length_t rowIdxsLen,
	const idx_t2* colIdxs, length_t colIdxsLen,
	length_t rowStrideIn=1, length_t colStrideIn=1,
	length_t rowStrideOut=1, length_t colStrideOut=1)
{
	slice_check_args(in, m, n, rowIdxs, rowIdxsLen, colIdxs, colIdxsLen, out);

	auto inStrideDiff = rowStrideIn - colStrideIn;
	auto outStrideDiff = rowStrideOut - colStrideOut;
	auto strideDiff = inStrideDiff + outStrideDiff;
	bool rowMajor = strideDiff >= 0; // rowStrides bigger than col strides

	if (rowMajor) {
		for (length_t i = 0; i < rowIdxsLen; i++) {
			auto rowIdx = rowIdxs[i];
			for (length_t j = 0; j < colIdxsLen; j++) {
				auto colIdx = colIdxs[j];
				auto inPtr = in + (rowIdx * rowStrideIn) + (colIdx * colStrideIn);
				auto outPtr = out + (colIdx * rowStrideOut) + (colIdx * colStrideOut);
				*outPtr = *inPtr;
			}
		}
	} else { // same as above, but loop order reversed
		for (length_t j = 0; j < colIdxsLen; j++) {
			auto colIdx = colIdxs[j];
			for (length_t i = 0; i < rowIdxsLen; i++) {
				auto rowIdx = rowIdxs[i];
				auto inPtr = in + (rowIdx * rowStrideIn) + (colIdx * colStrideIn);
				auto outPtr = out + (colIdx * rowStrideOut) + (colIdx * colStrideOut);
				*outPtr = *inPtr;
			}
		}
	}
}



// ------------------------ all data, colmajor to rowmajor

// ================================================================ Select

// ------------------------ select rows

template<class DenseT, class idx_t1>
static inline DenseT select_rows(const DenseT& X, idx_t1* idxs, length_t idxsLen) {
	DenseT ret(idxsLen, X.cols());
	auto X_ = X.eval();
	if (X_.IsRowMajor) {
		copy_rows_rowmajor(X_.data(), X_.rows(), X_.cols(),
			idxs, idxsLen, ret.data());
	} else {
		copy_rows_colmajor(X_.data(), X_.rows(), X_.cols(),
			idxs, idxsLen, ret.data());
	}
	return ret;
}
template<class DenseT, template <class...> class Container, class... Args>
static inline DenseT select_rows(const DenseT& X, const Container<Args...>& idxs) {
	return select_rows(X, idxs.data(), idxs.size());
}

// ------------------------ select cols

template<class DenseT, class idx_t1>
static inline DenseT select_cols(const DenseT& X, idx_t1* idxs, length_t idxsLen) {
	DenseT ret(X.rows(), idxsLen);
	auto X_ = X.eval();
	if (X_.IsRowMajor) {
		copy_cols_rowmajor(X_.data(), X_.rows(), X_.cols(),
			idxs, idxsLen, ret.data());
	} else {
		copy_cols_colmajor(X_.data(), X_.rows(), X_.cols(),
			idxs, idxsLen, ret.data());
	}
	return ret;
}
template<class DenseT, template <class...> class Container, class... Args>
static inline DenseT select_cols(const DenseT& X, const Container<Args...>& idxs) {
	return select_cols(X, idxs.data(), idxs.size());
}

// ------------------------ select rows and cols

template<class DenseT, class idx_t1, class idx_t2>
static inline DenseT select(const DenseT& X, idx_t1* rowIdxs, length_t rowIdxsLen,
	idx_t2* colIdxs, length_t colIdxsLen)
{
	DenseT ret(rowIdxsLen, colIdxsLen);
	auto X_ = X.eval();
	if (X_.IsRowMajor) {
		copy_rows_cols_rowmajor(X_.data(), X_.rows(), X_.cols(),
			rowIdxs, rowIdxsLen, colIdxs, colIdxsLen, ret.data());
	} else {
		copy_rows_cols_colmajor(X_.data(), X_.rows(), X_.cols(),
			rowIdxs, rowIdxsLen, colIdxs, colIdxsLen, ret.data());
	}
	return ret;
}
template<class DenseT, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline DenseT select(const DenseT& X,
	const Container1<Args1...>& rowIdxs, const Container2<Args2...>& colIdxs)
{
	return select(X, rowIdxs.data(), rowIdxs.size(),
		colIdxs.data(), colIdxs.size());
}

// ================================================================ Concat
// only works with dynamically sized Eigen objects

template<template <class...> class Container, class DenseT>
static typename remove_reference<DenseT>::type vstack(Container<DenseT> mats) {
	assert(mats.size());
	typedef typename DenseT::Index idx_t;

	// compute size of concatenated matrix
	idx_t nrows = 0;
	idx_t ncols = *begin(mats).cols();
	for (auto& mat : mats) {
		assert(mat.cols() == ncols);
		nrows += mat.rows();
	}

	typename remove_reference<DenseT>::type out(nrows, ncols);
	idx_t row = 0;
	for (auto& mat : mats) {
		idx_t numRows = mat.rows();
		out.middleRows(row, numRows) = mat;
		row += numRows;
	}

	return out;
}

template<template <class...> class Container, class DenseT>
static typename remove_reference<DenseT>::type hstack(Container<DenseT> mats) {
	assert(mats.size());
	typedef typename DenseT::Index idx_t;

	// compute size of concatenated matrix
	idx_t nrows = *begin(mats).cols();
	idx_t ncols = 0;
	for (auto& mat : mats) {
		assert(mat.rows() == ncols);
		ncols += mat.cols();
	}

	typename remove_reference<DenseT>::type out(nrows, ncols);
	idx_t col = 0;
	for (auto& mat : mats) {
		idx_t numCols = mat.rows();
		out.middleCols(col, numCols) = mat;
		col += numCols;
	}

	return out;
}



} // namespace ar
#endif
