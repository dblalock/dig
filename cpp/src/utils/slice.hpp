//
//  slice.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __SLICE_HPP
#define __SLICE_HPP

#include <type_traits>

#include "Dense"

#include "macros.hpp"
#include "array_utils.hpp"
// #include "restrict.h"

using std::remove_reference;
using Eigen::DenseBase;
using Eigen::Dynamic;

namespace ar {

// ================================================================
// slicing
// ================================================================
// TODO restrict keyword here, or compile with -fno-strict-aliasing

// template<class data_t, class len_t1, class len_t2, class idx_t>
// static inline data_t* row_start_idx_rowmajor(data_t* basePtr, len_t1 m,
// 	len_t2 n, idx_t rowIdx)
// {
// 	return basePtr + (rowIdx * n);
// }
// template<class data_t, class len_t1, class len_t2, class idx_t>
// static inline data_t* row_start_idx_rowmajor(data_t* basePtr, len_t1 m,
// 	len_t2 n, idx_t rowIdx)
// {
// 	return basePtr + (rowIdx * n);
// }

// ================================ slice args checking

template<class idx_t, class len_t1, class len_t2>
static inline bool check_idxs_valid(const idx_t* idxs, len_t1 idxsLen, len_t2 maxIdx) {
	if (idxsLen <= 0) {
		return false;
	}
	assert(idxs);

	for (len_t2 j = 0; j < idxsLen; j++) {
		auto idx = idxs[j];
		assert(0 <= idx);
		assert(idx <= maxIdx);
	}
	return true;
}

template<class data_t, class len_t, class idx_t1, class idx_t2,
	class len_t2, class len_t3>
static inline void slice_check_args(const data_t* in, len_t m, len_t n,
	const idx_t1* rowIdxs, len_t1 rowIdxsLen,
	const idx_t2* colIdxs, len_t2 colIdxsLen, data_t* out)
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

// ------------------------ rows, rowmajor

template<class data_t, class len_t, class len_t2, class len_t3>
static void copy_rows_rowmajor(const data_t *RESTRICT in, len_t m, len_t n,
	const len_t2* idxs, len_t3 idxsLen, data_t *RESTRICT out)
{
	slice_check_args(in, m, n, idxs, idxsLen, nullptr, 0, out);

	for (len_t2 i = 0; i < idxsLen; i++) {
		auto idx = idxs[i];
		auto inRowPtr = in + (n * idx);
		auto outRowPtr = out + (n * idx);
		ar::copy(inRowPtr, outRowPtr, n)
	}
}
template<class data_t, class len_t, class len_t2, class len_t3>
static inline unique_ptr<data_t[]> copy_rows_rowmajor(const data_t* in,
	len_t m, len_t n, const len_t2* idxs, len_t3 idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_rowmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ cols, rowmajor

template<class data_t, class len_t, class len_t2, class len_t3>
static void copy_cols_rowmajor(const data_t *RESTRICT in, len_t m, len_t n,
	const len_t2* idxs, len_t3 idxsLen, data_t *RESTRICT out)
{
	slice_check_args(in, m, n, nullptr, 0, idxs, idxsLen, out);

	for (len_t2 i = 0; i < m; i++) {
		auto inRowPtr = in + (n * i);
		auto outRowPtr = out + (n * i);
		for (len_t2 j = 0; j < idxsLen; j++) {
			auto idx = idxs[j];
			auto inPtr = inRowPtr + j;
			auto outPtr = outRowPtr + j;
			*outPtr = *inPtr;
		}
	}
}
template<class data_t, class len_t, class len_t2, class len_t3>
static inline unique_ptr<data_t[]> copy_cols_rowmajor(const data_t* in,
	len_t m, len_t n, const len_t2* idxs, len_t3 idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_cols_rowmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ cols, colmajor

template<class data_t, class len_t, class len_t2, class len_t3>
static void copy_cols_colmajor(const data_t* in, len_t m, len_t n,
	const len_t2* idxs, len_t3 idxsLen, data_t* out)
{
	copy_rows_rowmajor(in, n, m, idxs, idxsLen, out);
}
template<class data_t, class len_t, class len_t2, class len_t3>
static inline unique_ptr<data_t[]> copy_cols_colmajor(const data_t* in,
	len_t m, len_t n, const len_t2* idxs, len_t3 idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_cols_colmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ rows, colmajor

template<class data_t, class len_t, class len_t2, class len_t3>
static void copy_rows_colmajor(const data_t* in, len_t m, len_t n,
	const len_t2* idxs, len_t3 idxsLen, data_t* out)
{
	copy_cols_rowmajor(in, n, m, idxs, idxsLen, out);
}
template<class data_t, class len_t, class len_t2, class len_t3>
static inline unique_ptr<data_t[]> copy_rows_colmajor(const data_t* in,
	len_t m, len_t n, const len_t2* idxs, len_t3 idxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_colmajor(in, ret, m, n, idxs, idxsLen);
	return ret;
}

// ------------------------ rows and cols, rowmajor

template<class data_t, class len_t, class idx_t1, class idx_t2,
	class len_t2, class len_t3>
static void copy_rows_cols_rowmajor(const data_t *RESTRICT in, len_t m, len_t n,
	const idx_t1* rowIdxs, len_t1 rowIdxsLen,
	const idx_t2* colIdxs, len_t2 colIdxsLen, data_t *RESTRICT out)
{
	slice_check_args(in, m, n, rowIdxs, rowIdxsLen, colIdxs, colIdxsLen, out);

	for (len_t1 i = 0; i < rowIdxsLen; i++) {
		auto rowIdx = rowIdxs[i];
		auto inRowPtr = in + (n * rowIdx);
		auto outRowPtr = out + (n * rowIdx);
		for (len_t2 j = 0; j < colIdxsLen; j++) {
			auto colIdx = rowIdxs[i];
			auto inPtr = inRowPtr + colIdx;
			auto outPtr = outRowPtr + colIdx;
			*outPtr = *inPtr;
		}
	}
}
template<class data_t, class len_t, class idx_t1, class idx_t2,
	class len_t2, class len_t3>
static inline unique_ptr<data_t[]> copy_rows_cols_rowmajor(
	const data_t* in, len_t m, len_t n,
	const idx_t1* rowIdxs, len_t1 rowIdxsLen,
	const idx_t2* colIdxs, len_t2 colIdxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_cols_rowmajor(in, m, n, rowIdxs, rowIdxsLen,
		colIdxs, colIdxsLen, ret);
	return ret;
}

// ------------------------ rows and cols, colmajor

template<class data_t, class len_t, class idx_t1, class idx_t2,
	class len_t2, class len_t3>
static inline void copy_rows_cols_colmajor(const data_t* in, len_t m, len_t n,
	const idx_t1* rowIdxs, len_t1 rowIdxsLen,
	const idx_t2* colIdxs, len_t2 colIdxsLen, data_t* out)
{
	copy_rows_cols_rowmajor(in, n, m, colIdxs, colIdxsLen,
		rowIdxs, rowIdxsLen, out);
}
template<class data_t, class len_t, class idx_t1, class idx_t2,
	class len_t2, class len_t3>
static inline unique_ptr<data_t[]> copy_rows_cols_rowmajor(
	const data_t* in, len_t m, len_t n,
	const idx_t1* rowIdxs, len_t1 rowIdxsLen,
	const idx_t2* colIdxs, len_t2 colIdxsLen)
{
	unique_ptr<data_t[]> ret(new data_t[m*n]);
	copy_rows_cols_rowmajor(in, m, n, rowIdxs, rowIdxsLen,
		colIdxs, colIdxsLen, ret);
	return ret;
}

// ================================================================ Eigen

// ------------------------ select rows

template<class DenseT, class idx_t1, class len_t1>
static inline DenseT select_rows(const DenseT& X, idx_t1* idxs, len_t1 idxsLen) {
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

template<class DenseT, class idx_t1, class len_t1>
static inline DenseT select_cols(const DenseT& X, idx_t1* idxs, len_t1 idxsLen) {
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

template<class DenseT, class idx_t1, class len_t1, class idx_t2, class len_t2>
static inline DenseT select(const DenseT& X, idx_t1* rowIdxs, len_t1 rowIdxsLen,
	idx_t2* colIdxs, len_t2 colIdxsLen)
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
static remove_reference<DenseT>::type vstack(Container<DenseT> mats) {
	assert(mats.size());
	typedef typename DenseT::Index idx_t;

	// compute size of concatenated matrix
	idx_t nrows = 0;
	idx_t ncols = *begin(mats).cols();
	for (auto& mat : mats) {
		assert(mat.cols() == ncols);
		nrows += mat.rows();
	}

	remove_reference<DenseT>::type out(nrows, ncols);
	idx_t row = 0;
	for (auto& mat : mats) {
		idx_t numRows = mat.rows();
		out.middleRows(row, numRows) = mat;
		row += numRows;
	}

	return out;
}

template<template <class...> class Container, class DenseT>
static remove_reference<DenseT>::type hstack(Container<DenseT> mats) {
	assert(mats.size());
	typedef typename DenseT::Index idx_t;

	// compute size of concatenated matrix
	idx_t nrows = *begin(mats).cols();
	idx_t ncols = 0;
	for (auto& mat : mats) {
		assert(mat.rows() == ncols);
		ncols += mat.cols();
	}

	remove_reference<DenseT>::type out(nrows, ncols);
	idx_t col = 0;
	for (auto& mat : mats) {
		idx_t numCols = mat.rows();
		out.middleCols(col, numCols) = mat;
		row += numCols;
	}

	return out;
}



} // namespace ar
#endif
