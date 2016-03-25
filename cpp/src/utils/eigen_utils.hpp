//
//  eigen_utils.hpp
//
//  Created By Davis Blalock on 3/2/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __EIGEN_UTILS_HPP
#define __EIGEN_UTILS_HPP

#include "Dense"

#include <vector>

using Eigen::Map;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;

// ================================================================
// converting to eigen mats
// ================================================================
// note that all of these functions assume that the raw arrays being passed
// in are contiguous and stored in row-major order

// ------------------------------------------------ 1D output

// ------------------------ raw array input

template<typename T>
static inline Map<Matrix<T, Dynamic, 1> > eigenWrap1D_nocopy(T* x, size_t len) {
	Map<Matrix<T, Dynamic, 1> > v(x, len);
	return v;
}
template<typename T>
static inline Map<const Matrix<T, Dynamic, 1> >
eigenWrap1D_nocopy_const(const T* x, size_t len) {
	Map<const Matrix<T, Dynamic, 1> > v(x, len);
	return v;
}

template<typename T>
static inline Matrix<T, Dynamic, 1> eigenWrap1D_aligned(const T* x, size_t len) {
	Matrix<T, Dynamic, 1> v(len);
	for (int i = 0; i < len; i++) {
		v(i) = x[i];
	}
	return v;
}

// ------------------------ container input

template<template <class...> class Container, class T>
static inline auto eigenWrap1D_nocopy(const Container<T>& data)
	-> decltype(eigenWrap1D_nocopy(data.data(), data.size()))
{
	return eigenWrap1D_nocopy(data.data(), data.size());
}
template<template <class...> class Container, class T>
static inline auto eigenWrap1D_nocopy_const(const Container<T>& data)
	-> decltype(eigenWrap1D_nocopy_const(data.data(), data.size()))
{
	return eigenWrap1D_nocopy_const(data.data(), data.size());
}

template<template <class...> class Container, class T>
static inline Matrix<T, Dynamic, 1>
eigenWrap1D_aligned(const Container<T>& data) {
	return eigenWrap1D_aligned(data.data(), data.size());
}

// ------------------------------------------------ 2D output

template<typename T>
static inline Map<Matrix<T, Dynamic, Dynamic, RowMajor> >
eigenWrap2D_nocopy(T* X, int m, int n) {
	Map<Matrix<T, Dynamic, Dynamic, RowMajor> > A(X, m, n);
	return A;
}
template<typename T>
static inline Map<const Matrix<T, Dynamic, Dynamic, RowMajor> >
eigenWrap2D_nocopy_const(const T* X, int m, int n) {
	Map<const Matrix<T, Dynamic, Dynamic, RowMajor> > A(X, m, n);
	return A;
}

template<typename T>
static inline Matrix<T, Dynamic, Dynamic, RowMajor>
eigenWrap2D_aligned(const T* X, int m, int n) {
	Matrix<T, Dynamic, Dynamic, RowMajor> A(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A(i, j) = X[i*n + j];
		}
	}
	return A;
}

#endif




