//
//  eigen_utils.hpp
//
//  Created By Davis Blalock on 3/2/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __EIGEN_UTILS_HPP
#define __EIGEN_UTILS_HPP

#include "Dense"

using Eigen::Map;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;


// note that all of these functions assume that the raw arrays being passed
// in are contiguous and stored in row-major order

template<typename T>
Map<Matrix<T, Dynamic, 1> > eigenWrap1D_nocopy(T* x, int len) {
	Map<Matrix<T, Dynamic, 1> > v(x, len);
	return v;
}

template<typename T>
Matrix<T, Dynamic, 1> eigenWrap1D_aligned(const T* x, int len) {
	Matrix<T, Dynamic, 1> v(len);
	for (int i = 0; i < len; i++) {
		v(i) = x[i];
	}
	return v;
}

template<typename T>
Map<Matrix<T, Dynamic, Dynamic, RowMajor> > eigenWrap2D_nocopy(T* X, int m, int n) {
	Map<Matrix<T, Dynamic, Dynamic, RowMajor> > A(X, m, n);
	return A;
}

template<typename T>
Matrix<T, Dynamic, Dynamic, RowMajor> eigenWrap2D_aligned(const T* X, int m, int n) {
	Matrix<T, Dynamic, Dynamic, RowMajor> A(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A(i, j) = X[i*n + j];
		}
	}
	return A;
}

#endif
