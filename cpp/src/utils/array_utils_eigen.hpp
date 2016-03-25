//
//  array_utils.hpp
//
//  Created By Davis Blalock on 3/2/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __ARRAY_UTILS_EIGEN_HPP
#define __ARRAY_UTILS_EIGEN_HPP


#include <algorithm>
#include "array_utils.hpp"

using Eigen::Dynamic;
using Eigen::EigenBase;
using Eigen::Matrix;
using Eigen::RowMajor;

typedef int8_t axis_t;

namespace ar {

template<class F, class Derived1, class Derived2>
static inline void copy(const F&& func, const EigenBase<Derived1>& in,
	EigenBase<Derived2>& out, size_t len=-1) {
	len = min(len, in.size());
	assert(len <= out.size());

	for (size_t i = 0; i < len; i++) {
		out(i) = in(i);
	}
}

// ================================================================
// random numbers
// ================================================================

template<class EigenType, class float_t=double>
static inline void randn(EigenType& X, float_t mean=0, float_t std=1) {
	typedef typename EigenType::Index index_t;

	// create normal distro object
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, std);

	if (X.IsRowMajor) {
		for (index_t i = 0; i < X.rows(); i++) {
			for (index_t j = 0; i < X.cols(); j++) {
				X(i, j) = d(gen);
			}
		}
	} else {
		for (index_t j = 0; j < X.cols(); j++) {
			for (index_t i = 0; i < X.rows(); i++) {
				X(i, j) = d(gen);
			}
		}
	}
}

template<class EigenType>
static inline axis_t _infer_axis(const EigenType& X, axis_t axis) {
	if (axis < 0) {
		if (X.rows() == 1) { // row vect -> along rows
			axis = 1;
		} else if (X.cols() == 1) { // col vect -> along cols
			axis = 0;
		} else { // 2D mat -> along rows
			axis = 1;
		}
	}
	assert(axis < 2); // only 2D arrays supported
	return axis;
}

template<class EigenType, class float_t=double>
static inline void randwalk_inplace(EigenType& X, axis_t axis=-1, float_t std=1)
{
	typedef typename EigenType::Index index_t;
	axis = _infer_axis(X, axis);

	// create normal distro object
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, std);

	if (X.IsRowMajor && axis == 0) {
		for (index_t j = 0; j < X.cols(); j++) { // fill in 1st row
			X(0, j) = d(gen);
		}
		for (index_t i = 1; i < X.rows(); i++) { // fill in remaining rows
			for (index_t j = 0; j < X.cols(); j++) {
				X(i, j) = X(i-1, j) + d(gen);
			}
		}
	} else if (X.IsRowMajor && axis == 1) {
		for (index_t i = 0; i < X.rows(); i++) { // randwalk in each row
			X(i, 0) = d(gen);
			for (index_t j = 1; j < X.cols(); j++) {
				X(i, j) = X(i, j-1) + d(gen);
			}
		}
	} else if (!X.IsRowMajor && axis == 0) {
		for (index_t j = 0; j < X.cols(); j++) { // randwalk in each col
			X(0, j) = d(gen);
			for (index_t i = 1; i < X.rows(); i++) {
				X(i, j) = X(i-1, j) + d(gen);
			}
		}
	} else if (!X.IsRowMajor && axis == 1) {
		for (index_t i = 0; i < X.rows(); i++) { // fill in 1st col
			X(i, 0) = d(gen);
		}
		for (index_t j = 1; j < X.cols(); j++) { // fill in remaining cols
			for (index_t i = 0; i < X.rows(); i++) {
				X(i, j) = X(i, j-1) + d(gen);
			}
		}
	} else {
		assert(false); // impossible
	}
}

template<class data_t=double, class float_t=double>
static inline Matrix<data_t, Dynamic, Dynamic, RowMajor> randwalks(
	size_t nwalks, size_t walklen, float_t std=1)
{
	Matrix<data_t, Dynamic, Dynamic, RowMajor> ret(nwalks, walklen);
	axis_t axis = 1;
	randwalk_inplace(ret, axis, std);
	return ret;
}

// ================================================================
// nonzeros
// ================================================================

template<class DenseT>
static vector<size_t> nonzero_rows(const DenseT& X) {
	vector<size_t> ret;
	for (size_t i = 0; i < X.rows(); i++) {
		for (size_t j = 0; j < X.cols(); j++) {
			if (X(i, j)) {
				ret.push_back(i);
				continue;
			}
		}
	}
	return ret;
}
template<class DenseT>
static vector<size_t> nonzero_cols(const DenseT& X) {
	vector<size_t> ret;
	for (size_t j = 0; j < X.cols(); j++) {
		for (size_t i = 0; i < X.rows(); i++) {
			if (X(i, j)) {
				ret.push_back(j);
				continue;
			}
		}
	}
	return ret;
}

// ================================================================
// vstack and hstack
// ================================================================

} // namespace ar
#endif
