//
//  Flock.hpp
//  Dig
//
//  Created by DB on 3/9/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef DIG_FLOCK_HPP
#define DIG_FLOCK_HPP

#include <memory>

#include "Dense" // just for ArrayXd...

// #include "pimpl.hpp"
// #include "pimpl_impl.hpp" // include in header for SWIG; TODO does this fix it?

#include "shape_features.hpp"

// using Eigen::MatrixXd;
using Eigen::ArrayXXd;

// duplicate typedefs here (not just in shape_features) to make SWIG happy
// #include <vector>
// using std::vector;
// #include "type_defs.h"
// using Eigen::Matrix;
// using Eigen::MatrixXd;
// using Eigen::Dynamic;
// using Eigen::RowMajor;
// typedef Matrix<double, Dynamic, Dynamic, RowMajor> CMatrix;
// // typedef Matrix<double, Dynamic, Dynamic> FMatrix;
// typedef MatrixXd FMatrix;

class FlockLearner {
private:
	CMatrix _T;
	FMatrix _Phi;
	FMatrix _Phi_blur;
	ArrayXXd _pattern;
	vector<length_t> _startIdxs;
	vector<length_t> _endIdxs;
	length_t _Lmin;
	length_t _Lmax;
	length_t _Lfilt;

	// class Impl;
	// pimpl<Impl> _self;
	// std::unique_ptr<Impl> _self;
public:
	FlockLearner(const FlockLearner& other) = delete;
	FlockLearner& operator=(const FlockLearner&) = delete;


	// ------------------------ pimpl version
	// forward ctor args to pimpl -- all we need if only calling from cpp
	// template<typename ...Args> FlockLearner( Args&& ...args ):
	// 	_self{ std::forward<Args>(args)... }
	// { }

	// FMatrix getFeatureMat();
	// FMatrix getBlurredFeatureMat();
	// FMatrix getPattern();
	// vector<length_t> getInstanceStartIdxs();
	// vector<length_t> getInstanceEndIdxs();

//	FlockLearner(int foo);
	// explicit ctors for SWIG

	// ------------------------ not-so pimpl version

//	FlockLearner(const double* X, int d, int n, int m_min, int m_max,
//		int m_filt=-1);
	FlockLearner(const double* X, int d, int n, double m_min, double m_max,
		double m_filt=-1);
//	FlockLearner(const double* X, int d, int n);

	FMatrix getFeatureMat() { return _Phi; }
	FMatrix getBlurredFeatureMat() { return _Phi_blur; }
	FMatrix getPattern() { return _pattern.matrix(); }
	// ArrayXXd& getPattern() { return _pattern; }
	vector<length_t> getInstanceStartIdxs() { return _startIdxs; }
	vector<length_t> getInstanceEndIdxs() { return _endIdxs; }
};


#endif
