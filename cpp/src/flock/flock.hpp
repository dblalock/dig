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

#include "pimpl.hpp"

#include "shape_features.hpp" // for FMatrix and CMatrix typedefs

// template<class T>
class FlockLearner {
private:
	class Impl;
	pimpl<Impl> _self;
public:
	FlockLearner(const FlockLearner& other) = delete;
	FlockLearner& operator=(const FlockLearner&) = delete;

	// forward ctor args to pimpl -- all we need if only calling from cpp
//	template<typename ...Args> FlockLearner( Args&& ...args ):
//		_self{ std::forward<Args>(args)... }
//	{ }

//	FlockLearner(int foo);
	
	// explicit ctors for SWIG
//	FlockLearner(const double* X, int d, int n, int m_min, int m_max,
//		int m_filt=-1);
	FlockLearner(const double* X, int d, int n, double m_min, double m_max,
		double m_filt=-1);
//	FlockLearner(const double* X, int d, int n);
	
	FMatrix getFeatureMat();
	FMatrix getBlurredFeatureMat();
	FMatrix getPattern();
	vector<length_t> getInstanceStartIdxs();
	vector<length_t> getInstanceEndIdxs();
};

// template class FlockLearner<double>; // explicit instantiation for SWIG


#endif
