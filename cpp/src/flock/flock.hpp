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

template<class T>
class FlockLearner {
private:
	class Impl;
//	std::unique_ptr<Impl> _ths;
	pimpl<Impl> _self;
public:
//	~FlockLearner();
	FlockLearner(const FlockLearner& other) = delete;
	FlockLearner& operator=(const FlockLearner&) = delete;
	
	FlockLearner(const T* X, int n, int d, int m_min, int m_max);
	
	// forward ctor args to pimpl
	template<typename ...Args> FlockLearner( Args&& ...args ):
		_self{ std::forward<Args>(args)... }
	{ }
};

template class FlockLearner<double>; // explicit instantiation for SWIG


#endif
