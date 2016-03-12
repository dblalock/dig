//
//  array_utils.hpp
//
//  Created By Davis Blalock on 1/14/14.
//  Copyright (c) 2014 Davis Blalock. All rights reserved.
//

#ifndef __ARRAY_UTILS_HPP
#define __ARRAY_UTILS_HPP

#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <algorithm>
#include <memory>
#include <sstream>
#include <random>
#include <unordered_map>
#include "debug_utils.hpp"

using std::begin;
using std::end;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

namespace ar {

// ================================================================
// Functional Programming
// ================================================================

// ================================ Map
//
// throughout these funcs, we use our own for loop instead of
// std::tranform with a std::back_inserter so that we can use
// emplace_back(), instead of push_back()

// ------------------------------- 1 container version

template <class F, class data_t, class len_t=size_t>
static inline void map_inplace(const F&& func, const data_t* data, len_t len) {
	for (len_t i = 0; i < len; i++) {
		data[i] = (data_t) func(data[i]);
	}
}
template <class F, class data_t, class len_t=size_t>
static inline auto map(const F&& func, const data_t* data, len_t len)
	-> unique_ptr<decltype(func(data[0]))[]>
{
	unique_ptr<decltype(func(data[0]))[]> ret(new decltype(func(data[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(data[i]);
	}
	return ret;
}
/** Returns a new container holding the results of applying the function
 * to the given container */
template<class F, template <class...> class Container, class... Args>
static inline auto map(const F&& func, const Container<Args...>& container)
	-> Container<decltype(func(*begin(container)))>
{
	Container<decltype(func(*begin(container)))> ret;
	for (auto i = begin(container); i < end(container); i++) {
		ret.emplace_back(func(*i));
	}
	return ret;
}

// ------------------------------- 2 container version

template <class F, class data_t1, class data_t2, class len_t=size_t>
static inline auto map(const F&& func, const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(func(x[0], y[0]))[]>
{
	unique_ptr<decltype(func(x[0], y[0]))[]> ret(new decltype(func(x[0], y[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(x[i], y[i]);
	}
	return ret;
}
template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline auto map(const F&& func, const Container1<Args1...>& x, const Container2<Args2...>& y)
	-> Container1<decltype(func(*begin(x), *begin(y)))>
{
	assert(x.size() == y.size());
	Container1<decltype(func(*begin(x), *begin(y)))> ret;
	auto ity = begin(y);
	for (auto itx = begin(x); itx < end(x); itx++, ity++) {
		ret.emplace_back(func(*itx, *ity));
	}
	return ret;
}

// ------------------------------- mapi, 1 container version

template <class F, class data_t, class len_t=size_t>
static inline void mapi_inplace(const F&& func, const data_t* data, len_t len) {
	for (len_t i = 0; i < len; i++) {
		data[i] = (data_t) func(i, data[i]);
	}
}
template <class F, class data_t, class len_t=size_t>
static inline auto mapi(const F&& func, const data_t* data, len_t len)
	-> unique_ptr<decltype(func(len, data[0]))[]>
{
	unique_ptr<decltype(func(len, data[0]))[]>
			ret(new decltype(func(len, data[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(i, data[i]);
	}
	return ret;
}
/** Returns a new container holding the results of applying the function
 * to the given container; the index within the array, as well as the
 * array element itself, are passed to func. */
template<class F, template <class...> class Container, class... Args>
static inline auto mapi(const F&& func, const Container<Args...>& container)
	-> Container<decltype(func(container.size(), *begin(container)))>
{
	Container<decltype(func(container.size(), *begin(container)))> ret;
	size_t i = 0;
	for (const auto& el : container) {
		ret.emplace_back(func(i, el));
		i++;
	}
	// std::transform(begin(container), end(container), std::back_inserter(ret), func);
	return ret;
}

// ------------------------------- mapi, 2 container version
template <class F, class data_t1, class data_t2, class len_t=size_t>
static inline auto mapi(const F&& func, const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(func(len, x[0], y[0]))[]>
{
	unique_ptr<decltype(func(len, x[0], y[0]))[]>
		ret(new decltype(func(len, x[0], y[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(i, x[i], y[i]);
	}
	return ret;
}
template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline auto mapi(const F&& func, const Container1<Args1...>& x, const Container2<Args2...>& y)
	-> Container1<decltype(func(x.size(), *begin(x), *begin(y)))>
{
	assert(x.size() == y.size());
	Container1<decltype(func(x.size(), *begin(x), *begin(y)))> ret;
	auto ity = begin(y);
	size_t i = 0;
	for (auto itx = begin(x); itx < end(x); itx++, ity++, i++) {
		ret.emplace_back(func(i, *itx, *ity));
	}
	return ret;
}


// ================================ Filter

template<class F, template <class...> class Container, class... Args>
static inline Container<Args...> filter(const F&& func,
	const Container<Args...>& container)
{
	Container<Args...> ret;
	for (const auto& el : container) {
		if (func(el)) {
			ret.emplace_back(el);
		}
	}
	return ret;
}
/** Like filter(), but also passes the index within the container to func() as
 * a first argument */
template<class F, template <class...> class Container, class... Args>
static inline Container<Args...> filteri(const F&& func,
	const Container<Args...>& container)
{
	Container<Args...> ret;
	size_t i = 0;
	for (const auto& el : container) {
		if (func(i, el)) {
			ret.emplace_back(el);
		}
		i++;
	}
	return ret;
}

// ================================ Reduce

template<class F, class data_t, class len_t=size_t>
static auto reduce(const F&& func, const data_t* data, len_t len)
	-> decltype(func(data[0], data[0]))
{
	if (len < 1) {
		return NULL;
	}
	if (len == 1) {
		// ideally we would just return the first element,
		// but it might not be the right type
		printf("WARNING: reduce(): called on array with 1 element; ");
		printf("reducing the first element with itself.\n");
		return func(data[0], data[0]);
	}

	auto total = func(data[0], data[1]);
	for (len_t i = 2; i < len; i++) {
		total = func(total, data[i]);
	}
	return total;
}
template<class F, template <class...> class Container, class... Args>
static auto reduce(const F&& func, const Container<Args...>& container)
	-> decltype(func(*begin(container), *end(container)))
{
	auto it = begin(container);
	if (it >= end(container)) {			// 0 elements
		return NULL;
	}
	if (it == end(container) - 1) {		// 1 element
		// ideally we would just return the first element,
		// but it might not be the right type
		printf("WARNING: reduce(): called on container with 1 element; ");
		printf("reducing the first element with itself.\n");
		return func(*it, *it);
	}

	// call func on idxs {0,1}, then on total({0,1}, 2)
	auto total = func(*it, *(it+1));
	for (it += 2; it < end(container); it++) {
		total = func(total, *it);
	}
	return total;
}


// ================================ Where

template<class F, template <class...> class Container, class... Args>
static inline Container<size_t> where(const F&& func, const Container<Args...>& container) {
	Container<size_t> ret;
	size_t i = 0;
	for (const auto& el : container) {
		if (func(el)) {
			ret.emplace_back(i);
		}
		i++;
	}
	return ret;
}

/** Like where(), but also passes the index within the container to func() as
 * a first argument */
template<class F, template <class...> class Container, class... Args>
static inline Container<size_t> wherei(const F&& func, const Container<Args...>& container) {
	Container<size_t> ret;
	size_t i = 0;
	for (const auto& el : container) {
		if (func(i, el)) {
			ret.emplace_back(i);
		}
		i++;
	}
	return ret;
}

// ================================ Find (All)

template<template <class...> class Container, class... Args,
class data_t>
static inline int32_t find(const Container<Args...>& container,
	data_t val) {
	int32_t i = 0;
	for (auto it = std::begin(container); it != std::end(container); ++it) {
		if ((*it) == val) {
			return i;
		}
		i++;
	}
	return -1;
}

template<template <class...> class Container, class... Args,
class data_t>
static inline int32_t rfind(const Container<Args...>& container,
	data_t val) {
	int32_t i = container.size() - 1;
	for (auto it = std::end(container)-1; it >= std::begin(container); --it) {
		if ((*it) == val) {
			return i;
		}
		i--;
	}
	return -1;
}

template<template <class...> class Container, class... Args,
class data_t>
static inline Container<size_t> findall(const Container<Args...>& container,
	data_t val) {
	return where([&val](data_t a) {return a == val;} );
}


template<template <class...> class Container, class... Args,
class data_t>
static inline size_t contains(const Container<Args...>& container,
	data_t val) {
	auto idx = find(container, val);
	return idx >= 0;
}


// ================================ at_idxs

/** note that this requires that the container implement operator[] */
template<class data_t,
	template <class...> class Container2, class... Args2>
static inline vector<data_t> at_idxs(const data_t data[],
		const Container2<Args2...>& indices) {
	vector<data_t> ret;
	for (auto idx : indices) {
		auto val = data[idx];
		ret.push_back(val);
	}
	return ret;
}

/** note that this requires that the container implement operator[] */
template<template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline Container1<Args1...> at_idxs(const Container1<Args1...>& container,
		const Container2<Args2...>& indices,
		bool boundsCheck=true) {
	Container1<Args1...> ret;

	if (boundsCheck) {
		auto len = container.size();
		for (auto idx : indices) {
			if (idx >= 0 && idx < len) {
				auto val = container[static_cast<size_t>(idx)];
				ret.push_back(val);
			}
		}
	} else {
		for (auto idx : indices) {
			auto val = container[static_cast<size_t>(idx)];
			ret.push_back(val);
		}
	}
	return ret;
}


// ================================================================
// Sequence creation
// ================================================================

/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> range(data_t startVal, data_t stopVal, data_t step=1) {
	assertf( (stopVal - startVal) / step > 0,
			"ERROR: range: invalid args min=%.3f, max=%.3f, step=%.3f\n",
			startVal, stopVal, step);

	//allocate a new array
	len_t len = (len_t) floor( (stopVal - startVal) / step );
	unique_ptr<data_t[]> data(new data_t[len]);

	data[0] = startVal;
	for (len_t i = 1; i < len; i++ ) {
		data[i] = data[i-1] + step;
	}
	return data;
}
/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t, class len_t=size_t>
static inline vector<data_t> range_vect(data_t startVal, data_t stopVal, data_t step=1) {
	assertf( (stopVal - startVal) / step > 0,
			"ERROR: range_vect: invalid args min=%.3f, max=%.3f, step=%.3f\n",
			startVal, stopVal, step);

	//allocate a new array
	len_t len = (len_t) floor( (stopVal - startVal) / step );
	vector<data_t> data(len);

	data[0] = startVal;
	for (len_t i = 1; i < len; i++ ) {
		data[i] = data[i-1] + step;
	}
	return data;
}

// ================================================================
// Reshaping
// ================================================================

// reads in a 1D array and returns an array of ND arrays
template <class data_t, class len_t=size_t>
static data_t** split(const data_t* data, len_t len, len_t newNumDims) {
	size_t newArraysLen = len / newNumDims;

	if ( newArraysLen * newNumDims != len) {
		printf("WARNING: reshape: newNumDims %d is not factor of array length %d\n",
			newNumDims, len);
		return nullptr;
	}

	size_t sample,dimension,readFrom=0;
	//initialize each array ptr and the array containing them; note
	//that the arrays are allocated as one contiguous block of memory
	data_t** arrays = new data_t*[newNumDims];
	data_t* arrayContents = new data_t[len];
	for (dimension = 0; dimension < newNumDims; dimension++) {
		arrays[dimension] = arrayContents + dimension*newArraysLen;
	}

	//copy the values from the 1D array to be reshaped
	for (sample = 0; sample < newArraysLen; sample++) {
		for (dimension = 0; dimension < newNumDims; dimension++, readFrom++) {
			arrays[dimension][sample] = data[readFrom];
		}
	}

	return arrays;
}


// ================================================================
// Statistics (V -> R)
// ================================================================

// ================================ Max

/** Returns the maximum value in data[0..len-1] */
template <class data_t, class len_t=size_t>
static inline data_t max(const data_t *data, len_t len) {
	data_t max = std::numeric_limits<data_t>::min();
	for (len_t i = 0; i < len; i++) {
		if (data[i] > max) {
			max = data[i];
		}
	}
	return max;
}
/** Returns the maximum value in data[0..len-1] */
template<template <class...> class Container, class data_t>
static inline data_t max(const Container<data_t>& data) {
	return max(&data[0], data.size());
}

// ================================ Min

/** Returns the minimum value in data[0..len-1] */
template <class data_t, class len_t=size_t>
static inline data_t min(const data_t *data, len_t len) {
	data_t min = std::numeric_limits<data_t>::max();
	for (len_t i = 0; i < len; i++) {
		if (data[i] < min) {
			min = data[i];
		}
	}
	return min;
}
/** Finds the minimum of the elements in data */
template<template <class...> class Container, class data_t>
static inline data_t min(const Container<data_t>& data) {
	return min(&data[0], data.size());
}

// ================================ Sum

/** Computes the sum of data[0..len-1] */
template <class data_t, class len_t=size_t>
static inline data_t sum(const data_t *data, len_t len) {
	return reduce([](data_t x, data_t y){ return x+y;}, data, len);
}
/** Computes the sum of the elements in data */
// template <class data_t>
// data_t sum(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
static inline data_t sum(const Container<data_t>& data) {
	return reduce([](data_t x, data_t y){ return x+y;}, data);
}

// ================================ Sum of Squares

/** Computes the sum of data[i]^2 for i = [0..len-1] */
template <class data_t, class len_t=size_t>
static inline data_t sumsquares(const data_t *data, len_t len) {
	data_t sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += data[i]*data[i];
	}
	return sum;
}
/** Computes the sum of data[i]^2 for i = [0..len-1] */
template<template <class...> class Container, class data_t>
static inline data_t sumsquares(const Container<data_t>& data) {
	return sumsquares(&data[0], data.size());
}

// ================================ Mean

/** Computes the arithmetic mean of data[0..len-1] */
template <class data_t, class len_t=size_t>
static inline double mean(const data_t* data, len_t len) {
	return sum(data, len) / ((double) len);
}
/** Computes the arithmetic mean of data[0..len-1] */
// template <class data_t>
// data_t mean(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
static inline double mean(const Container<data_t>& data) {
	return sum(data) / ((double) data.size());
}

// ================================ Variance

/** Computes the population variance of data[0..len-1] */
template <class data_t, class len_t=size_t>
static inline double variance(const data_t *data, len_t len) {
	if (len <= 1) {
		if (len < 1) {
			printf("WARNING: variance(): received length %lu, returning 0",
				len);
		}
		return 0;
	}

	//use Knuth's numerically stable algorithm
	double mean = data[0];
	double sse = 0;
	double delta;
	for (len_t i=1; i < len; i++) {
		delta = data[i] - mean;
		mean += delta / (i+1);
		sse += delta * (data[i] - mean);
	}
	return sse / len;
}
// template <class data_t>
// data_t variance(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
static inline double variance(const Container<data_t>& data) {
	return variance(&data[0], data.size());
}

// ================================ Standard deviation

/** Computes the population standard deviation of data[0..len-1] */
template <class data_t, class len_t=size_t>
static inline double stdev(const data_t *data, len_t len) {
	return sqrt(variance(data,len));
}

/** Computes the population standard deviation of data[0..len-1] */
template<template <class...> class Container, class data_t>
static inline double stdev(const Container<data_t>& data) {
	return sqrt(variance(data));
}

// ================================================================
// V x V -> R
// ================================================================

// ================================ Dot Product
/** Returns the the dot product of x and y */
template<class data_t1, class data_t2, class len_t=size_t>
static inline auto dot(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] * y[0])
{
	// auto sum = x[0] * y[0]; // get type of sum correct
	// sum = 0;
	decltype(x[0] * y[0]) sum = 0; // get type of sum correct
	for (len_t i = 0; i < len; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dot(const Container1<data_t1>& x, const Container2<data_t2>& y) {
	assert(x.size() == y.size());
	return dot(&x[0],&y[0],x.size());
}

// ================================ L1 Distance

// std::abs is finicky; just inline our own
template<class data_t1, class data_t2>
static inline auto absDiff(data_t1 x, data_t2 y) -> decltype(x - y) {
	return x >= y ? x - y : y - x;
}

template<class data_t1, class data_t2, class len_t=size_t>
static inline auto dist_L1(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
	// auto sum = x[0] * y[0]; // get type of sum correct
	// sum = 0;
	decltype(x[0] - y[0]) sum = 0; // get type of sum correct
	for (len_t i = 0; i < len; i++) {
		sum += absDiff(x[i], y[i]);
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dist_L1(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	assert(x.size() == y.size());
	return dist_L1(&x[0],&y[0],x.size());
}

// ================================ L2^2 Distance

template<class data_t1, class data_t2, class len_t=size_t>
static inline auto dist_sq(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
	decltype(x[0] - y[0]) sum = 0; // get type of sum correct
	for (len_t i = 0; i < len; i++) {
		auto diff = x[i] - y[i];
		sum += diff * diff;
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dist_sq(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	assert(x.size() == y.size());
	return dist_sq(&x[0],&y[0],x.size());
}

// ================================ L2 Distance

template<class data_t1, class data_t2, class len_t=size_t>
static inline auto dist_L2(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
	return sqrt(dist_sq(x, y, len));
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline double dist_L2(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	assert(x.size() == y.size());
	return sqrt(dist_sq(&x[0],&y[0],x.size()));
}

// ================================================================
// Cumulative Statistics (V[1:i] -> R[i])
// ================================================================

// ================================ Cumulative Sum

/** Cumulative sum of elements in src, storing the result in dest */
template <class data_t, class len_t=size_t>
static inline void cumsum(const data_t* src, data_t* dest, len_t len) {
	dest[0] = src[0];
	for (len_t i=1; i < len; i++) {
		dest[i] = src[i] + dest[i-1];
	}
}
/** Returns a new array composed of the cumulative sum of the data */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> cumsum(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	cumsum(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative sum of the data */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cumsum(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	cumsum(&data[0],&ret[0],data.size());
	return ret;
}

// ================================ Cumulative Mean

/** Cumulative mean of elements in src, storing the result in dest */
template <class data_t, class len_t=size_t>
static inline void cummean(const data_t* src, data_t* dest, len_t len) {
	double sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += src[i];
		dest[i] = sum / (i+1);
	}
}
/** Returns a new array composed of the cumulative mean of the data */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> cummean(data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cummean(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative mean of the data */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cummean(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cummean(&data[0],&ret[0],data.size());
	return ret;
}

// ================================ Cumulative SSE

/** Cumulative SSE of elements in src, storing the result in dest */
template <class data_t, class len_t=size_t>
static inline void cumsxx(const data_t* src, data_t* dest, len_t len) {
	if (len < 1) {
		printf("WARNING: cumsxx(): received length %lu, returning 0",
			len);
		return;
	}
	dest[0] = 0;
	if (len == 1) {
		return;
	}

	//use Knuth's numerically stable algorithm
	double mean = src[0];
	double sse = 0;
	double delta;
	for (len_t i=1; i < len; i++) {
		delta = src[i] - mean;
		mean += delta / (i+1);
		sse += delta * (src[i] - mean);
		dest[i] = sse;
	}
}
/** Returns the sum of squared differences from the mean of data[0..i] */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> cumsxx(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cumsxx(data, ret, len);
	return ret;
}
/** Returns the sum of squared differences from the mean of data[0..i] */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cumsxx(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cumsxx(&data[0],&ret[0],data.size());
	return ret;
}

// ================================================================
// V x V -> V
// ================================================================

// ================================ Add

/** Elementwise x + y, storing the result in dest */
template <class data_t1, class data_t2, class data_t3, class len_t=size_t>
static inline void add(const data_t1* x, const data_t2* y, data_t3* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] + y[i];
	}
}
/** Returns a new array composed of elementwise x + y */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto add(const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(x[0]+y[0])[]>
{
	return map([](data_t1 a, data_t2 b){ return a + b;}, x, y, len);
}
/** Returns a new array composed of elementwise x + y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline auto add(const Container1<data_t1>& x, const Container2<data_t2>& y)
	-> Container1<decltype(x[0]+y[0])>
{
	return map([](data_t1 a, data_t2 b){ return a + b;}, x, y);
}

// ================================ Subtract

/** Elementwise x - y, storing the result in dest */
template <class data_t1, class data_t2, class data_t3, class len_t=size_t>
static inline void sub(const data_t1* x, const data_t2* y, data_t3* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] - y[i];
	}
}
/** Returns a new array composed of elementwise x - y */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto sub(const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(x[0]+y[0])[]>
{
	return map([](data_t1 a, data_t2 b){ return a - b;}, x, y, len);
}
/** Returns a new array composed of elementwise x - y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline auto sub(const Container1<data_t1>& x, const Container2<data_t2>& y)
	-> Container1<decltype(x[0]+y[0])>
{
	return map([](data_t1 a, data_t2 b){ return a - b;}, x, y);
}

// ================================ Multiply

/** Elementwise x * y, storing the result in dest */
template <class data_t1, class data_t2, class data_t3, class len_t=size_t>
static inline void mul(const data_t1* x, const data_t2* y, data_t3* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] * y[i];
	}
}
/** Returns a new array composed of elementwise x * y */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto mul(const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(x[0]*y[0])[]>
{
	return map([](data_t1 a, data_t2 b){ return a * b;}, x, y, len);
}
/** Returns a new array composed of elementwise x * y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline auto mul(const Container1<data_t1>& x, const Container2<data_t2>& y)
	-> Container1<decltype(x[0]+y[0])>
{
	return map([](data_t1 a, data_t2 b){ return a * b;}, x, y);
}

// ================================ Divide
//
// TODO decide if we like forcing everything to be a double for this

/** Elementwise x / y, storing the result in dest */
template <class data_t1, class data_t2, class len_t=size_t>
static inline void div(const data_t1* x, const data_t2* y, double* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] / (double) y[i];
	}
}
/** Returns a new array composed of elementwise x / y */
template <class data_t1, class data_t2, class len_t=size_t>
static inline unique_ptr<double[]> div(const data_t1* x, const data_t2* y, len_t len) {
	return map([](data_t1 a, data_t2 b){ return (double)a / b;}, x, y, len);
}
/** Returns a new array composed of elementwise x / y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline Container1<double> div(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	return map([](data_t1 a, data_t2 b){ return (double) a / b;}, x, y);
}

// ================================ Concatenate

template <class data_t, class len_t1=size_t, class len_t2=size_t>
static inline unique_ptr<data_t[]> concat(const data_t* x, const data_t* y,
	len_t1 len1, len_t2 len2)
{
	auto combinedLen = len1 + len2;
	unique_ptr<data_t[]> ret(new data_t[combinedLen]);
	size_t i = 0;
	for( ; i < len1; i++) {
		ret[i] = x[i];
	}
	for( ; i < combinedLen; i++) {
		ret[i] = y[i];
	}
	return ret;
}

template<template <class...> class Container1,
	template <class...> class Container2, class data_t>
static inline Container1<data_t> concat(const Container1<data_t>& x,
	const Container1<data_t>& y, data_t val)
{
	Container1<data_t> ret(x);
	std::copy(begin(y), end(y), std::back_inserter(ret));
	return ret;
}

// ================================================================
// V x R -> V
// ================================================================

// ================================ Add, scalar

/** Adds each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
static inline void adds_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x + val;}, data, len);
}
/** Returns a new array composed of (data[i] + val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto adds(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x + val;}, data, len);
}
/** Returns a new vector composed of (data[i] + val) for all i */
// template <class data_t>
// vector<data_t> add(const vector<data_t>& data, data_t val) {
template<template <class...> class Container, class data_t1, class data_t2>
static inline auto adds(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x + val;}, data);
}

// ================================ Subtract, scalar

/** Adds each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
static inline void subs_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x - val;}, data, len);
}
/** Returns a new array composed of (data[i] + val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto subs(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x - val;}, data, len);
}
/** Returns a new vector composed of (data[i] + val) for all i */
template<template <class...> class Container, class data_t1, class data_t2>
static inline auto subs(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x - val;}, data);
}

// ================================ Multiply, scalar

/** Multiplies each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
static inline void muls_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x * val;}, data, len);
}
/** Returns a new array composed of (data[i] * val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto muls(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x * val;}, data, len);
}
/** Returns a new vector composed of (data[i] * val) for all i */
template<template <class...> class Container, class data_t1, class data_t2>
static inline auto muls(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x * val;}, data);
}

// ================================ Divide, scalars

/** Divides each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
static inline void divs_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x / val;}, data, len);
}
/** Returns a new array composed of (data[i] / val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
static inline auto divs(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x / val;}, data, len);
}
/** Returns a new vector composed of (data[i] / val) for all i */
template<template <class...> class Container, class data_t1, class data_t2>
static inline auto divs(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x / val;}, data);
}

// ================================ Copy

/** Copies src[0..len-1] to dest[0..len-1] */
template <class data_t, class len_t=size_t>
static inline void copy_inplace(const data_t* src, data_t* dest, len_t len) {
	std::copy(src, src+len, dest);
}
/** Returns a copy of the provided array */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> copy(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	std::copy(data, data+len, ret);
	return ret;
}
/** Returns a copy of the provided array */
template<template <class...> class Container, class data_t>
static inline Container<data_t> copy(const Container<data_t>& data) {
	Container<data_t> ret(data);
	return ret;
}

// ================================ Reverse

/** Copies src[0..len-1] to dest[len-1..0] */
template <class data_t, class len_t=size_t>
static inline void reverse(const data_t *src, data_t *dest, len_t len) {
	len_t j = len - 1;
	for (len_t i = 0; i < len; i++, j--) {
		dest[i] = src[j];
	}
}
/** Returns data[len-1..0] */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> reverse(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_reverse(data, ret, len);
	return ret;
}
/** Returns data[len-1..0] */
template<template <class...> class Container, class data_t>
static inline Container<data_t> add(const Container<data_t>& data) {
	Container<data_t> ret(data.size());
	array_reverse(data, &ret[0], data.size());
	return ret;
}

// ================================ Create constant array

/** Sets each element of the array to the value specified */
template <class data_t1, class data_t2, class len_t=size_t>
static inline void set_to_constant(data_t1 *x, data_t2 value, len_t len) {
	for (len_t i = 0; i < len; i++) {
		x[i] = value;
	}
}
template<template <class...> class Container,
	class data_t1, class data_t2>
static inline void set_to_constant(const Container<data_t1>& data, data_t2 value) {
	array_set_to_constant(&data[0], value, data.size());
}

/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> constant(data_t value, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_set_to_constant(ret, value, len);
	return ret;
}
/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> constant_vect(data_t value, len_t len) {
	vector<data_t> ret(len, value);
	return ret;
}

// ================================ Resample

/** Writes the elements of src to dest such that
 * dest[i] = src[ floor(i*srcLen/destLen) ]; note that this function does no
 * filtering of any kind */
template <class data_t, class len_t=size_t>
static inline void resample(const data_t *src, data_t *dest,
	len_t srcLen, len_t destLen)
{
	len_t srcIdx;
	data_t scaleFactor = ((double)srcLen) / destLen;
	for(len_t i = 0; i < destLen; i++) {
		srcIdx = i * scaleFactor;
		dest[i] = src[srcIdx];
	}
}
template <class data_t, class len_t=size_t>
static inline unique_ptr<data_t[]> resample(const data_t* data,
	len_t currentLen, len_t newLen)
{
	unique_ptr<data_t[]> ret(new data_t[newLen]);
	array_resample(data, ret, currentLen, newLen);
	return ret;
}
// template <class data_t, class len_t=size_t>
// vector<data_t> resample(const vector<data_t>& data, len_t newLen) {
template<template <class...> class Container, class data_t, class len_t>
static inline Container<data_t> resample(const Container<data_t>& data, len_t newLen) {
	Container<data_t> ret(newLen);
	array_resample(&data[0], &ret[0], data.size(), newLen);
	return ret;
}

// ================================ Equality

/** Returns true if elements 0..(len-1) of x and y are equal, else false */
template <class data_t1, class data_t2, class len_t=size_t>
static inline bool all_eq(const data_t1 *x, const data_t2 *y, len_t len) {
	for (len_t i = 0; i < len; i++) {
		//TODO define as a const somewhere
		if (std::fabs(x[i] - y[i]) > .00001) return false;
	}
	return true;
}
//template <class Ptr1, class Ptr2, class data_t1, class data_t2,
//	class len_t=size_t>
//bool equal(const Ptr1<data_t1> x, const Ptr2<data_t2> y, len_t len) {
//	for (len_t i = 0; i < len; i++) {
//		//TODO define as a const somewhere
//		if ( fabs(x[i] - y[i]) > .00001 ) return false;
//	}
//	return true;
//}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static inline bool all_eq(const Container1<data_t1>& x, const Container2<data_t2>& y) {
	if (x.size() != y.size()) return 0;
	return all_eq(&x[0], &y[0], x.size());
}

// ================================ Unique
template<template <class...> class Container, class data_t>
static inline Container<data_t> unique(const Container<data_t>& data) {
	Container<data_t> sorted(data);
	auto begin = std::begin(sorted);
	auto end = std::end(sorted);
	std::sort(begin, end);

	Container<data_t> ret;
	std::unique_copy(begin, end, std::back_inserter(ret));
	return ret;
}


// ================================================================
// Stringification / IO
// ================================================================

// ================================ Stringification
template <class data_t, class len_t=size_t>
static std::string to_string(const data_t *x, len_t len) {
	std::ostringstream os;
	os.precision(3);
	os << "[";
	for (len_t i = 0; i < len; i++) {
		os << x[i] << " ";
	}
	os << "]";
	return os.str();
}
// Template specializations--not really needed since no harming
// in setting stream precision unnecessarily
//
//template<>
//std::string to_string<float>(const float *x, size_t len) {
//	std::ostringstream os;
//	os.precision(3);
//	os << "[";
//	for (size_t i = 0; i < len; i++) {
//		os << x[i] << " ";
//	}
//	os << "]";
//	return os.str();
//}
//template<>
//std::string to_string<double>(const double *x, size_t len) {
//	std::ostringstream os;
//	os.precision(3);
//	os << "[";
//	for (size_t i = 0; i < len; i++) {
//		os << x[i] << " ";
//	}
//	os << "]";
//	return os.str();
//}
//template<>
//std::string to_string<char>(const char *x, size_t len) {
//	return std::string(x);
//}

template<template <class...> class Container, class data_t>
static inline std::string to_string(const Container<data_t>& data) {
	return to_string(&data[0], data.size());
}

// ================================ Printing
template <class data_t, class len_t=size_t>
static inline void print(const data_t *x, len_t len) {
	printf("%s\n", to_string(x, len).c_str());
}
template<template <class...> class Container, class data_t>
static inline void print(const Container<data_t>& data) {
	print(&data[0], data.size());
}

template <class data_t, class len_t=size_t>
static inline void print_with_name(const data_t *data, len_t len, const char* name) {
	printf("%s:\t%s\n", name, to_string(data, len).c_str());
}
template<template <class...> class Container, class data_t>
static inline void print_with_name(const Container<data_t>& data, const char* name) {
	print_with_name(&data[0], data.size(), name);
}

// ================================================================
// Randomness
// ================================================================

// ================================ Random Number Generation

// utility func for rand_ints
template<template <class...> class Container, typename K, typename V>
inline V map_get(Container<K, V> map, K key, V defaultVal) {
	if (map.count(key)) {
		return map[key];
	}
	return defaultVal;
}

static inline vector<int64_t> rand_ints(int64_t minVal, int64_t maxVal, uint64_t howMany,
						 bool replace=false) {
	vector<int64_t> ret;

	int64_t numPossibleVals = maxVal - minVal + 1;
	assertf(numPossibleVals >= 1, "No values between min %lld and max %lld",
			minVal, maxVal);

	if (replace) {
		for (size_t i = 0; i < howMany; i++) {
			int64_t val = (rand() % numPossibleVals) + minVal;
			ret.push_back(val);
		}
		return ret;
	}

	assertf(numPossibleVals >= howMany,
			"Can't sample %llu values without replacement between min %lld and max %lld",
			howMany, minVal, maxVal);

	// sample without replacement; each returned int is unique
	unordered_map<int64_t, int64_t> possibleIdxs;
	for (size_t i = 0; i < howMany; i++) {
		int64_t idx = (rand() % (numPossibleVals - i)) + minVal;

		// next value to add to array; just the idx, unless we've picked this
		// idx before, in which case it's whatever the highest unused value
		// was the last time we picked it
		auto val = map_get(possibleIdxs, idx, idx);

		// move highest unused idx into this idx; the result is that the
		// first numPossibleVals-i idxs are all available idxs
		int64_t highestUnusedIdx = maxVal - i;
		possibleIdxs[idx] = map_get(possibleIdxs, highestUnusedIdx,
									highestUnusedIdx);

		ret.push_back(val);
	}
	return ret;
}

// ================================ Random Sampling

template<template <class...> class Container, class data_t>
static inline vector<data_t> rand_choice(const Container<data_t>& data, size_t howMany,
							   bool replace=false) {
	auto maxIdx = data.size() - 1;
	auto idxs = rand_ints(0, maxIdx, howMany, replace);
	return at_idxs(data, idxs, false); // false = no bounds check
}


// ================================ Sorting

template<template <class...> class Container, class data_t>
static inline void sort_inplace(Container<data_t>& data) {
	std::sort(std::begin(data), std::end(data));
}

template<template <class...> class Container, class data_t>
static inline void sort(Container<data_t>& data) {
	Container<data_t> ret(data);
	sort(ret);
	return ret;
}


} // namespace ar
#endif
