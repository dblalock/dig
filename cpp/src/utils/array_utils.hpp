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

// #include "restrict.h"
#include "macros.hpp"
#include "debug_utils.hpp"

using std::begin;
using std::end;
using std::log2;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

namespace ar {

static const double kDefaultNonzeroThresh = .001;

// ================================================================
// Scalar funcs
// ================================================================

// ------------------------ less picky min/max/abs funcs than stl

template<class data_t>
static inline data_t abs(data_t x) {
	return x >= 0 ? x : -x;
}

template<class data_t1, class data_t2>
static inline auto max(data_t1 x, data_t2 y) -> decltype(x + y) {
	return x >= y ? x : y;
}

template<class data_t1, class data_t2>
static inline auto min(data_t1 x, data_t2 y) -> decltype(x + y) {
	return x <= y ? x : y;
}

template<class data_t1, class data_t2>
static inline auto absDiff(data_t1 x, data_t2 y) -> decltype(x - y) {
	return x >= y ? x - y : y - x;
}

// ------------------------ logical ops

template<class data_t1, class data_t2>
static inline bool logical_and(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) && static_cast<bool>(y);
}

template<class data_t1, class data_t2>
static inline bool logical_nand(data_t1 x, data_t2 y) {
	return !logical_and(x, y);
}

template<class data_t1, class data_t2>
static inline bool logical_or(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) || static_cast<bool>(y);
}

template<class data_t1, class data_t2>
static inline bool logical_nor(data_t1 x, data_t2 y) {
	return !logical_or(x, y);
}

template<class data_t1, class data_t2>
static inline bool logical_xor(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) != static_cast<bool>(y);
}

template<class data_t1, class data_t2>
static inline bool logical_xnor(data_t1 x, data_t2 y) {
	return static_cast<bool>(x) == static_cast<bool>(y);
}

template<class data_t>
static inline bool logical_not(data_t x) {
	return !static_cast<bool>(x);
}

// ------------------------ bitwise ops

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_and(data_t1 x, data_t2 y) {
	return static_cast<data_t1>(x & y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_nand(data_t1 x, data_t2 y) {
	return ~bitwise_and(x, y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_or(data_t1 x, data_t2 y) {
	return static_cast<data_t1>(x | y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline bool bitwise_nor(data_t1 x, data_t2 y) {
	return ~bitwise_or(x, y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline data_t1 bitwise_xor(data_t1 x, data_t2 y) {
	return static_cast<data_t1>(x ^ y);
}

template<class data_t1, class data_t2, REQUIRE_PRIMITIVE(data_t1),
	REQUIRE_PRIMITIVE(data_t2)>
static inline data_t1 bitwise_xnor(data_t1 x, data_t2 y) {
	return ~bitwise_xor(x, y);
}

template<class data_t, REQUIRE_PRIMITIVE(data_t)>
static inline data_t bitwise_not(data_t x) {
	return ~x;
}

// ================================================================
// Functional Programming
// ================================================================

// ================================ Map
// throughout these funcs, we use our own for loop instead of
// std::tranform with a std::back_inserter so that we can use
// emplace_back(), instead of push_back()

// ------------------------------- 1 container version

template <class F, class data_t, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline void map(const F&& func, const data_t *RESTRICT data,
	len_t len, data_t2 *RESTRICT out) {
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t2>(func(data[i]));
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void map_inplace(const F&& func, const data_t* data, len_t len) {
	// static_assert(std::is_integral<len_t>::value, "");
//	ASSERT_INTEGRAL(len_t);
	for (len_t i = 0; i < len; i++) {
		data[i] = static_cast<data_t>(func(data[i]));
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline auto map(const F&& func, const data_t* data, len_t len)
	-> unique_ptr<decltype(func(data[0]))[]>
{
//	ASSERT_INTEGRAL(len_t);
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

template <class F, class data_t1, class data_t2, class data_t3, class len_t,
	 REQUIRE_INT(len_t)>
static inline void map(const F&& func, const data_t1* x, const data_t2* y,
	len_t len, data_t3 *RESTRICT out)
{
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t3>(func(x[i], y[i]));
	}
}
template <class F, class data_t1, class data_t2, class len_t,
	REQUIRE_INT(len_t)>
static inline auto map(const F&& func, const data_t1* x, const data_t2* y,
	len_t len) -> unique_ptr<decltype(func(x[0], y[0]))[]>
{
	unique_ptr<decltype(func(x[0], y[0]))[]> ret(
		new decltype(func(x[0], y[0]))[len]);
	for (len_t i = 0; i < len; i++) {
		ret[i] = func(x[i], y[i]);
	}
	return ret;
}
template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static inline auto map(const F&& func, const Container1<Args1...>& x,
	const Container2<Args2...>& y)
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

template <class F, class data_t, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline void mapi(const F&& func, const data_t *RESTRICT data, len_t len,
	data_t2 *RESTRICT out) {
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t2>(func(i, data[i]));
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void mapi_inplace(const F&& func, const data_t* data, len_t len) {
	for (len_t i = 0; i < len; i++) {
		data[i] = (data_t) func(i, data[i]);
	}
}
template <class F, class data_t, class len_t, REQUIRE_INT(len_t)>
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
	return ret;
}

// ------------------------------- mapi, 2 container version
template <class F, class data_t1, class data_t2, class data_t3, class len_t,
	REQUIRE_INT(len_t)>
static inline void mapi(const F&& func, const data_t1* x, const data_t2* y,
	len_t len, data_t3 *RESTRICT out)
{
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t3>(func(i, x[i], y[i]));
	}
}
template <class F, class data_t1, class data_t2, class len_t,
	REQUIRE_INT(len_t)>
static inline auto mapi(const F&& func, const data_t1* x,
	const data_t2* y, len_t len)
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
static inline auto mapi(const F&& func, const Container1<Args1...>& x,
	const Container2<Args2...>& y)
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

template<class F, class data_t, class len_t, REQUIRE_INT(len_t)>
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
static inline Container<size_t> where(const F&& func,
	const Container<Args...>& container)
{
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
static inline Container<size_t> wherei(const F&& func,
	const Container<Args...>& container)
{
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

// ================================ Contains

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
		bool boundsCheck=false) {
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

// ================================ range

template <class data_t1, class data_t2, class step_t=int8_t>
static inline int32_t num_elements_in_range(data_t1 startVal,
											data_t2 stopVal, step_t step)
{
	assertf( (stopVal - startVal) / step > 0,
			"ERROR: range: invalid args min=%.3f, max=%.3f, step=%.3f\n",
			(double)startVal, (double)stopVal, (double)step);
	return ceil((stopVal - startVal) / step);
}

template <class data_t0, class data_t1, class data_t2, class step_t=int8_t>
static inline void range_inplace(data_t0* data, data_t1 startVal,
								 data_t2 stopVal, step_t step=1)
{
	int32_t len = num_elements_in_range(startVal, stopVal, step);
	data[0] = startVal;
	for (int32_t i = 1; i < len; i++) {
		data[i] = data[i-1] + step;
	}
}

/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t1, class data_t2, class step_t=int8_t>
static inline auto range_ar(data_t1 startVal, data_t2 stopVal, step_t step=1)
	-> unique_ptr<decltype(stopVal - startVal + step)[]>
{
	int32_t len = num_elements_in_range(startVal, stopVal, step);
	unique_ptr<decltype(stopVal - startVal + step)[]> data(new decltype(stopVal - startVal + step)[len]);
	range_inplace(data, startVal, stopVal, step);
	return data;
}
/** Create an array containing a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t1, class data_t2, class step_t=int8_t>
static inline auto range(data_t1 startVal, data_t2 stopVal, step_t step=1)
	-> vector<decltype(stopVal - startVal + step)>
{
	int32_t len = num_elements_in_range(startVal, stopVal, step);
	vector<decltype(stopVal - startVal + step)> data(len);
	range_inplace(data.data(), startVal, stopVal, step);
	return data;
}

// ================================ exprange

template <class data_t1, class data_t2, class step_t=int8_t>
static inline int32_t num_elements_in_exprange(data_t1 startVal,
											   data_t2 stopVal, step_t step)
{
	assertf(startVal != 0, "exprange(): start value == 0!");
	assertf(stopVal != 0, "exprange(): end value == 0!");
	assertf(step != 0, "exprange(): step == 0!");

	auto absStartVal = abs(startVal);
	auto absStopVal = abs(stopVal);
	auto absStep = abs(step);

	if (absStartVal > absStopVal) {
		assertf(absStep < 1,
				"exprange(): |startVal| %.3g > |stopVal| %.3g, but |step| %.3g >= 1",
				(double)absStartVal, (double)absStopVal, (double)absStep);
	} else if (absStartVal < absStopVal) {
		assertf(absStep > 1,
				"exprange(): |startVal| %.3g > |stopVal| %.3g, but |step| %.3g <= 1",
				(double)absStartVal, (double)absStopVal, (double)absStep);
	} else {
		return 1; // startVal == stopVal
	}

	double ratio = static_cast<double>(absStopVal) / absStartVal;
	double logBaseStep = log2(ratio) / log2(absStep);
	return 1 + floor(logBaseStep);
}
template <class data_t0, class data_t1, class data_t2, class step_t=int8_t>
static inline void exprange_inplace(data_t0* data, data_t1 startVal,
								 data_t2 stopVal, step_t step=2)
{
	int32_t len = num_elements_in_exprange(startVal, stopVal, step);
	data[0] = startVal;
	for (int32_t i = 1; i < len; i++) {
		data[i] = data[i-1] * step;
	}
}

// step is int so it can be powers of 2 by default
template <class data_t, class step_t=int8_t>
static inline auto exprange_ar(data_t startVal, data_t stopVal, step_t step=2)
	-> unique_ptr<decltype(stopVal * startVal * step)[]>
{
	int32_t len = num_elements_in_exprange(startVal, stopVal, step);
	unique_ptr<decltype(stopVal * startVal * step)[]> data(
		new decltype(stopVal * startVal * step)[len]);
	exprange_inplace(data, startVal, stopVal, step);
	return data;
}

template <class data_t1, class data_t2, class step_t=int8_t>
static inline auto exprange(data_t1 startVal, data_t2 stopVal, step_t step=2)
-> vector<decltype(stopVal * startVal * step)>
{
	int32_t len = num_elements_in_exprange(startVal, stopVal, step);
	vector<decltype(stopVal * startVal * step)> data(len);
	exprange_inplace(data.data(), startVal, stopVal, step);
	return data;
}

// ================================ Create constant array

/** Sets each element of the array to the value specified */
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline void constant_inplace(data_t1 *x, len_t len, data_t2 value) {
	for (len_t i = 0; i < len; i++) {
		x[i] = static_cast<data_t1>(value);
	}
}
template<template <class...> class Container,
	class data_t1, class data_t2>
static inline void constant_inplace(Container<data_t1>& data, data_t2 value) {
	constant_inplace(data.data(), value, data.size());
}

/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> constant_ar(len_t len, data_t value) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	constant_inplace(ret, value, len);
	return ret;
}
/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline vector<data_t> constant(len_t len, data_t value) {
	vector<data_t> ret(len, value);
	return ret;
}

// ================================================================
// Reshaping
// ================================================================

// reads in a 1D array and returns an array of ND arrays
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static data_t** split(const data_t* data, len_t len, len_t newNumDims) {
	size_t newArraysLen = len / newNumDims;
	assertf(newArraysLen * newNumDims == len,
		"reshape(): newNumDims %d is not factor of array length %d",
		newNumDims, len);

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
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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
	return max(data.data(), data.size());
}

// ================================ Min

/** Returns the minimum value in data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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
	return min(data.data(), data.size());
}

// ================================ Sum

/** Computes the sum of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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
	return sumsquares(data.data(), data.size());
}

// ================================ Mean

/** Computes the arithmetic mean of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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

// Knuth's numerically stable algorithm for online mean + variance. See:
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
template<class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void knuth_sse_stats(const data_t* data, len_t len,
	double& mean, double& sse)
{
	mean = data[0];
	sse = 0;
	double delta;
	for (len_t i=1; i < len; i++) {
		delta = data[i] - mean;
		mean += delta / (i+1);
		sse += delta * (data[i] - mean);
	}
}

/** Computes the population variance of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double variance(const data_t *data, len_t len) {
	assert(len > 0);
	if (len == 1) {
		return 0;
	}
	double mean, sse;
	knuth_sse_stats(data, len, mean, sse);
	return sse / len;
}
template<template <class...> class Container, class data_t>
static inline double variance(const Container<data_t>& data) {
	return variance(data.data(), data.size());
}

// ================================ Standard deviation

/** Computes the population standard deviation of data[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double stdev(const data_t *data, len_t len) {
	return sqrt(variance(data,len));
}

/** Computes the population standard deviation of data[0..len-1] */
template<template <class...> class Container, class data_t>
static inline double stdev(const Container<data_t>& data) {
	return sqrt(variance(data));
}

// ================================ LP norm

/** Computes the arithmetic mean of data[0..len-1] */
template <class data_t, class len_t, class pow_t=int>
static inline double norm(const data_t* data, len_t len, pow_t p=2) {
	data_t sum = 0;
	for(len_t i = 0; i < len; i++) {
		sum += std::pow(data[i], p);
	}
	return std::pow(sum, 1.0 / p);
}
/** Computes the arithmetic mean of data[0..len-1] */
template <int P, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline double norm(const data_t* data, len_t len) {
	return norm(data, len, P);
}

template<int P, template <class...> class Container, class data_t>
static inline double norm(const Container<data_t>& data) {
	return norm<P>(data.data(), data.size());
}
template<template <class...> class Container, class data_t, class pow_t=int>
static inline double norm(const Container<data_t>& data, pow_t p=2) {
	return norm(data.data(), data.size(), p);
}

// ================================ L1 norm

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t norm_L1(const data_t* data, len_t len) {
	return sum(data, len);
}
template<template <class...> class Container, class data_t>
static inline data_t norm_L1(const Container<data_t>& data) {
	return sum(data);
}

// ================================ L2 norm

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline data_t norm_L2(const data_t* data, len_t len) {
	return sqrt(sumsquares(data, len));
}
template<template <class...> class Container, class data_t>
static inline data_t norm_L2(const Container<data_t>& data) {
	return sqrt(sumsquares(data));
}

// ================================================================
// V x V -> R
// ================================================================

// ================================ Dot Product
/** Returns the the dot product of x and y */
template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline auto dot(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] * y[0])
{
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
	return dot(x.data(), y.data(), x.size());
}

// ================================ L1 Distance

template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
static inline auto dist_L1(const data_t1* x, const data_t2* y, len_t len)
	-> decltype(x[0] - y[0])
{
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
	return dist_L1(x.data(), y.data(), x.size());
}

// ================================ L2^2 Distance

template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
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
	return dist_sq(x.data(), y.data(), x.size());
}

// ================================ L2 Distance

template<class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
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
	return sqrt(dist_sq(x.data(), y.data(), x.size()));
}

// ================================================================
// Cumulative Statistics (V[1:i] -> R[i])
// ================================================================

// ================================ Cumulative Sum

/** Cumulative sum of elements in src, storing the result in dest */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void cumsum(const data_t* src, data_t* dest, len_t len) {
	dest[0] = src[0];
	for (len_t i=1; i < len; i++) {
		dest[i] = src[i] + dest[i-1];
	}
}
/** Returns a new array composed of the cumulative sum of the data */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> cumsum(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	cumsum(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative sum of the data */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cumsum(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	cumsum(data.data(),ret.data(),data.size());
	return ret;
}

// ================================ Cumulative Mean

/** Cumulative mean of elements in src, storing the result in dest */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void cummean(const data_t* src, data_t* dest, len_t len) {
	double sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += src[i];
		dest[i] = sum / (i+1);
	}
}
/** Returns a new array composed of the cumulative mean of the data */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> cummean(data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cummean(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative mean of the data */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cummean(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cummean(data.data(), ret.data(), data.size());
	return ret;
}

// ================================ Cumulative SSE

/** Cumulative SSE of elements in src, storing the result in dest */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void cumsxx(const data_t* src, len_t len, data_t* dest) {
	assert(len > 0);
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
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> cumsxx(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cumsxx(data, len, ret);
	return ret;
}
/** Returns the sum of squared differences from the mean of data[0..i] */
template<template <class...> class Container, class data_t>
static inline Container<data_t> cumsxx(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cumsxx(data.data(), data.size(), ret.data());
	return ret;
}

// ================================================================
// V x V -> V
// ================================================================

#define WRAP_VECTOR_VECTOR_OP_WITH_NAME(OP, NAME) \
template <class data_t1, class data_t2, class len_t, class data_t3, \
	REQUIRE_INT(len_t)> \
static inline void NAME(const data_t1* x, const data_t2* y, len_t len, \
	data_t3* out) \
{ \
	return map([](data_t1 a, data_t2 b){ return a OP b;}, x, y, len, out); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* x, const data_t2* y, len_t len) \
	-> unique_ptr<decltype(x[0] OP y[0])[]> \
{ \
	return map([](data_t1 a, data_t2 b){ return a OP b;}, x, y, len); \
} \
template<template <class...> class Container1, class data_t1, \
template <class...> class Container2, class data_t2> \
static inline auto NAME(const Container1<data_t1>& x, \
	const Container2<data_t2>& y) -> Container1<decltype(x[0] OP y[0])> \
{ \
	return map([](data_t1 a, data_t2 b){ return a OP b;}, x, y); \
} \

// TODO do +=, etc, also via map_inplace
WRAP_VECTOR_VECTOR_OP_WITH_NAME(+, add)
WRAP_VECTOR_VECTOR_OP_WITH_NAME(-, sub)
WRAP_VECTOR_VECTOR_OP_WITH_NAME(*, mul)
// WRAP_VECTOR_VECTOR_OP_WITH_NAME(/ (double), div) // cast denominator to double
WRAP_VECTOR_VECTOR_OP_WITH_NAME(/, div)


#define WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, NAME) \
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME(const data_t1* x, const data_t2* y, data_t3* out, \
	len_t len) \
{ \
	for (len_t i = 0; i < len; i++) { \
		out[i] = static_cast<data_t3>(FUNC(x[i], y[i])); \
	} \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* x, const data_t2* y, \
	len_t len) -> unique_ptr<decltype(FUNC(x[0], y[0]))[]> \
{ \
	return map([](data_t1 a, data_t2 b){ return FUNC(a, b);}, x, y, len); \
} \
template<template <class...> class Container1, class data_t1, \
	template <class...> class Container2, class data_t2> \
static inline auto NAME(const Container1<data_t1>& x, \
	const Container2<data_t2>& y) -> Container1<decltype(FUNC(x[0], y[0]))> \
{ \
	return map([](data_t1 a, data_t2 b){ return FUNC(a, b);}, x, y); \
} \

#define WRAP_VECTOR_VECTOR_FUNC(FUNC) \
	WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_VECTOR_VECTOR_STD_FUNC_WITH_NAME(FUNC, NAME) \
	using std::FUNC; \
	WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_VECTOR_VECTOR_STD_FUNC(FUNC) \
	WRAP_VECTOR_VECTOR_STD_FUNC_WITH_NAME(FUNC, FUNC)

// WRAP_VECTOR_VECTOR_FUNC(max);
// WRAP_VECTOR_VECTOR_FUNC(min);
// WRAP_VECTOR_VECTOR_FUNC(logical_and);
// WRAP_VECTOR_VECTOR_FUNC(logical_nand);
// WRAP_VECTOR_VECTOR_FUNC(logical_or);
// WRAP_VECTOR_VECTOR_FUNC(logical_nor);
// WRAP_VECTOR_VECTOR_FUNC(logical_xor);
// WRAP_VECTOR_VECTOR_FUNC(logical_xnor);
// WRAP_VECTOR_VECTOR_FUNC(logical_not);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_and);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_nand);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_or);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_nor);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_xor);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_xnor);
// WRAP_VECTOR_VECTOR_FUNC(bitwise_not);
// WRAP_VECTOR_VECTOR_STD_FUNC(pow);

// ================================ Concatenate

template <class data_t, class len_t1, class len_t2>
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

// ================================ wrap V x R (and R x V) operators

#define WRAP_VECTOR_SCALAR_OP_WITH_NAME(OP, NAME) \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME(data_t1 *RESTRICT data, len_t len, data_t2 val, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return x OP val;}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace(data_t1* data, len_t len, data_t2 val) { \
	map_inplace([val](data_t1 x){return x OP val;}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* data, len_t len, data_t2 val) \
	-> unique_ptr<decltype(data[0] OP val)[]> \
{ \
	return map([val](data_t1 x) {return x OP val;}, data, len); \
} \
template<template <class...> class Container, class data_t1, class data_t2> \
static inline auto NAME(const Container<data_t1>& data, data_t2 val) \
	-> Container<decltype(*begin(data) OP val)> \
{ \
	return map([val](data_t1 x) {return x OP val;}, data); \
} \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NUM(data_t2)> \
static inline void NAME(data_t2 val, data_t1 *RESTRICT data, len_t len, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return val OP x;}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NUM(data_t2)> \
static inline void NAME ## _inplace(data_t2 val, data_t1* data, len_t len) { \
	map_inplace([val](data_t1 x){return val OP x;}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NUM(data_t2)> \
static inline auto NAME(data_t2 val, const data_t1* data, len_t len) \
	-> unique_ptr<decltype(val OP data[0])[]> \
{ \
	return map([val](data_t1 x) {return val OP x;}, data, len); \
} \
template<template <class...> class Container, class data_t1, \
	class data_t2, REQUIRE_NUM(data_t2)> \
static inline auto NAME(data_t2 val, const Container<data_t1>& data) \
	-> Container<decltype(val OP *begin(data))> \
{ \
	return map([val](data_t1 x) {return val OP x;}, data); \
} \

// TODO do +=, etc, also via map_inplace
// note that we need a separate macro for operators because there's
// no associated function for them in C++
WRAP_VECTOR_SCALAR_OP_WITH_NAME(+, add)
WRAP_VECTOR_SCALAR_OP_WITH_NAME(-, sub)
WRAP_VECTOR_SCALAR_OP_WITH_NAME(*, mul)
WRAP_VECTOR_SCALAR_OP_WITH_NAME(/, div)

// ================================ wrap V x R (and R x V) funcs

#define WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, NAME) \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t)> \
static inline void NAME(data_t1 *RESTRICT data, len_t len, data_t2 val, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return FUNC(x, val);}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace(data_t1* data, len_t len, data_t2 val) { \
	map_inplace([val](data_t1 x){return FUNC(x, val);}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t1* data, len_t len, data_t2 val) \
	-> unique_ptr<decltype(FUNC(data[0], val))[]> \
{ \
	return map([val](data_t1 x) {return FUNC(x, val);}, data, len); \
} \
template<template <class...> class Container, class data_t1, class data_t2> \
static inline auto NAME(const Container<data_t1>& data, data_t2 val) \
	-> Container<decltype(FUNC(*begin(data), val))> \
{ \
	return map([val](data_t1 x) {return FUNC(x, val);}, data); \
} \
\
template <class data_t1, class data_t2, class data_t3, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NUM(data_t2)> \
static inline void NAME(data_t2 val, data_t1 *RESTRICT data, len_t len, \
	data_t3 *RESTRICT out) \
{ \
	map([val](data_t1 x){return FUNC(val, x);}, data, len, out); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NUM(data_t2)> \
static inline void NAME ## _inplace(data_t2 val, data_t1* data, len_t len) { \
	map_inplace([val](data_t1 x){return FUNC(val, x);}, data, len); \
} \
template <class data_t1, class data_t2, class len_t, \
	REQUIRE_INT(len_t), REQUIRE_NUM(data_t2)> \
static inline auto NAME(data_t2 val, const data_t1* data, len_t len) \
	-> unique_ptr<decltype(FUNC(val, data[0]))[]> \
{ \
	return map([val](data_t1 x) {return FUNC(val, x);}, data, len); \
} \
template<template <class...> class Container, class data_t1, \
	class data_t2, REQUIRE_NUM(data_t2)> \
static inline auto NAME(data_t2 val, const Container<data_t1>& data) \
	-> Container<decltype(FUNC(val, *begin(data)))> \
{ \
	return map([val](data_t1 x) {return FUNC(val, x);}, data); \
} \

#define WRAP_VECTOR_SCALAR_FUNC(FUNC) \
	WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_VECTOR_SCALAR_STD_FUNC_WITH_NAME(FUNC, NAME) \
	using std::FUNC; \
	WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_VECTOR_SCALAR_STD_FUNC(FUNC) \
	WRAP_VECTOR_SCALAR_STD_FUNC_WITH_NAME(FUNC, FUNC)

// ================================ wrap binary op

#define WRAP_BINARY_FUNC_WITH_NAME(FUNC, NAME) \
	WRAP_VECTOR_VECTOR_FUNC_WITH_NAME(FUNC, NAME) \
	WRAP_VECTOR_SCALAR_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_BINARY_FUNC(FUNC) \
	WRAP_BINARY_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_BINARY_FUNC_WITH_NAME_IN_NAMESPACE(FUNC, NAME, NAMESPACE) \
	using NAMESPACE::FUNC;
	WRAP_BINARY_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_BINARY_STD_FUNC_WITH_NAME(FUNC, NAME) \
	WRAP_BINARY_FUNC_WITH_NAME_IN_NAMESPACE(FUNC, NAME, std)

#define WRAP_BINARY_STD_FUNC(FUNC) \
	WRAP_BINARY_STD_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_BINARY_FUNC(FUNC) \
	WRAP_BINARY_FUNC_WITH_NAME(FUNC, FUNC)

WRAP_BINARY_FUNC(max);
WRAP_BINARY_FUNC(min);
WRAP_BINARY_FUNC(logical_and);
WRAP_BINARY_FUNC(logical_nand);
WRAP_BINARY_FUNC(logical_or);
WRAP_BINARY_FUNC(logical_nor);
WRAP_BINARY_FUNC(logical_xor);
WRAP_BINARY_FUNC(logical_xnor);
WRAP_BINARY_FUNC(logical_not);
WRAP_BINARY_FUNC(bitwise_and);
WRAP_BINARY_FUNC(bitwise_nand);
WRAP_BINARY_FUNC(bitwise_or);
WRAP_BINARY_FUNC(bitwise_nor);
WRAP_BINARY_FUNC(bitwise_xor);
WRAP_BINARY_FUNC(bitwise_xnor);
WRAP_BINARY_FUNC(bitwise_not);
WRAP_BINARY_STD_FUNC(pow);

// ================================ Pow

// // ------------------------ elements to scalar power

// template <class data_t1, class data_t2, class data_t3, class len_t,
// 	REQUIRE_INT(len_t)>
// static inline void pow(data_t1 *RESTRICT data, len_t len, data_t2 val,
// 	data_t3 *RESTRICT out)
// {
// 	map([val](data_t1 x){return std::pow(x, val);}, data, len, out);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline void pows_inplace(data_t1* data, len_t len, data_t2 val) {
// 	map_inplace([val](data_t1 x){return std::pow(x, val);}, data, len);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline unique_ptr<double[]> pow(const data_t1* data, len_t len,
// 	data_t2 val)
// {
// 	return map([val](data_t1 x) {return std::pow(x, val);}, data, len);
// }
// template<template <class...> class Container, class data_t1, class data_t2>
// static inline Container<double> pow(const Container<data_t1>& data,
// 	data_t2 val)
// {
// 	return map([val](data_t1 x) {return std::pow(x, val);}, data);
// }

// // ------------------------ scalar to power of elements

// template <class data_t1, class data_t2, class data_t3, class len_t,
// 	REQUIRE_INT(len_t)>
// static inline void pow(data_t2 val, data_t1 *RESTRICT data, len_t len,
// 	data_t3 *RESTRICT out)
// {
// 	map([val](data_t1 x){return std::pow(val, x);}, data, len, out);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline void pow_inplace(data_t2 val, data_t1* data, len_t len) {
// 	map_inplace([val](data_t1 x){return std::pow(val, x);}, data, len);
// }
// template <class data_t1, class data_t2, class len_t, REQUIRE_INT(len_t)>
// static inline unique_ptr<double[]> pow(data_t2 val, const data_t1* data,
// 	len_t len)
// {
// 	return map([val](data_t1 x) {return std::pow(val, x);}, data, len);
// }
// template<template <class...> class Container, class data_t1, class data_t2>
// static inline Container<double> pow(data_t2 val,
// 	const Container<data_t1>& data)
// {
// 	return map([val](data_t1 x) {return std::pow(val, x);}, data);
// }

// ================================ Copy

/** Copies src[0..len-1] to dest[0..len-1] */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void copy(const data_t* src, data_t* dest, len_t len) {
	std::copy(src, src+len, dest);
}
/** Returns a copy of the provided array */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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

// ================================================================
// V -> V
// ================================================================

#define WRAP_UNARY_FUNC_WITH_NAME(FUNC, NAME) \
\
template <class data_t, class data_t2, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME(data_t *RESTRICT data, len_t len, \
	data_t2 *RESTRICT out) \
{ \
	map([](data_t x){return FUNC(x);}, data, len, out); \
} \
template <class data_t, class len_t, REQUIRE_INT(len_t)> \
static inline void NAME ## _inplace (data_t* data, len_t len) { \
	map_inplace([](data_t x){return FUNC(x);}, data, len); \
} \
template <class data_t, class len_t, REQUIRE_INT(len_t)> \
static inline auto NAME(const data_t* data, len_t len) \
	-> unique_ptr<decltype(FUNC(*std::begin(data)))[]> \
{ \
	return map([](data_t x) {return FUNC(x);}, data, len); \
} \
template<template <class...> class Container, class data_t> \
static inline auto NAME(const Container<data_t>& data) \
	-> Container<decltype(FUNC(*std::begin(data)))> \
{ \
	return map([](data_t x) {return FUNC(x);}, data); \
} \

#define WRAP_UNARY_FUNC(FUNC) WRAP_UNARY_FUNC_WITH_NAME(FUNC, FUNC)

#define WRAP_UNARY_STD_FUNC_WITH_NAME(FUNC, NAME) \
using std::FUNC; \
WRAP_UNARY_FUNC_WITH_NAME(FUNC, NAME)

#define WRAP_UNARY_STD_FUNC(FUNC) WRAP_UNARY_STD_FUNC_WITH_NAME(FUNC, FUNC)

// ================================ Exp

// exponents and logs
WRAP_UNARY_FUNC(abs);
WRAP_UNARY_FUNC(sqrt);
WRAP_UNARY_STD_FUNC(cbrt);
WRAP_UNARY_STD_FUNC(exp);
WRAP_UNARY_STD_FUNC(exp2);
WRAP_UNARY_STD_FUNC(log);
WRAP_UNARY_STD_FUNC(log2);
WRAP_UNARY_STD_FUNC(log10);

// trig
WRAP_UNARY_STD_FUNC(sin);
WRAP_UNARY_STD_FUNC(asin);
WRAP_UNARY_STD_FUNC(sinh);
WRAP_UNARY_STD_FUNC(cos);
WRAP_UNARY_STD_FUNC(acos);
WRAP_UNARY_STD_FUNC(cosh);
WRAP_UNARY_STD_FUNC(tan);
WRAP_UNARY_STD_FUNC(atan);
WRAP_UNARY_STD_FUNC(tanh);

// err and gamma
WRAP_UNARY_STD_FUNC(erf);
WRAP_UNARY_STD_FUNC(erfc);
WRAP_UNARY_STD_FUNC_WITH_NAME(lgamma, log_gamma);
WRAP_UNARY_STD_FUNC_WITH_NAME(tgamma, gamma);

// rounding
WRAP_UNARY_STD_FUNC(ceil);
WRAP_UNARY_STD_FUNC(floor);
WRAP_UNARY_STD_FUNC(round);

// ================================ Reverse

template <class data_t, class len_t, REQUIRE_INT(len_t)> // TODO test this
static inline void reverse_inplace(const data_t* data, len_t len) {
	for (len_t i = 0; i < len / 2; i++) {
		std::swap(data[i], data[len-i-1]);
	}
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void reverse(const data_t *RESTRICT src, data_t *RESTRICT dest,
	len_t len)
{
	len_t j = len - 1;
	for (len_t i = 0; i < len; i++, j--) {
		dest[i] = src[j];
	}
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> reverse(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_reverse(data, ret, len);
	return ret;
}
template<template <class...> class Container, class data_t>
static inline Container<data_t> reverse(const Container<data_t>& data) {
	Container<data_t> ret(data.size());
	array_reverse(data, ret.data(), data.size());
	return ret;
}

// ================================ Resample

/** Writes the elements of src to dest such that
 * dest[i] = src[ floor(i*srcLen/destLen) ]; note that this function does no
 * filtering of any kind */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
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
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> resample(const data_t* data,
	len_t currentLen, len_t newLen)
{
	unique_ptr<data_t[]> ret(new data_t[newLen]);
	array_resample(data, ret, currentLen, newLen);
	return ret;
}
template<template <class...> class Container, class data_t, class len_t,
	REQUIRE_INT(len_t)>
static inline Container<data_t> resample(const Container<data_t>& data,
	len_t newLen)
{
	Container<data_t> ret(newLen);
	array_resample(data.data(), ret.data(), data.size(), newLen);
	return ret;
}

// ================================ Equality

/** Returns true if elements 0..(len-1) of x and y are equal, else false */
template <class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t)>
static inline bool all_eq(const data_t1 *x, const data_t2 *y, len_t len,
	float_t thresh=kDefaultNonzeroThresh)
{
	for (len_t i = 0; i < len; i++) {
		if (abs(x[i] - y[i]) > thresh) return false;
	}
	return true;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2, class float_t=double>
static inline bool all_eq(const Container1<data_t1>& x,
	const Container2<data_t2>& y, float_t thresh=kDefaultNonzeroThresh) {
	if (x.size() != y.size()) return 0;
	return all_eq(x.data(), y.data(), x.size());
}

// ================================ All

/** Returns true iff func(x[i]) is true for all i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool all(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (!func(x[i])) return false;
	}
	return true;
}
/** Returns true iff func(i, x[i]) is true for all i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool alli(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (!func(i, x[i])) return false;
	}
	return true;
}
/** Returns true iff x[i] is true for all i */
template <class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool all(const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (!x[i]) return false;
	}
	return true;
}
/** Returns true iff x[i] is true for all i */
template<template <class...> class Container1, class data_t1>
static inline bool all(const Container1<data_t1>& x) {
	return all(x.data(), x.size());
}

// ================================ Any

/** Returns true iff func(x[i]) is true for any i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool any(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (func(x[i])) return true;
	}
	return false;
}
/** Returns true iff func(i, x[i]) is true for any i */
template <class F, class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool anyi(const F&& func, const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (func(i, x[i])) return true;
	}
	return false;
}
/** Returns true iff x[i] is true for any i */
template <class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool any(const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (x[i]) return true;
	}
	return false;
}
/** Returns true iff x[i] is true for any i */
template<template <class...> class Container1, class data_t1>
static inline bool any(const Container1<data_t1>& x) {
	return any(x.data(), x.size());
}

// ================================ Nonnegativity
/** Returns true if elements 0..(len-1) of x are >= 0, else false */
template <class data_t1, class len_t, REQUIRE_INT(len_t)>
static inline bool all_nonnegative(const data_t1 *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (x[i] < 0) return false;
	}
	return true;
}
template<template <class...> class Container1, class data_t1>
static inline bool all_nonnegative(const Container1<data_t1>& x) {
	return all_nonnegative(x.data(), x.size());
}

// ================================ Positivity
/** Returns true if elements 0..(len-1) of x are > 0, else false */
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline bool all_positive(const data_t *x, len_t len) {
	for (len_t i = 0; i < len; i++) {
		if (x[i] <= 0) return false;
	}
	return true;
}
template<template <class...> class Container1, class data_t>
static inline bool all_positive(const Container1<data_t>& x) {
	return all_positive(x.data(), x.size());
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
// Normalizing
// ================================================================

#define WRAP_NORMALIZE_FUNC(FUNC) \
template <class data_t, class len_t, class float_t=double, REQUIRE_INT(len_t)> \
static inline unique_ptr<data_t[]> FUNC(const data_t* data, len_t len, \
	float_t nonzeroThresh=kDefaultNonzeroThresh) \
{ \
	unique_ptr<data_t[]> ret(new data_t[len]); \
	FUNC(data, len, ret, nonzeroThresh); \
	return ret; \
} \
template <template <class...> class Container, class data_t, \
	class float_t=double> \
static inline Container<data_t> FUNC(Container<data_t> data, \
	float_t nonzeroThresh=kDefaultNonzeroThresh) \
{ \
	Container<data_t> ret(data.size()); \
	FUNC(data.data(), data.size(), ret.data(), nonzeroThresh); \
	return ret; \
} \

// ------------------------ znormalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t)>
static inline bool znormalize(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	double mean, sse;
	knuth_sse_stats(data, len, mean, sse);
	double std = sqrt(sse / len);
	if (std < nonzeroThresh) {
		return false;
	}
	for (len_t i = 0; i < len; i++) {
		out[i] = static_cast<data_t2>((data[i] - mean) / std);
	}
	return true;
}
template<class data_t1, class len_t, class float_t=double, REQUIRE_INT(len_t)>
static inline bool znormalize_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	double mean, sse;
	knuth_sse_stats(data, len, mean, sse);
	double std = sqrt(sse / len);
	if (std < nonzeroThresh) {
		return false;
	}
	for (len_t i = 0; i < len; i++) {
		data[i] = static_cast<data_t1>((data[i] - mean) / std);
	}
	return true;
}
WRAP_NORMALIZE_FUNC(znormalize)

// ------------------------ mean normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t)>
static inline bool normalize_mean(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto avg = mean(data, len);
	sub(data, len, mean, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double, REQUIRE_INT(len_t)>
static inline bool normalize_mean_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto avg = mean(data, len);
	sub(data, len, mean);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_mean)

// ------------------------ std normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t)>
static inline bool normalize_stdev(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto std = stdev(data, len);
	if (std < nonzeroThresh) {
		return false;
	}
	div(data, len, std, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double, REQUIRE_INT(len_t)>
static inline bool normalize_stdev_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto std = stdev(data, len);
	if (std < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, std);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_stdev)

// ------------------------ L1 normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t)>
static inline bool normalize_L1(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sum(data, len);
	if (norm < nonzeroThresh) {
		return false;
	}
	div(data, len, norm, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double, REQUIRE_INT(len_t)>
static inline bool normalize_L1_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sum(data, len);
	if (norm < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, norm);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_L1)

// ------------------------ L2 normalize

template<class data_t1, class data_t2, class len_t, class float_t=double,
	REQUIRE_INT(len_t)>
static inline bool normalize_L2(data_t1 *RESTRICT data, len_t len,
	data_t2 *RESTRICT out, float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sqrt(sumsquares(data, len));
	if (norm < nonzeroThresh) {
		return false;
	}
	div(data, len, norm, out);
	return true;
}
template<class data_t1, class len_t, class float_t=double, REQUIRE_INT(len_t)>
static inline bool normalize_L2_inplace(data_t1* data, len_t len,
	float_t nonzeroThresh=kDefaultNonzeroThresh)
{
	auto norm = sqrt(sumsquares(data, len));
	if (norm < nonzeroThresh) {
		return false;
	}
	div_inplace(data, len, norm);
	return true;
}
WRAP_NORMALIZE_FUNC(normalize_L2)

// ================================================================
// Stringification / IO
// ================================================================

// ================================ Stringification

// ------------------------ with name

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static std::string to_string(const data_t *x, len_t len, const char* name="")
{
	std::ostringstream os;
	os.precision(3);
	if (name && name[0] != '\0') {
		os << name << ": ";
	}
	os << "[";
	for (len_t i = 0; i < len; i++) {
		os << x[i] << " ";
	}
	os << "]";
	return os.str();
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static std::string to_string(const data_t *x, len_t len, std::string name)
{
	return to_string(x, len, name.c_str());
}
template<template <class...> class Container, class data_t>
static inline std::string to_string(const Container<data_t>& data,
	const char* name="")
{
	return to_string(data.data(), data.size(), name);
}
template<template <class...> class Container, class data_t>
static inline std::string to_string(const Container<data_t>& data,
	std::string name)
{
	return to_string(data.data(), data.size(), name);
}

// ================================ Printing
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void print(const data_t *x, len_t len, const char* name="") {
	printf("%s\n", to_string(x, len, name).c_str());
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void print(const data_t *x, len_t len, std::string name) {
	printf("%s\n", to_string(x, len, name).c_str());
}

template<template <class...> class Container, class data_t>
static inline void print(const Container<data_t>& data, const char* name="") {
	print(data.data(), data.size(), name);
}
template<template <class...> class Container, class data_t>
static inline void print(const Container<data_t>& data, std::string name) {
	print(data.data(), data.size(), name);
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

	assertf(replace || (numPossibleVals >= howMany),
		"Can't sample %llu values without replacement between min %lld and max %lld",
		howMany, minVal, maxVal);

	if (replace) {
		for (size_t i = 0; i < howMany; i++) {
			int64_t val = (rand() % numPossibleVals) + minVal;
			ret.push_back(val);
		}
		return ret;
	}

	// sample without replacement; each returned int is unique
	unordered_map<int64_t, int64_t> possibleIdxs;
	int64_t idx;
	for (size_t i = 0; i < howMany; i++) {
		idx = (rand() % (numPossibleVals - i)) + minVal;

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

template<class float_t>
static inline vector<int64_t> rand_ints(int64_t minVal, int64_t maxVal, uint64_t howMany,
						 bool replace, const float_t* probs) {
	vector<int64_t> ret;
	int64_t numPossibleVals = maxVal - minVal + 1;

	assertf(numPossibleVals >= 1, "No values between min %lld and max %lld",
			minVal, maxVal);

	assertf(replace || (numPossibleVals >= howMany),
			"Can't sample %llu values without replacement between min %lld and max %lld",
			howMany, minVal, maxVal);

	assertf(all_nonnegative(probs), "Probabilities must be nonnegative!");
	auto totalProb = sum(probs, numPossibleVals);
	assertf(totalProb > 0, "Probabilities sum to a value <= 0");

    // init random distro object
	std::random_device rd;
    std::mt19937 gen(rd());
	auto possibleIdxsAr = range(minVal, maxVal+2); // end range at max+1
	std::piecewise_constant_distribution<float_t> distro(
		std::begin(possibleIdxsAr), std::end(possibleIdxsAr), probs);

	if (replace) {
		for (size_t i = 0; i < howMany; i++) {
			ret.push_back(static_cast<int64_t>(distro(gen)));
		}
		return ret;
	}

	// sample without replacement; each returned int is unique
	unordered_map<int64_t, int64_t> possibleIdxs;
	for (size_t i = 0; i < howMany; i++) {
		int64_t idx = static_cast<int64_t>(distro(gen));

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

template<template <class...> class Container, class float_t>
static inline vector<int64_t> rand_ints(int64_t minVal, int64_t maxVal, uint64_t howMany,
						 bool replace, const Container<float_t>& probs) {
	int64_t numPossibleVals = maxVal - minVal + 1;
	assertf(probs.size() == numPossibleVals,
		"Number of probabilities %llu doesn't match number of possible values %lld",
		probs.size(), numPossibleVals);
	return rand_ints(minVal, maxVal, howMany, replace, probs.data());
}

// ================================ Random Sampling

template<template <class...> class Container, class data_t>
static inline vector<data_t> rand_choice(const Container<data_t>& data, size_t howMany,
							   bool replace=false) {
	auto maxIdx = data.size() - 1;
	auto idxs = rand_ints(0, maxIdx, howMany, replace);
	return at_idxs(data, idxs);
}

template<template <class...> class Container1, class data_t,
	template <class...> class Container2, class float_t>
static inline vector<data_t> rand_choice(const Container1<data_t>& data, size_t howMany,
							   bool replace, const Container2<float_t>& probs) {
	auto maxIdx = data.size() - 1;
	auto idxs = rand_ints(0, maxIdx, howMany, replace, probs);
	return at_idxs(data, idxs);
}

// ================================ Random Data Generation

// ------------------------ iid gaussians

template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline void randn_inplace(const data_t* data, len_t len,
	float_t mean=0., float_t std=1) {
	assert(len > 0);

	// create normal distro object
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, std);

	for (len_t i = 0; i < len; i++) {
		data[i] = static_cast<data_t>(d(gen));
	}
}
// note that this must be called as, e.g., randn<double>(...)
template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline unique_ptr<data_t[]> randn(len_t len,
	float_t mean=0., float_t std=1)
{
	assert(len > 0);
	unique_ptr<data_t[]> ret(new data_t[len]);
	randn_inplace(ret.get(), len, mean, std);
	return ret;
}

// ------------------------ gaussian random walk

template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline void randwalk_inplace(const data_t* data, len_t len, float_t std=1) {
	assert(len > 0);

	// create normal distro object
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0., std);

    data[0] = static_cast<data_t>(d(gen));
	for (len_t i = 1; i < len; i++) {
		data[i] = data[i-1] + static_cast<data_t>(d(gen));
	}
}
// note that this must be called as, e.g., randwalk<double>(...)
template <class data_t, class len_t, REQUIRE_INT(len_t), class float_t=double>
static inline unique_ptr<data_t[]> randwalk(len_t len, float_t std=1)
{
	assert(len > 0);
	unique_ptr<data_t[]> ret(new data_t[len]);
	randwalk_inplace(ret.get(), len, std);
	return ret;
}

// ================================================================
// Miscellaneous
// ================================================================

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
