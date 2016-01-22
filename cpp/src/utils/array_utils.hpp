//
//  array_utils.c
//  edtw
//
//  Created By <Anonymous> on 1/14/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <memory>
#include <sstream>
#include "debug_utils.hpp"

using std::unique_ptr;
using std::vector;

//TODO massively refactor this to just use map() for basically everything

//TODO put everything in this namespace and then add using
//statement in files to unbreak them
//namespace ar {

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
void map_inplace(const F&& func, const data_t* data, len_t len) {
	for (len_t i = 0; i < len; i++) {
		data[i] = (data_t) func(data[i]);
	}
}
template <class F, class data_t, class len_t=size_t>
auto map(const F&& func, const data_t* data, len_t len)
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
auto map(const F&& func, const Container<Args...>& container)
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
auto map(const F&& func, const data_t1* x, const data_t2* y, len_t len)
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
auto map(const F&& func, const Container1<Args1...>& x, const Container2<Args2...>& y)
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
void mapi_inplace(const F&& func, const data_t* data, len_t len) {
	for (len_t i = 0; i < len; i++) {
		data[i] = (data_t) func(i, data[i]);
	}
}
template <class F, class data_t, class len_t=size_t>
auto mapi(const F&& func, const data_t* data, len_t len)
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
auto mapi(const F&& func, const Container<Args...>& container)
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
auto mapi(const F&& func, const data_t1* x, const data_t2* y, len_t len)
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
auto mapi(const F&& func, const Container1<Args1...>& x, const Container2<Args2...>& y)
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
Container<Args...> filter(const F&& func,
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
Container<Args...> filteri(const F&& func,
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
auto reduce(const F&& func, const data_t* data, len_t len)
	-> decltype(func(data[0], data[0]))
{
	if (len < 1) {
		return NULL;
	}
	if (len == 1) {
		// ideally we would just return the first element,
		// but it might not be the right type
		printf("WARNING: reduce(): called on array with 1 element; ");
		printf("reducing the first element with itself.");
		return func(data[0], data[0]);
	}

	auto total = func(data[0], data[1]);
	for (len_t i = 2; i < len; i++) {
		total = func(total, data[i]);
	}
	return total;
}
template<class F, template <class...> class Container, class... Args>
auto reduce(const F&& func, const Container<Args...>& container)
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
		printf("reducing the first element with itself.");
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
Container<size_t> where(const F&& func, const Container<Args...>& container) {
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
Container<size_t> wherei(const F&& func, const Container<Args...>& container) {
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

// ================================ at_idxs

/** note that this requires that the container implement operator[] */
template<template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
Container1<Args1...> at_idxs(const Container1<Args1...>& container,
		const Container2<Args2...>& indices,
		bool boundsCheck=true) {
	Container1<Args1...> ret;
	size_t j = 0;
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

/** Fills the array with a sequence of values; equivalent to Python
 * range(startVal, stopVal, step), or MATLAB startVal:step:stopVal */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_range(data_t startVal, data_t stopVal, data_t step=1) {
	assertf( (stopVal - startVal) * step > 0,
			"ERROR: array_sequence: invalid args min=%.3f, max=%.3f, step=%.3f\n",
			startVal, stopVal, step);

	//allocate a new array
	len_t len = (len_t) floor( (stopVal - startVal) / step ) + 1;
	unique_ptr<data_t[]> data(new data_t[len]);

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
data_t** array_split(const data_t* data, len_t len, len_t newNumDims) {
	size_t newArraysLen = len / newNumDims;

	if ( newArraysLen * newNumDims != len) {
		printf("WARNING: array_reshape: newNumDims %d is not factor of array length %d",
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
data_t array_max(const data_t *data, len_t len) {
	data_t max = -INFINITY;
	for (len_t i = 0; i < len; i++) {
		if (data[i] > max) {
			max = data[i];
		}
	}
	return max;
}
/** Returns the maximum value in data[0..len-1] */
template<template <class...> class Container, class data_t>
data_t array_max(const Container<data_t>& data) {
	return array_max(&data[0], data.size());
}

// ================================ Min

/** Returns the minimum value in data[0..len-1] */
template <class data_t, class len_t=size_t>
data_t array_min(const data_t *data, len_t len) {
	data_t min = INFINITY;
	for (len_t i = 0; i < len; i++) {
		if (data[i] < min) {
			min = data[i];
		}
	}
	return min;
}
/** Finds the minimum of the elements in data */
template<template <class...> class Container, class data_t>
data_t array_min(const Container<data_t>& data) {
	return array_min(&data[0], data.size());
}

// ================================ Sum

/** Computes the sum of data[0..len-1] */
template <class data_t, class len_t=size_t>
data_t array_sum(const data_t *data, len_t len) {
	return reduce([](data_t x, data_t y){ return x+y;}, data, len);
}
/** Computes the sum of the elements in data */
// template <class data_t>
// data_t array_sum(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
data_t array_sum(const Container<data_t>& data) {
	return reduce([](data_t x, data_t y){ return x+y;}, data);
}

// ================================ Sum of Squares

/** Computes the sum of data[i]^2 for i = [0..len-1] */
template <class data_t, class len_t=size_t>
data_t array_sumsquares(const data_t *data, len_t len) {
	data_t sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += data[i]*data[i];
	}
	return sum;
}
/** Computes the sum of data[i]^2 for i = [0..len-1] */
template<template <class...> class Container, class data_t>
data_t array_sumsquares(const Container<data_t>& data) {
	return array_sumsquares(&data[0], data.size());
}

// ================================ Mean

/** Computes the arithmetic mean of data[0..len-1] */
template <class data_t, class len_t=size_t>
double array_mean(const data_t* data, len_t len) {
	return array_sum(data, len) / ((double) len);
}
/** Computes the arithmetic mean of data[0..len-1] */
// template <class data_t>
// data_t array_mean(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
double array_mean(const Container<data_t>& data) {
	return array_sum(data) / ((double) data.size());
}

// ================================ Variance

/** Computes the population variance of data[0..len-1] */
template <class data_t, class len_t=size_t>
double array_variance(const data_t *data, len_t len) {
	if (len <= 1) {
		if (len < 1) {
			printf("WARNING: array_variance(): received length %lu, returning 0",
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
// data_t array_variance(const vector<data_t>& data) {
template<template <class...> class Container, class data_t>
double array_variance(const Container<data_t>& data) {
	return array_variance(&data[0], data.size());
}

// ================================ Standard deviation

/** Computes the population standard deviation of data[0..len-1] */
template <class data_t, class len_t=size_t>
double array_std(const data_t *data, len_t len) {
	return sqrt(array_variance(data,len));
}

/** Computes the population standard deviation of data[0..len-1] */
template<template <class...> class Container, class data_t>
double array_std(const Container<data_t>& data) {
	return sqrt(array_variance(data));
}

// ================================================================
// V x V -> R
// ================================================================

// ================================ Dot Product
/** Returns the the dot product of x and y */
template <class data_t, class len_t=size_t>
data_t array_dot(const data_t* x, const data_t* y, len_t len) {
	data_t sum = 0;
	for (len_t i = 0; i < len; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
double array_dot(const Container1<data_t1>& x, const Container2<data_t2>& y) {
	assert(x.size() == y.size());
	return array_dot(&x[0],&y[0],x.size());
}

// ================================================================
// Cumulative Statistics (V[1:i] -> R[i])
// ================================================================

// ================================ Cumulative Sum

/** Cumulative sum of elements in src, storing the result in dest */
template <class data_t, class len_t=size_t>
void array_cum_sum(const data_t* src, data_t* dest, len_t len) {
	dest[0] = src[0];
	for (len_t i=1; i < len; i++) {
		dest[i] = src[i] + dest[i-1];
	}
}
/** Returns a new array composed of the cumulative sum of the data */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_cumsum(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cumsum(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative sum of the data */
template<template <class...> class Container, class data_t>
Container<data_t> array_cumsum(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cumsum(&data[0],&ret[0],data.size());
	return ret;
}

// ================================ Cumulative Mean

/** Cumulative mean of elements in src, storing the result in dest */
template <class data_t, class len_t=size_t>
void array_cummean(const data_t* src, data_t* dest, len_t len) {
	double sum = 0;
	for (len_t i=0; i < len; i++) {
		sum += src[i];
		dest[i] = sum / (i+1);
	}
}
/** Returns a new array composed of the cumulative mean of the data */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_cummean(data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cummean(data, ret, len);
	return ret;
}
/** Returns a new array composed of the cumulative mean of the data */
template<template <class...> class Container, class data_t>
Container<data_t> array_cummean(const Container<data_t>& data) {
	Container<data_t> ret{data.size()};
	array_cummean(&data[0],&ret[0],data.size());
	return ret;
}

// ================================ Cumulative SSE

/** Cumulative SSE of elements in src, storing the result in dest */
template <class data_t, class len_t=size_t>
void array_cumsxx(const data_t* src, data_t* dest, len_t len) {
	if (len < 1) {
		printf("WARNING: array_cumsxx(): received length %lu, returning 0",
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
unique_ptr<data_t[]> array_cumsxx(data_t *data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_cumsxx(data, ret, len);
	return ret;
}
/** Returns the sum of squared differences from the mean of data[0..i] */
template<template <class...> class Container, class data_t>
Container<data_t> array_cumsxx(const Container<data_t>& data) {
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
void array_add(const data_t1* x, const data_t2* y, data_t3* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] + y[i];
	}
}
/** Returns a new array composed of elementwise x + y */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_add(const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(x[0]+y[0])[]>
{
	return map([](data_t1 a, data_t2 b){ return a + b;}, x, y, len);
}
/** Returns a new array composed of elementwise x + y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
auto array_add(const Container1<data_t1>& x, const Container2<data_t2>& y)
	-> Container1<decltype(x[0]+y[0])>
{
	return map([](data_t1 a, data_t2 b){ return a + b;}, x, y);
}

// ================================ Subtract

/** Elementwise x - y, storing the result in dest */
template <class data_t1, class data_t2, class data_t3, class len_t=size_t>
void array_sub(const data_t1* x, const data_t2* y, data_t3* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] - y[i];
	}
}
/** Returns a new array composed of elementwise x - y */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_sub(const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(x[0]+y[0])[]>
{
	return map([](data_t1 a, data_t2 b){ return a - b;}, x, y, len);
}
/** Returns a new array composed of elementwise x - y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
auto array_sub(const Container1<data_t1>& x, const Container2<data_t2>& y)
	-> Container1<decltype(x[0]+y[0])>
{
	return map([](data_t1 a, data_t2 b){ return a - b;}, x, y);
}

// ================================ Multiply

/** Elementwise x * y, storing the result in dest */
template <class data_t1, class data_t2, class data_t3, class len_t=size_t>
void array_mul(const data_t1* x, const data_t2* y, data_t3* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] * y[i];
	}
}
/** Returns a new array composed of elementwise x * y */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_mul(const data_t1* x, const data_t2* y, len_t len)
	-> unique_ptr<decltype(x[0]*y[0])[]>
{
	return map([](data_t1 a, data_t2 b){ return a * b;}, x, y, len);
}
/** Returns a new array composed of elementwise x * y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
auto array_mul(const Container1<data_t1>& x, const Container2<data_t2>& y)
	-> Container1<decltype(x[0]+y[0])>
{
	return map([](data_t1 a, data_t2 b){ return a * b;}, x, y);
}

// ================================ Divide
//
// TODO decide if we like forcing everything to be a double for this

/** Elementwise x / y, storing the result in dest */
template <class data_t1, class data_t2, class len_t=size_t>
void array_div(const data_t1* x, const data_t2* y, double* dest, len_t len) {
	for (len_t i = 0; i < len; i++) {
		dest[i] = x[i] / (double) y[i];
	}
}
/** Returns a new array composed of elementwise x / y */
template <class data_t1, class data_t2, class len_t=size_t>
unique_ptr<double[]> array_div(const data_t1* x, const data_t2* y, len_t len) {
	return map([](data_t1 a, data_t2 b){ return (double)a / b;}, x, y, len);
}
/** Returns a new array composed of elementwise x / y */
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
Container1<double> array_div(const Container1<data_t1>& x,
	const Container2<data_t2>& y)
{
	return map([](data_t1 a, data_t2 b){ return (double) a / b;}, x, y);
}

// ================================ Concatenate

template <class data_t, class len_t1=size_t, class len_t2=size_t>
unique_ptr<data_t[]> array_concat(const data_t* x, const data_t* y,
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
Container1<data_t> array_concat(const Container1<data_t>& x,
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
void array_adds_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x + val;}, data, len);
}
/** Returns a new array composed of (data[i] + val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_adds(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x + val;}, data, len);
}
/** Returns a new vector composed of (data[i] + val) for all i */
// template <class data_t>
// vector<data_t> array_add(const vector<data_t>& data, data_t val) {
template<template <class...> class Container, class data_t1, class data_t2>
auto array_adds(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x + val;}, data);
}

// ================================ Subtract, scalar

/** Adds each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
void array_subs_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x - val;}, data, len);
}
/** Returns a new array composed of (data[i] + val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_subs(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x - val;}, data, len);
}
/** Returns a new vector composed of (data[i] + val) for all i */
template<template <class...> class Container, class data_t1, class data_t2>
auto array_subs(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x - val;}, data);
}

// ================================ Multiply, scalar

/** Multiplies each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
void array_muls_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x * val;}, data, len);
}
/** Returns a new array composed of (data[i] * val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_muls(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x * val;}, data, len);
}
/** Returns a new vector composed of (data[i] * val) for all i */
template<template <class...> class Container, class data_t1, class data_t2>
auto array_muls(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x * val;}, data);
}

// ================================ Divide, scalars

/** Divides each element in data[0..len-1] by the scalar val */
template <class data_t1, class data_t2, class len_t=size_t>
void array_divs_inplace(data_t1* data, data_t2 val, len_t len) {
	map_inplace([val](data_t1 x){return x / val;}, data, len);
}
/** Returns a new array composed of (data[i] / val) for all i */
template <class data_t1, class data_t2, class len_t=size_t>
auto array_divs(const data_t1* data, data_t2 val, len_t len)
	-> unique_ptr<decltype(data[0] + val)[]>
{
	return map([val](data_t1 x) {return x / val;}, data, len);
}
/** Returns a new vector composed of (data[i] / val) for all i */
template<template <class...> class Container, class data_t1, class data_t2>
auto array_divs(const Container<data_t1>& data, data_t2 val)
	-> Container<decltype(*begin(data) + val)>
{
	return map([val](data_t1 x) {return x / val;}, data);
}

// ================================ Copy

/** Copies src[0..len-1] to dest[0..len-1] */
template <class data_t, class len_t=size_t>
void array_copy_inplace(const data_t* src, data_t* dest, len_t len) {
	std::copy(src, src+len, dest);
}
/** Returns a copy of the provided array */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_copy(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	std::copy(data, data+len, ret);
	return ret;
}
/** Returns a copy of the provided array */
template<template <class...> class Container, class data_t>
Container<data_t> array_copy(const Container<data_t>& data) {
	Container<data_t> ret(data);
	return ret;
}

// ================================ Reverse

/** Copies src[0..len-1] to dest[len-1..0] */
template <class data_t, class len_t=size_t>
void array_reverse(const data_t *src, data_t *dest, len_t len) {
	len_t j = len - 1;
	for (len_t i = 0; i < len; i++, j--) {
		dest[i] = src[j];
	}
}
/** Returns data[len-1..0] */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_reverse(const data_t* data, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_reverse(data, ret, len);
	return ret;
}
/** Returns data[len-1..0] */
template<template <class...> class Container, class data_t>
Container<data_t> array_add(const Container<data_t>& data) {
	Container<data_t> ret(data.size());
	array_reverse(data, &ret[0], data.size());
	return ret;
}

// ================================ Create constant array

/** Sets each element of the array to the value specified */
template <class data_t1, class data_t2, class len_t=size_t>
void array_set_to_constant(data_t1 *x, data_t2 value, len_t len) {
	for (len_t i = 0; i < len; i++) {
		x[i] = value;
	}
}
template<template <class...> class Container,
	class data_t1, class data_t2>
void array_set_to_constant(const Container<data_t1>& data, data_t2 value) {
	array_set_to_constant(&data[0], value, data.size());
}

/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_constant(data_t value, len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	array_set_to_constant(ret, value, len);
	return ret;
}
/** Returns an array of length len with all elements equal to value */
template <class data_t, class len_t=size_t>
unique_ptr<data_t[]> array_constant_vect(data_t value, len_t len) {
	vector<data_t> ret(len, value);
	return ret;
}

// ================================ Resample

/** Writes the elements of src to dest such that
 * dest[i] = src[ floor(i*srcLen/destLen) ]; note that this function does no
 * filtering of any kind */
template <class data_t, class len_t=size_t>
void array_resample(const data_t *src, data_t *dest,
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
unique_ptr<data_t[]> array_resample(const data_t* data,
	len_t currentLen, len_t newLen)
{
	unique_ptr<data_t[]> ret(new data_t[newLen]);
	array_resample(data, ret, currentLen, newLen);
	return ret;
}
// template <class data_t, class len_t=size_t>
// vector<data_t> array_resample(const vector<data_t>& data, len_t newLen) {
template<template <class...> class Container,
	class data_t, class len_t>
Container<data_t> array_resample(const Container<data_t>& data, len_t newLen) {
	Container<data_t> ret(newLen);
	array_resample(&data[0], &ret[0], data.size(), newLen);
	return ret;
}

// ================================ Equality

/** Returns true if elements 0..(len-1) of x and y are equal, else false */
template <class data_t1, class data_t2, class len_t=size_t>
bool array_equal(const data_t1 *x, const data_t2 *y, len_t len) {
	for (len_t i = 0; i < len; i++) {
		//TODO define as a const somewhere
		if (std::fabs(x[i] - y[i]) > .00001) return false;
	}
	return true;
}
//template <class Ptr1, class Ptr2, class data_t1, class data_t2,
//	class len_t=size_t>
//bool array_equal(const Ptr1<data_t1> x, const Ptr2<data_t2> y, len_t len) {
//	for (len_t i = 0; i < len; i++) {
//		//TODO define as a const somewhere
//		if ( fabs(x[i] - y[i]) > .00001 ) return false;
//	}
//	return true;
//}
template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
bool array_equal(const Container1<data_t1>& x, const Container2<data_t2>& y) {
	if (x.size() != y.size()) return 0;
	return array_equal(&x[0], &y[0], x.size());
}

// ================================================================
// Stringification / IO
// ================================================================

// ================================ Stringification
std::string array_to_string(const float *x, size_t len) {
	std::ostringstream os;
	os.precision(3);
	os << "[";
	for (size_t i = 0; i < len; i++) {
		os << x[i] << " ";
	}
	os << "]";
	return os.str();
}
std::string array_to_string(const double *x, size_t len) {
	std::ostringstream os;
	os.precision(3);
	os << "[";
	for (size_t i = 0; i < len; i++) {
		os << x[i] << " ";
	}
	os << "]";
	return os.str();
}
std::string array_to_string(const char *x, size_t len) {
	return std::string(x);
}
template <class data_t, class len_t=size_t>
std::string array_to_string(const data_t *x, len_t len) {
	std::ostringstream os;
	os << "[";
	for (len_t i = 0; i < len; i++) {
		os << x[i] << " ";
	}
	os << "]";
	return os.str();
}

template<template <class...> class Container, class data_t>
std::string array_to_string(const Container<data_t>& data) {
	return array_to_string(&data[0], data.size());
}

// ================================ Printing
template <class data_t, class len_t=size_t>
void array_print(const data_t *x, len_t len) {
	printf("%s\n", array_to_string(x, len));
}
template<template <class...> class Container, class data_t>
double array_print(const Container<data_t>& data) {
	array_print(&data[0], data.size());
}

template <class data_t, class len_t=size_t>
void array_print_with_name(const data_t *data, len_t len, const char* name) {
	printf("%s:\t%s\n", name, array_to_string(data, len));
}
template<template <class...> class Container, class data_t>
void array_print_with_name(const Container<data_t>& data, const char* name) {
	array_print_with_name(&data[0], data.size(), name);
}
