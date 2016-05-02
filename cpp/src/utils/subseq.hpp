//
//  subseq.hpp
//
//  Created By Davis Blalock on 3/10/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __DIG_SUBSEQ_HPP
#define __DIG_SUBSEQ_HPP

// #include <stdlib.h> // for size_t
#include <deque>
#include <memory>

#include "array_utils.hpp"

using std::begin;
using std::end;
using std::unique_ptr;

using ar::abs;
using ar::dot;
using ar::dist_L1;
using ar::dist_L2;
using ar::dist_sq;

// using ar::length_t;

namespace subs {

typedef int64_t length_t;

// ================================ Map

// note that only nonnegative strides are supported

// ------------------------ raw arrays, no query, with output array

template<class F, class data_t2, class data_t3>
static inline void mapSubseqs(const F&& func, length_t m, const data_t2* x,
	length_t n, data_t3* out, length_t inStride=1, length_t outStride=1)
{
	assert(n >= m);
	for (length_t i = 0; i < n - m + 1; i += inStride) {
		out[i * outStride] = func(x+i);
//		out[i * outStride] = static_cast<data_t3>(func(x+i));
	}
}

template<class F, class data_t2, class data_t3>
static inline void mapiSubseqs(const F&& func, length_t m, const data_t2* x,
	length_t n, data_t3* out, length_t inStride=1, length_t outStride=1)
{
	assert(n >= m);
	for (length_t i = 0; i < n - m + 1; i += inStride) {
		out[i * outStride] = func(i, x+i);
	}
}

// ------------------------ raw arrays, with output array

template<class F, class data_t1, class data_t2, class data_t3>
static inline void mapSubseqs(const F&& func, const data_t1* q, length_t m,
	const data_t2* x, length_t n, data_t3* out, length_t stride=1)
{
	assert(n >= m);
	for (length_t i = 0; i < n - m + 1; i += stride) {
		out[i] = func(q, x+i);
	}
}

template<class F, class data_t1, class data_t2, class data_t3>
static inline void mapiSubseqs(const F&& func, const data_t1* q, length_t m,
	const data_t2* x, length_t n, data_t3* out, length_t stride=1)
{
	assert(n >= m);
	for (length_t i = 0; i < n - m + 1; i += stride) {
		out[i] = func(i, q, x+i);
	}
}

// ------------------------ raw arrays with new array allocated, no query

template<class F, class data_t2>
static inline auto mapSubseqs(const F&& func, length_t m,
	const data_t2* x, length_t n, length_t stride=1)
	-> unique_ptr<decltype(func(x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(x))[]> ret(new decltype(func(x))[l]);
	for (length_t i = 0; i < l; i += stride) {
		ret[i] = func(x+i);
	}
	return ret;
}

template<class F, class data_t1, class data_t2>
static inline auto mapiSubseqs(const F&& func, const data_t1* q, length_t m,
	const data_t2* x, length_t n, length_t stride=1)
	-> unique_ptr<decltype(func(0, x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(0, x))[]> ret(new decltype(func(0, x))[l]);
	for (length_t i = 0; i < l; i += stride) {
		ret[i] = func(i, x+i);
	}
	return ret;
}

// ------------------------ raw arrays with new array allocated

template<class F, class data_t1, class data_t2>
static inline auto mapSubseqs(const F&& func, const data_t1* q, length_t m,
	const data_t2* x, length_t n, length_t stride=1)
	-> unique_ptr<decltype(func(q, x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(q, x))[]> ret(new decltype(func(q, x))[l]);
	for (length_t i = 0; i < l; i += stride) {
		ret[i] = func(q, x+i);
	}
	return ret;
}

template<class F, class data_t1, class data_t2>
static inline auto mapiSubseqs(const F&& func, const data_t1* q, length_t m,
	const data_t2* x, length_t n, length_t stride=1)
	-> unique_ptr<decltype(func(0, q, x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(0, q, x))[]> ret(new decltype(func(0, q, x))[l]);
	for (length_t i = 0; i < l; i += stride) {
		ret[i] = func(i, q, x+i);
	}
	return ret;
}

// ------------------------ containers with new container allocated

template<class F, template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static auto mapSubseqs(const F&& func, const Container1<data_t1>& q,
					   const Container2<data_t2>& x, length_t stride=1)
	-> Container2<decltype(func(&q[0], &x[0]))>
{
	auto m = q.size();
	auto n = x.size();
	assert(n >= m);
	auto l = n - m + 1;
	Container2<decltype( func(&q[0], &x[0]) )> ret;
	for (length_t i = 0; i < l; i += stride) {
		ret.emplace_back( func(&q[0], &x[i]) );
	}
	return ret;
}

template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2>
static auto mapiSubseqs(const F&& func, const Container1<Args1...>& q,
	const Container2<Args2...>& x, length_t stride=1)
	-> Container2<decltype(func(0, begin(q), begin(x)))>
{
	auto m = q.size();
	auto n = x.size();
	assert(n >= m);
	auto l = n - m + 1;
	Container2<decltype(func(0, begin(q), begin(x)))> ret;
	for (length_t i = 0; i < l; i += stride) {
		ret.emplace_back( func(i, &q[0], &x[i]) );
	}
	return ret;
}

// ================================ Cross Correlation

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3>
static inline void crossCorrs(const data_t1* q, length_t m, const data_t2* x,
	length_t n, data_t3* out, length_t inStride=1, length_t outStride=1)
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dot(q, x, m);
	}, m, x, n, out, inStride, outStride);
}

template<class data_t1, class data_t2>
static inline auto crossCorrs(const data_t1* q, length_t m, const data_t2* x,
	length_t n, length_t stride=1)
	-> unique_ptr<decltype(q[0] * x[0])[]>
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dot(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static auto crossCorrs(const Container1<data_t1>& q,
	const Container2<data_t2>& x, length_t stride=1)
	-> Container2<decltype(q[0] * x[0])>
{
	// NOTE: this will cryptically fail to compile and complain about
	// being unable to match the lambda if the pointers aren't const
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dot(_q, _x, m);
	}, q, x, stride);
}

// ================================ L1 distance

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3>
static inline void dists_L1(const data_t1* q, length_t m, const data_t2* x,
	length_t n, data_t3* out, length_t stride=1)
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dist_L1(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2>
static inline auto dists_L1(const data_t1* q, length_t m, const data_t2* x,
	length_t n, length_t stride=1)
	-> unique_ptr<decltype(q[0] - x[0])[]>
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dist_L1(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static auto dists_L1(const Container1<data_t1>& q,
	const Container2<data_t2>& x, length_t stride=1)
	-> Container2<decltype(q[0] - x[0])>
{
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dist_L1(_q, _x, m);
	}, q, x, stride);
}

// ================================ L2 distance

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3>
static inline void dists_L2(const data_t1* q, length_t m, const data_t2* x,
	length_t n, data_t3* out, length_t stride=1)
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dist_L2(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2>
static inline auto dists_L2(const data_t1* q, length_t m, const data_t2* x,
	length_t n, length_t stride=1)
	-> unique_ptr<decltype(q[0] - x[0])[]>
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dist_L2(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static auto dists_L2(const Container1<data_t1>& q,
	const Container2<data_t2>& x, length_t stride=1)
	-> Container2<decltype(q[0] - x[0])>
{
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dist_L2(_q, _x, m);
	}, q, x, stride);
}

// ================================ L2^2 distance

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3>
static inline void dists_sq(const data_t1* q, length_t m, const data_t2* x,
	length_t n, data_t3* out, length_t stride=1)
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dist_sq(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2>
static inline auto dists_sq(const data_t1* q, length_t m, const data_t2* x,
	length_t n, length_t stride=1)
	-> unique_ptr<decltype(q[0] * x[0])[]>
{
	return mapSubseqs([m, q](const data_t2* x) {
		return dist_sq(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2>
static auto dists_sq(const Container1<data_t1>& q,
	const Container2<data_t2>& x, length_t stride=1)
	-> Container2<decltype(q[0] - x[0])>
{
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dist_sq(_q, _x, m);
	}, q, x, stride);
}

// ================================ 1st discrete derivative

// ------------------------ raw arrays

template<class data_t2, class data_t3>
static inline void first_derivs(const data_t2* x, length_t n, data_t3* out,
	length_t stride=1)
{
	data_t2 q[2] = {-1, 1};
	return crossCorrs(q, 2, x, n, out, stride);
}

template<class data_t2>
static inline unique_ptr<data_t2[]> first_derivs(const data_t2* x, length_t n,
	length_t stride=1)
{
	data_t2 q[2] = {-1, 1};
	return crossCorrs(q, 2, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container2, class data_t2>
static Container2<data_t2> first_derivs(const Container2<data_t2>& x,
	length_t stride=1)
{
	vector<data_t2> q {-1, 1};
	return crossCorrs(q, x, stride);
}


// ================================ Max subarray

template<class data_t>
static void maximum_subarray(const data_t* data,
	length_t len, length_t& start_best, length_t& end_best,
	length_t minSpacing=1)
{
	data_t sum_current = 0, sum_best = 0;
	length_t start_current = 0;
	start_best = 0;
	end_best = 0;
	for (int i = 0; i < len; i++) {
		auto val = data[i];
		if (sum_current + val > 0) {
			sum_current += val;
		} else { // reset start position
			sum_current = 0;
			start_current = i + 1;
		}

		if (sum_current > sum_best) { // check if new best-so-far
			sum_best = sum_current;
			start_best = start_current;
			end_best = i + 1; // non-inclusive end
		}
	}
}

template<class F, class data_t>
static inline std::pair<length_t, length_t> maximum_subarray(const data_t* data,
	length_t len, length_t minSpacing=1)
{
	length_t start_best = 0, end_best = 0;
	maximum_subarray(data, len, start_best, end_best, minSpacing);
	return std::pair<length_t, length_t>(start_best, end_best);
}
template<template<class...> class Container, class data_t>
static inline std::pair<length_t, length_t> maximum_subarray(
	const Container<data_t>& data, length_t minSpacing=1)
{
	return maximum_subarray(data.data(), data.size(), minSpacing);
}

// ================================ Relative extrema
// i.e., the set of indices i such that |i - j| <= m -> (x[i] > x[j]) for some m.

// generic func for relative maxima or minima; pass in lambda to get
// relative extrema
template<class F, class data_t>
static vector<length_t> local_optima(const F&& func, const data_t* data,
	length_t len, length_t minSpacing=1)
{
	vector<length_t> idxs;
	if (len <= 0) {
		return idxs;
	}

	length_t candidateIdx = 0;
	data_t candidateVal = data[0];

	for(length_t idx = 1; idx < len; idx++) {
		data_t val = data[idx];
		// if no candidate
		if (candidateIdx == -1) {
			// check if this point is eligible and make it
			// the candidate if so
			if (func(val, data[idx-1])) {
			// if (val >= data[idx-1]) { // TODO remove
				// printf("val %g at idx %lld > prev val %g \n", val, idx, data[idx-1]);
				candidateIdx = idx;
				candidateVal = val;
			}
		// there's a candidate, and this idx is within minSpacing of it
		} else if ((idx - candidateIdx) <= minSpacing) {
			if (func(val, candidateVal)) {
			// if (val >= candidateVal) {
				// printf("%lld) setting candidate val to %g cuz > prev val %g \n", idx, val, data[idx-1]);
				candidateIdx = idx;
				candidateVal = val;
			}
		} else { // no overlap, so flush candidate
			// printf("outputting candidate %lld", candidateIdx);
			idxs.push_back(candidateIdx);
			// set current point as new candidate
			bool isLocalOptimum = func(val, data[idx-1]);
			if (isLocalOptimum) {
				// printf("and setting new candidate %lld\n", idx);
				candidateIdx = idx;
				candidateVal = val;
			} else {
				// printf("and resetting\n");
				candidateIdx = -1;
				candidateVal = 0; // arbitrary value
			}
		}
	}
	if (candidateIdx >= 0) { // add final candidate idx, if there's a candidate
		idxs.push_back(candidateIdx);
	}
	return idxs;
}

// note that if there are multiple copies of the same val within minSpacing of
// each other, the behavior is undefined; what actually happens at present is
// that the first index among the tied values is returned
template<class data_t>
static vector<length_t> local_maxima(const data_t* data, length_t len,
		length_t minSpacing=1)
{
	return local_optima([](data_t val, data_t candidateVal) {
		return val >= candidateVal; // bool for when new candidate is better
	}, data, len, minSpacing);
}
template<class data_t>
static vector<length_t> local_minima(const data_t* data, length_t len,
		length_t minSpacing=1)
{
	return local_optima([](data_t val, data_t candidateVal) {
		return val <= candidateVal; // bool for when new candidate is better
	}, data, len, minSpacing);
}

template<template<class...> class Container, class data_t>
static vector<length_t> local_maxima(const Container<data_t>& data,
		length_t minSpacing=1)
{
	return local_maxima(data.data(), data.size(), minSpacing);
}
template<template<class...> class Container, class data_t>
static vector<length_t> local_minima(const Container<data_t>& data,
		length_t minSpacing=1)
{
	return local_minima(data.data(), data.size(), minSpacing);
}


} // namespace subs

#endif

















