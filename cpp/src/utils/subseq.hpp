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

using ar::dot;
using ar::dist_L1;
using ar::dist_L2;
using ar::dist_sq;

namespace subs {

// ================================ Map

// note that only nonnegative strides are supported

// ------------------------ raw arrays, no query, with output array

template<class F, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void mapSubseqs(const F&& func, len_t1 m, const data_t2* x,
	len_t2 n, data_t3* out, len_t3 stride=1)
{
	assert(n >= m);
	for (size_t i = 0; i < n - m; i += stride) {
		out[i] = func(x+i);
	}
}

template<class F, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void mapiSubseqs(const F&& func, len_t1 m, const data_t2* x,
	len_t2 n, data_t3* out, len_t3 stride=1)
{
	assert(n >= m);
	for (size_t i = 0; i < n - m; i += stride) {
		out[i] = func(i, x+i);
	}
}

// ------------------------ raw arrays, with output array

template<class F, class data_t1, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void mapSubseqs(const F&& func, const data_t1* q, len_t1 m,
	const data_t2* x, len_t2 n, data_t3* out, len_t3 stride=1)
{
	assert(n >= m);
	for (size_t i = 0; i < n - m; i += stride) {
		out[i] = func(q, x+i);
	}
}

template<class F, class data_t1, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void mapiSubseqs(const F&& func, const data_t1* q, len_t1 m,
	const data_t2* x, len_t2 n, data_t3* out, len_t3 stride=1)
{
	assert(n >= m);
	for (size_t i = 0; i < n - m; i += stride) {
		out[i] = func(i, q, x+i);
	}
}

// ------------------------ raw arrays with new array allocated, no query

template<class F, class data_t2, class len_t1=size_t,
	class len_t2=size_t, class len_t3=size_t>
static inline auto mapSubseqs(const F&& func, len_t1 m,
	const data_t2* x, len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(func(x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(x))[]> ret(new decltype(func(x))[l]);
	for (size_t i = 0; i < l; i += stride) {
		ret[i] = func(x+i);
	}
	return ret;
}

template<class F, class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto mapiSubseqs(const F&& func, const data_t1* q, len_t1 m,
	const data_t2* x, len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(func(0, x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(0, x))[]> ret(new decltype(func(0, x))[l]);
	for (size_t i = 0; i < l; i += stride) {
		ret[i] = func(i, x+i);
	}
	return ret;
}

// ------------------------ raw arrays with new array allocated

template<class F, class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto mapSubseqs(const F&& func, const data_t1* q, len_t1 m,
	const data_t2* x, len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(func(q, x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(q, x))[]> ret(new decltype(func(q, x))[l]);
	for (size_t i = 0; i < l; i += stride) {
		ret[i] = func(q, x+i);
	}
	return ret;
}

template<class F, class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto mapiSubseqs(const F&& func, const data_t1* q, len_t1 m,
	const data_t2* x, len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(func(0, q, x))[]>
{
	auto l = n - m + 1;
	unique_ptr<decltype(func(0, q, x))[]> ret(new decltype(func(0, q, x))[l]);
	for (size_t i = 0; i < l; i += stride) {
		ret[i] = func(i, q, x+i);
	}
	return ret;
}

// ------------------------ containers with new container allocated

template<class F, template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2, class len_t3=size_t>
static auto mapSubseqs(const F&& func, const Container1<data_t1>& q,
					   const Container2<data_t2>& x, len_t3 stride=1)
	-> Container2<decltype(func(&q[0], &x[0]))>
{
	auto m = q.size();
	auto n = x.size();
	assert(n >= m);
	auto l = n - m + 1;
	Container2<decltype( func(&q[0], &x[0]) )> ret;
	for (size_t i = 0; i < l; i += stride) {
		ret.emplace_back( func(&q[0], &x[i]) );
	}
	return ret;
}

template<class F, template <class...> class Container1, class... Args1,
	template <class...> class Container2, class... Args2, class len_t3=size_t>
static auto mapiSubseqs(const F&& func, const Container1<Args1...>& q,
	const Container2<Args2...>& x, len_t3 stride=1)
	-> Container2<decltype(func(0, begin(q), begin(x)))>
{
	auto m = q.size();
	auto n = x.size();
	assert(n >= m);
	auto l = n - m + 1;
	Container2<decltype(func(0, begin(q), begin(x)))> ret;
	for (size_t i = 0; i < l; i += stride) {
		ret.emplace_back( func(i, &q[0], &x[i]) );
	}
	return ret;
}

// ================================ Cross Correlation

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void crossCorrs(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, data_t3* out, len_t3 stride=1)
{
	return mapSubseqs([m, q](data_t2* x) {
		return dot(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto crossCorrs(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(q[0] * x[0])[]>
{
	return mapSubseqs([m, q](data_t2* x) {
		return dot(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2, class len_t3=size_t>
static auto crossCorrs(const Container1<data_t1>& q,
	const Container2<data_t2>& x, len_t3 stride=1)
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

template<class data_t1, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void dists_L1(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, data_t3* out, len_t3 stride=1)
{
	return mapSubseqs([m, q](data_t2* x) {
		return dist_L1(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto dists_L1(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(q[0] - x[0])[]>
{
	return mapSubseqs([m, q](data_t2* x) {
		return dist_L1(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2, class len_t3=size_t>
static auto dists_L1(const Container1<data_t1>& q,
	const Container2<data_t2>& x, len_t3 stride=1)
	-> Container2<decltype(q[0] - x[0])>
{
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dist_L1(_q, _x, m);
	}, q, x, stride);
}

// ================================ L2 distance

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void dists_L2(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, data_t3* out, len_t3 stride=1)
{
	return mapSubseqs([m, q](data_t2* x) {
		return dist_L2(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto dists_L2(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(q[0] - x[0])[]>
{
	return mapSubseqs([m, q](data_t2* x) {
		return dist_L2(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2, class len_t3=size_t>
static auto dists_L2(const Container1<data_t1>& q,
	const Container2<data_t2>& x, len_t3 stride=1)
	-> Container2<decltype(q[0] - x[0])>
{
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dist_L2(_q, _x, m);
	}, q, x, stride);
}

// ================================ L2^2 distance

// ------------------------ raw arrays

template<class data_t1, class data_t2, class data_t3,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline void dists_sq(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, data_t3* out, len_t3 stride=1)
{
	return mapSubseqs([m, q](data_t2* x) {
		return dist_sq(q, x, m);
	}, m, x, n, out, stride);
}

template<class data_t1, class data_t2,
	class len_t1=size_t, class len_t2=size_t, class len_t3=size_t>
static inline auto dists_sq(const data_t1* q, len_t1 m, const data_t2* x,
	len_t2 n, len_t3 stride=1)
	-> unique_ptr<decltype(q[0] * x[0])[]>
{
	return mapSubseqs([m, q](data_t2* x) {
		return dist_sq(q, x, m);
	}, m, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container1, class data_t1,
	template <class...> class Container2, class data_t2, class len_t3=size_t>
static auto dists_sq(const Container1<data_t1>& q,
	const Container2<data_t2>& x, len_t3 stride=1)
	-> Container2<decltype(q[0] - x[0])>
{
	auto m = q.size();
	return mapSubseqs([m](const data_t1* _q, const data_t2* _x) {
		return dist_sq(_q, _x, m);
	}, q, x, stride);
}

// ================================ 1st discrete derivative

// ------------------------ raw arrays

template<class data_t2, class data_t3, class len_t2=size_t, class len_t3=size_t>
static inline void first_derivs(const data_t2* x, len_t2 n, data_t3* out,
	len_t3 stride=1)
{
	data_t2 q[2] = {-1, 1};
	return crossCorrs(q, 2, x, n, out, stride);
}

template<class data_t2, class len_t2=size_t, class len_t3=size_t>
static inline unique_ptr<data_t2[]> first_derivs(const data_t2* x, len_t2 n,
	len_t3 stride=1)
{
	data_t2 q[2] = {-1, 1};
	return crossCorrs(q, 2, x, n, stride);
}

// ------------------------ containers

template<template <class...> class Container2, class data_t2,
	class len_t3=size_t>
static Container2<data_t2> first_derivs(const Container2<data_t2>& x,
	len_t3 stride=1)
{
	vector<data_t2> q {-1, 1};
	return crossCorrs(q, x, stride);
}


// ================================================================
// can't get eigen stuff to work because lambdas and eigen::blocks don't
// really play together at all; there's also basically no point because
// most of the subseqs won't be aligned properly so eigen won't even
// be faster necessarily

//#include <Dense>
//
//using Eigen::VectorBlock;
//
//using Eigen::Dynamic;
//using Eigen::EigenBase;
//using Eigen::MatrixBase; // can't use eigenbase so can index objs in decltype
//using Eigen::Matrix;

//template<class F, class Derived1, class Derived2, class Derived3>
//static void mapSubseqs(const F&& func, const MatrixBase<Derived1>& shorter,
//	const MatrixBase<Derived2>& longer, MatrixBase<Derived3>& out) {
//	auto n = longer.size();
//	auto m = shorter.size();
//	auto l = n - m + 1;
//	assert(n >= m);
//	for (size_t i = 0; i < l; i++) {
//		auto subseq = longer.segment(i, m);
////		out.noalias()(i) = func(subseq, shorter);
//		out(i) = func(subseq, shorter);
//	}
//}

//template<class F, class Derived1, class Derived2, class Derived3>
//static void mapSubseqs(const F& func, const MatrixBase<Derived1>& shorter,
//					   const MatrixBase<Derived2>& longer, MatrixBase<Derived3>& out) {
//	auto n = longer.size();
//	auto m = shorter.size();
//	auto l = n - m + 1;
//	assert(n >= m);
//	for (size_t i = 0; i < l; i++) {
//		auto subseq = longer.segment(i, m);
//		//		out.noalias()(i) = func(subseq, shorter);
//		out(i) = func(subseq, shorter);
//	}
//}

//template<class F, class Derived1, class Derived2>
//static auto mapSubseqs(const F&& func, const MatrixBase<Derived1>& shorter,
//	const MatrixBase<Derived2>& longer) ->
////Matrix<decltype( func(longer.segment(0, shorter.size()), shorter) ), Dynamic, 1>
//Matrix<decltype( longer(0) * shorter(0) ), Dynamic, 1>
//{
//	auto n = longer.size();
//	auto m = shorter.size();
//	auto l = n - m + 1;
//	assert(n >= m);
////	Matrix<decltype( func(shorter, longer.segment(0, shorter.size())) ), Dynamic, 1> out(l);
//	Matrix<decltype( longer(0) * shorter(0) ), Dynamic, 1> out(l);
//	for (size_t i = 0; i < l; i++) {
//		auto subseq = longer.segment(i, m);
//		out(i) = func(subseq, shorter);
//	}
//	return out;
//
////	auto l = longer.size() - shorter.size() + 1;
//	//	VectorXd out(l);
////	Matrix<decltype( func(shorter, longer.segment(0, shorter.size())) ), Dynamic, 1> out(l);
////	Matrix<decltype( shorter.dot(longer.segment(0, shorter.size())) ), Dynamic, 1> out(l);
////	mapSubseqs(std::forward<F>(func), shorter, longer, out);
////	mapSubseqs(func, shorter, longer, out);
////	return out;
//}

// ================================ Cross-correlate

//template<class Derived1, class Derived2, class Derived3>
//static void crossCorrs(const MatrixBase<Derived1>& shorter,
//	const MatrixBase<Derived2>& longer, MatrixBase<Derived3>& out) {
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return x.dot(y);
//	}, shorter, longer, out);
//}

//template<class Derived1, class Derived2>
//static auto crossCorrs(const MatrixBase<Derived1>& shorter,
//					const MatrixBase<Derived2>& longer) ->
//	Matrix<decltype(shorter(0) * longer(0)), Dynamic, 1>
//{
////	return mapSubseqs([](const VectorBlock<MatrixBase<Derived1> >& x,
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
////		return x * y;
//		return x.dot(y);
////		return (x.array() * y.array()).sum();
//	}, shorter, longer);
//
////	// TODO remove
////	auto n = longer.size();
////	auto m = shorter.size();
////	auto l = n - m + 1;
////	assert(n >= m);
////	Matrix<decltype( shorter.dot(longer.segment(0, shorter.size())) ), Dynamic, 1> out(l);
////	for (size_t i = 0; i < l; i++) {
////		auto subseq = longer.segment(i, m);
////		//		out.noalias()(i) = func(subseq, shorter);
//////		out(i) = func(subseq, shorter);
////		out(i) = subseq.dot(shorter);
////	}
////	return out;
//}

// ================================ Distances

// ------------------------ L1

//template<class Derived1, class Derived2, class Derived3>
//static inline void L1Dist(const MatrixBase<Derived1>& shorter,
//	const MatrixBase<Derived2>& longer, MatrixBase<Derived3>& out) {
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return (x - y).array().abs().sum();
//	}, shorter, longer, out);
//}
//
//template<class Derived1, class Derived2>
//static inline auto L1Dist(const MatrixBase<Derived1>& shorter,
//				 const MatrixBase<Derived2>& longer) ->
//Matrix<decltype(shorter(0) * longer(0)), Dynamic, 1>
//{
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return (x - y).array().abs().sum();
//	}, shorter, longer);
//}

// ------------------------ L2

//template<class Derived1, class Derived2, class Derived3>
//static inline void L2Dist(const MatrixBase<Derived1>& shorter,
//	const MatrixBase<Derived2>& longer, MatrixBase<Derived3>& out) {
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return (x - y).norm();
//	}, shorter, longer, out);
//}
//
//template<class Derived1, class Derived2>
//static inline auto L2Dist(const MatrixBase<Derived1>& shorter,
//				 const MatrixBase<Derived2>& longer) ->
//Matrix<decltype(shorter(0) * longer(0)), Dynamic, 1>
//{
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return (x - y).norm();
//	}, shorter, longer);
//}

// ------------------------ L2^2

//template<class Derived1, class Derived2, class Derived3>
//static inline void squaredDist(const MatrixBase<Derived1>& shorter,
//	const MatrixBase<Derived2>& longer, MatrixBase<Derived3>& out) {
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return (x - y).squaredNorm();
//	}, shorter, longer, out);
//}
//
//template<class Derived1, class Derived2>
//static inline auto squaredDist(const MatrixBase<Derived1>& shorter,
//				 const MatrixBase<Derived2>& longer) ->
//Matrix<decltype(shorter(0) * longer(0)), Dynamic, 1>
//{
//	return mapSubseqs([](const MatrixBase<Derived1>& x,
//						 const MatrixBase<Derived2>& y) {
//		return (x - y).squaredNorm();
//	}, shorter, longer);
//}

} // namespace subs

#endif
