//
//  filter.hpp
//
//  Created By Davis Blalock on 3/14/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __DIG_FILTER_HPP
#define __DIG_FILTER_HPP

// #include <stdlib.h> // for size_t
#include <deque>
#include <memory>

using std::deque;
using std::unique_ptr;

namespace filter {

// ================================ MinMax filter

// constructed using Lemire's algorithm; see "Faster Retrieval with a
// Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
template <class data_t, class len_t, REQUIRE_INT(len_t)>
void min_max_filter(const data_t *t, len_t len, len_t r,
	data_t *l, data_t *u)
{
    len_t i = 0;
	len_t width = 2 * r + 1;

    deque du(width + 1);
    deque dl(width + 1);

    du.push_back(0);
    dl.push_back(0);

    for (i = 1; i < len; i++) {
        if (i > r) {
            u[i - r - 1] = t[du.front()];
            l[i - r - 1] = t[dl.front()];
        }
        // remove idxs from deque that are less extreme than the value at
        // this idx; they can't be extrema now or in the future
        if (t[i] > t[i - 1]) {
            du.pop_back();
            while (!du.empty() && t[i] > t[du.back()]) {
                du.pop_back();
            }
        } else {
            dl.pop_back();
            while (!dl.empty() && t[i] < t[dl.back()]) {
                dl.pop_back();
            }
        }
        du.push_back(i);
        dl.push_back(i);
        // check if current extrema are passing out of window
        if (i == width + du.front()) {
            du.pop_front();
        } else if (i == width + dl.front()) {
            dl.pop_front();
        }
    }

    for (i = len; i < len + r + 1; i++) {
        u[i - r - 1] = t[du.front()];
        l[i - r - 1] = t[dl.front()];
        if (i - du.front() >= width) {
            du.pop_front();
        }
        if (i - dl.front() >= width) {
            dl.pop_front();
        }
    }
}

template <class data_t, class len_t, REQUIRE_INT(len_t)>
void min_filter(const data_t *t, len_t len, len_t r, data_t *l) {
    len_t i = 0;
	len_t width = 2 * r + 1;

    deque dl(width + 1);
    dl.push_back(0);

    for (i = 1; i < len; i++) {
        if (i > r) {
            l[i - r - 1] = t[dl.front()];
        }
        if (t[i] <= t[i - 1]) {
            dl.pop_back();
            while (!dl.empty() && t[i] < t[dl.back()]) {
                dl.pop_back();
            }
        }
        dl.push_back(i);
        if (i == width + dl.front()) {
            dl.pop_front();
        }
    }

    for (i = len; i < len + r + 1; i++) {
        l[i - r - 1] = t[dl.front()];
        if (i - dl.front() >= width) {
            dl.pop_front();
        }
    }
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
void max_filter(const data_t *t, len_t len, len_t r, data_t *u) {
    len_t i = 0;
	len_t width = 2 * r + 1;

    deque du(width + 1);
    du.push_back(0);

    for (i = 1; i < len; i++) {
        if (i > r) {
            u[i - r - 1] = t[du.front()];
        }
        if (t[i] > t[i - 1]) {
            du.pop_back();
            while (!du.empty() && t[i] > t[du.back()]) {
                du.pop_back();
            }
        }
        du.push_back(i);
        if (i == width + du.front()) {
            du.pop_front();
        }
    }

    for (i = len; i < len + r + 1; i++) {
        u[i - r - 1] = t[du.front()];
        if (i - du.front() >= width) {
            du.pop_front();
        }
    }
}

template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> min_filter(const data_t* data,
	len_t len, len_t r)
{
	unique_ptr<data_t[]> ret(new data_t[len]);
	min_filter(data, len, r, ret.get());
	return ret;
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> max_filter(const data_t* data,
	len_t len, len_t r)
{
	unique_ptr<data_t[]> ret(new data_t[len]);
	max_filter(data, len, r, ret.get());
	return ret;
}

template<template <class...> class Container, class data_t,
	class len_t, REQUIRE_INT(len_t)>
static inline Container<data_t> min_filter(const Container& data,
	len_t len, len_t r)
{
	Container<data_t> ret(len);
	min_filter(t, len, r, ret.data());
	return ret;
}
template<template <class...> class Container, class data_t,
	class len_t, REQUIRE_INT(len_t)>
static inline Container<data_t> max_filter(const Container& data,
	len_t len, len_t r)
{
	Container<data_t> ret(len);
	max_filter(t, len, r, ret.data());
	return ret;
}

// ================================ Apodization windows

template<alpha, beta, class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void _generalized_hamming(data_t* out, len_t len) {
	for (len_t i = 0; i < len; i++) {
		double val = alpha - beta * std::cos(2.0 * pi * i / (len-1));
		out[i] = static_cast<data_t>(val);
	}
}

// ------------------------ hann
template<class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void hann(data_t* out, len_t len) {
	return _generalized_hamming<.5, .5>(out, len);
}
template <class data_t=double, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> hann(len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	hann(ret.get(), len);
	return ret;
}
template<class data_t=double, class len_t, REQUIRE_INT(len_t)>
static inline vector<data_t> hann(len_t len) {
	vector<data_t> ret(len);
	hann(ret.data(), len);
	return ret;
}

// ------------------------ hamming
template<class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void hamming(data_t* out, len_t len) {
	return _generalized_hamming<.54, .46>(out, len);
}
template <class data_t=double, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> hamming(len_t len) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	hamming(ret.get(), len);
	return ret;
}
template<class data_t=double, class len_t, REQUIRE_INT(len_t)>
static inline vector<data_t> hamming(len_t len) {
	vector<data_t> ret(len);
	hamming(ret.data(), len);
	return ret;
}

// ------------------------ gaussian
template<class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void gaussian(data_t* out, len_t len, double sigma) {
	for (len_t i = 0; i < len; i++) {
		double numerator = i - (len-1) / 2;
		double denominator = sigma * (len-1) / 2;
		double frac = numerator / denominator;
		double exponent = -.5 * (frac * frac)
		out[i] = static_cast<data_t>(std::exp(exponent));
	}
}
template <class data_t=double, class len_t, REQUIRE_INT(len_t)>
static inline unique_ptr<data_t[]> gaussian_ar(len_t len, double sigma) {
	unique_ptr<data_t[]> ret(new data_t[len]);
	gaussian(ret.get(), len, sigma);
	return ret;
}
template<class data_t=double, class len_t, REQUIRE_INT(len_t)>
static inline vector<data_t> gaussian(len_t len, double sigma) {
	vector<data_t> ret(len);
	hamming(ret.data(), len, sigma);
	return ret;
}


} // namespace filter
#endif
