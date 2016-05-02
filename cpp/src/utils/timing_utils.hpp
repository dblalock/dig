//
//  timing_utils.cpp
//  Dig
//
//  Created by Davis Blalock on 2016-3-28
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef _TIMING_UTILS_HPP
#define _TIMING_UTILS_HPP

#include <chrono>

namespace {

using cputime_t = std::chrono::high_resolution_clock::time_point;
//#define clock std::chrono::high_resolution_clock // because so much typing

static inline cputime_t timeNow() {
	return std::chrono::high_resolution_clock::now();
}

static inline double durationMs(cputime_t t1, cputime_t t0) {
	double diffMicros = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	return std::abs(diffMicros) / 1000.0;
}

} // anon namespace
#endif // _TIMING_UTILS_HPP
