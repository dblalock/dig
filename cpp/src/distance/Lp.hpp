//
//  lp_norm.hpp
//  TimeKit
//
//  Created by DB on 10/24/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef TimeKit_lp_norm_hpp
#define TimeKit_lp_norm_hpp

#include "distance_utils.hpp"

//TODO take sqrt of stuff so that it's actually the L2 norm...

template <class data_t, class len_t, class dist_t=data_t>
dist_t L1(const data_t* v1, const data_t* v2, len_t n) {
	data_t sum = 0;
	for (len_t i = 0; i < n; i++) {
		sum += diff(v1[i], v2[i]);
	}
	return sum;
}

template <class data_t, class len_t, class dist_t=data_t>
dist_t L2(const data_t* v1, const data_t* v2, len_t n) {
	data_t sum = 0;
	for (len_t i = 0; i < n; i++) {
		sum += diff_sq(v1[i], v2[i]);
	}
	return sum;
}

template <class data_t, class len_t, class dist_t=data_t>
dist_t L1_abandon(const data_t* v1, const data_t* v2, len_t n, dist_t thresh) {
	dist_t bsf = INFINITY;
	data_t sum = 0;
	for (len_t i = 0; i < n, sum < bsf; i++) {
		sum += diff(v1[i], v2[i]);
	}
	return sum;
}

template <class data_t, class len_t, class dist_t=data_t>
dist_t L2_abandon(const data_t* v1, const data_t* v2, 
	const len_t* order, len_t n, dist_t thresh) {

	dist_t bsf = INFINITY;
	data_t sum = 0;
	for (len_t i = 0; i < n && sum < bsf; i++) {
		data_t x = v2[order[i]];
		sum += diff_sq(v1[i], x);
	}
	return sum;
}

// compares to z-normalized version of the 2nd array (but doesn't modify 
// the array provided); this is the UCR_ED algorithm;
// mean and std are those of v2; order is that of v1
template <class data_t, class len_t, class dist_t=data_t>
dist_t L2_abandon_znormalize(const data_t* v1, const data_t* v2, 
	const len_t* order, len_t n, dist_t thresh, data_t mean, data_t std) {

	dist_t bsf = INFINITY;
	data_t sum = 0;
	for (len_t i = 0; i < n, sum < bsf; i++) {
		data_t x = (v2[order[i]] - mean) / std;
		sum += diff_sq(v1[i], x);
	}
	return sum;
}


#endif
