//
//  distance_utils.hpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TimeKit__distance_utils__
#define __TimeKit__distance_utils__

#include "distance_utils.h"

#include <vector>

template <class T, class T2>
auto L1_loss(const T x, const T2 y) -> decltype(x-y) {
	auto difference = x - y;
	return difference >= 0 ? difference : -difference;
}
//class F_L1_loss {
//	auto operator()(...) const { return L1_loss
//};


template <class T, class T2>
auto L2_loss(const T x, const T2 y) -> decltype(x*y) {
	return (x - y) * (x - y);
}

template <class data_t, class len_t>
void z_normalize(data_t* ar, len_t len) {
	data_t sumx = 0, sumx2 = 0;
	float x, mean, std;
	
	//compute mean and standard deviation
	for (len_t i = 0; i < len; i++) {
		x = ar[i];
		sumx += x;
		sumx2 += x*x;
	}
	mean = sumx / len;
	std = sqrt(sumx2 / len - mean*mean);
	if (std == 0) {
		return;
	}
	
	//z-normalize array in-place
	for (len_t i = 0; i < len; i++) {
		ar[i] = (ar[i] - mean) / std;
	}
}

template <class T, class len_t>
std::vector<len_t> sorted_indices(const std::vector<T>& v, len_t len, bool increasing=true) {
	std::vector<len_t> idx(len);
	for (len_t i = 0; i < len; i++) {
		idx[i] = i;
	}
 	std::sort(std::begin(idx), std::end(idx),
       [&v,&increasing](len_t i1, len_t i2) {return (v[i1] < v[i2]) == increasing;} );
 	return idx;
}

#endif /* defined(__TimeKit__distance_utils__) */
