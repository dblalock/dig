//
//  distance_utils.hpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TimeKit__distance_utils_h__
#define __TimeKit__distance_utils_h__


//TODO remove this file and the refactor the files that depend on it
//(ucr_funcs.c and test_ucr_funcs.c)


#include "type_defs.h"		//TODO template everything

// stupidly simple macros to save writing out different template instantations
// to deal with mixed types and avoid picking between abs() and fabs(); more
// importantly, this gets included from ucr_funcs.c, which gets compiled as
// c code, and thus can't deal with templates
#define diff(x,y) ( ((x)-(y)) >=0 ? ((x)-(y)) : ((y)-(x)) )
#define diff_sq(x,y) ( ((x)-(y)) * ((x)-(y)) )
//template <class T, class T2>
//auto diff(const T x, const T2 y) -> decltype(x-y) {
//	auto difference = x - y;
//	return difference >= 0 ? difference : -difference;
//}

#ifdef __cplusplus
extern "C" {
#endif

void znormalize(data_t* ar, length_t m);
void znormalize_prefix(data_t *ar, length_t m, length_t stopAfter);

int32_t sort_abs_decreasing(data_t* normalizedAr, idx_t* order, length_t m);
int32_t sort_abs_increasing(data_t* normalizedAr, idx_t* order, length_t m);

#ifdef __cplusplus
}
#endif

#endif