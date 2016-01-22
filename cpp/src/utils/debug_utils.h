//
//  debug_utils.h
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TimeKit__debug_utils__
#define __TimeKit__debug_utils__

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define clean_errno() (errno == 0 ? "None" : strerror(errno))
#define log_error(M, ...) fprintf(stderr, "[ERROR] (%s:%d: errno: %s) " M "\n", __FILE__, __LINE__, clean_errno(), ##__VA_ARGS__)
#define assertf(A, M, ...) if(!(A)) {log_error(M, ##__VA_ARGS__); assert(A); }

inline void print_var(const char* name, double val) {
	printf("%s=%g\n",name, val);
}

// won't work cuz we don't know size of elements for void*
//inline void print_array(const char* name, void* ar, int len) {
//	double*v = (double*) ar;
//	printf("%s: ", name);
//	for(unsigned i = 0; i < len; ++i) {
//		printf("%g ", v[i]);
//	}
//	printf("\n");
//}

#ifdef __cplusplus
}
#endif



#endif