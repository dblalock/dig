//
//  debug_utils.hpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TimeKit__debug_utils_hpp__
#define __TimeKit__debug_utils_hpp__

#include "debug_utils.h"	//include the pure c functions

#include <vector>
#include <iostream>

template <class T>
void print_array(const char* name, const std::vector<T>& v) {
	printf("%s:\t", name);
	for(T element : v) {
		std::cout << element << " ";
		// printf("%g", static<castelement);
	}
	printf("\n");
}

// ================================ Array printing

/** Prints: "[ data[0] data[1] ... data[n] ]" on its own line */
inline void array_print(const double *x, unsigned int len) {
	printf("[");
	for (unsigned int i = 0; i < len; i++) {
		printf("%.3f ", x[i]);
	}
	printf("]\n");
}

/** Prints: "[ data[0] data[1] ... data[n] ]" on its own line */
inline void intarray_print(const int *x, unsigned int len) {
	printf("[");
	for (unsigned int i = 0; i < len; i++) {
		printf("%d ", x[i]);
	}
	printf("]\n");
}

/** Prints: "[ data[0] data[1] ... data[n] ]" on its own line */
inline void bytearray_print(const uint8_t *x, unsigned int len) {
	printf("[");
	for (unsigned int i = 0; i < len; i++) {
		printf("%d ", x[i]);
	}
	printf("]\n");
}

/** Prints: "<name>:\t[ data[0] data[1] ... data[n] ]" on its own line */
inline void array_print_with_name(const double *x, unsigned int len, const char* name) {
	printf("%s:\t",name);
	array_print(x, len);
}

/** Prints: "<name>:\t[ data[0] data[1] ... data[n] ]" on its own line */
inline void intarray_print_with_name(const int *x, unsigned int len, const char* name) {
	printf("%s:\t",name);
	intarray_print(x, len);
}

/** Prints: "<name>:\t[ data[0] data[1] ... data[n] ]" on its own line */
inline void bytearray_print_with_name(const uint8_t *x, unsigned int len, const char* name) {
	printf("%s:\t",name);
	bytearray_print(x, len);
}


#endif