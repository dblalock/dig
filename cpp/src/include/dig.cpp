//
//  dig.cpp
//  Dig
//
//  Created by DB on 7/28/14.
//  Copyright (c) 2014 D Blalock. All rights reserved.
//

#include "dig.hpp"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <array>

#include "Lp.hpp"
#include "dtw.hpp"

int swigTest(int x) {
	printf("swigTest: received int: %d\n", x);
	return ++x;
}

double swigArrayTest(const double* x, int len) {
	double sum = 0;
	for (int i = 0; i < len; i++) {
		sum += x[i];
	}
	return sum;
}



// static inline short 	diff(short x, short y) { 	return abs(x - y);  }
// static inline int 		diff(int x, int y) { 		return abs(x - y);  }
// static inline float 	diff(float x, float y) { 	return fabs(x - y); }
// static inline double 	diff(double x, double y) { 	return fabs(x - y); }

// template <class data_t>
// static inline data_t diff_sq(data_t x, data_t y) {
// 	data_t diff = x - y;
// 	return diff * diff;
// }

// ================================================================
// Core distance functions		//TODO add stride to these
// ================================================================


// ------------------------------- L1 distance wrappers

int dist_L1(const int* v1, const int* v2, int n) {
	return L1(v1, v2, n);
}
int dist_L1(const int* v1, int m, const int* v2, int n) {
	assert(m == n);		//only support equal-length arrays
	return L1(v1, v2, n);
}
double dist_L1(const double* v1, const double* v2, int n) {
	return L1(v1, v2, n);
}
double dist_L1(const double* v1, int m, const double* v2, int n) {
	assert(m == n);		//only support equal-length arrays
	return L1(v1, v2, n);
}

// ------------------------------- L2 distance wrappers

int dist_L2(const int* v1, const int* v2, int n) {
	return L2(v1, v2, n);
}
int dist_L2(const int* v1, int m, const int* v2, int n) {
	assert(m == n);		//only support equal-length arrays
	return L2(v1, v2, n);
}
double dist_L2(const double* v1, const double* v2, int n) {
	return L2(v1, v2, n);
}
double dist_L2(const double* v1, int m, const double* v2, int n) {
	assert(m == n);		//only support equal-length arrays
	return L2(v1, v2, n);
}

// ------------------------------- DTW distance wrappers

int dist_dtw(const int* v1, int m, const int* v2, int n, int r) {
	assert(m == n);		//only support equal-length arrays
	return dtw(v1, v2, m, r);
}
int dist_dtw(const int* v1, const int* v2, int m, int r) {
	return dtw(v1, v2, m, r);
}
double dist_dtw(const double* v1, int m, const double* v2, int n, int r) {
	assert(m == n);		//only support equal-length arrays
	return dtw(v1, v2, m, r);
}
double dist_dtw(const double* v1, const double* v2, int m, int r) {
	return dtw(v1, v2, m, r);
}
