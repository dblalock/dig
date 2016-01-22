//
//  array_utils.h
//  edtw
//
//  Created By <Anonymous> on 1/14/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#ifndef timekit_array_utils_h
#define timekit_array_utils_h

#include <stdint.h>
#include "type_defs.h"

#ifdef __cplusplus
extern "C" {
#endif


//TODO probably remove these comments and replace them with the ones in the c file


// array creation
/** Creates an array with a sequence of values; equivalent to
 * startVal:step:stopVal in MATLAB, seq(startVal, stopVal, step) in R,
 * etc. */
data_t* array_create_sequence(data_t min, data_t max, data_t step);

/** Reshapes a 1D array to k dimensions such that every kth element is appended
 * to one of k arrays corresponding to each dimension */
data_t** array_reshape(const data_t* data, unsigned int len, unsigned int newNumDims);

// statistics
/** Returns the maximum element in an array */
data_t array_max(const data_t* data, unsigned int len);

/** Returns the minimum element in an array */
data_t array_min(const data_t *data, unsigned int len);

/** Returns the sum of the elements of an array */
data_t array_sum(const data_t *data, unsigned int len);

/** Returns the sum of the squares of the elments of an array */
data_t array_sum_squared(const data_t *data, unsigned int len);

/** Returns the mean of an array */
data_t array_mean(const data_t *data, unsigned int len);

/** Returns the variance of an array */
data_t array_variance(const data_t *data, unsigned int len);

/** Returns the sums of the variances of an array along each dimension */
data_t array_variance_nd(const data_t *data, unsigned int dimLen, unsigned int nDims);
	
/** Returns the square of the Euclidean distance (L2 norm) between two arrays */
data_t array_euclidean_distance(const data_t *x, const data_t *y, unsigned int len);

/** Returns the sum of the elements in an array of longs */
long larray_sum(const long *data, unsigned int len);

/** Returns the mean of an array of longs */
data_t larray_mean(const long *data, unsigned int len);
	
/** Returns the variance of an array of longs */
data_t larray_variance(const long *data, unsigned int len);
	
// in-place modification, two vectors

/** Element-wise vector addition */
void array_add(data_t *src, data_t *dest, unsigned int len);

/** Element-wise vector subtraction (subtracts src from dest) */
void array_sub(data_t *src, data_t *dest, unsigned int len);
	
/** Element-wise vector multiplication */
void array_mult(data_t *src, data_t *dest, unsigned int len);

/** Element-wise vector division (divides dest by src) */
void array_div(data_t *src, data_t *dest, unsigned int len);
	
/** Copies values in src to dest */
void array_copy(const data_t *src, data_t *dest, unsigned int len);
	
/** Copies values in src to dest in reverse order */
void array_copy_reverse(data_t *src, data_t *dest, unsigned int len);

/** Resamples values in src and writes result to dest */
void array_resample(const data_t *src, data_t *dest, unsigned int srcLen,
					unsigned int destLen);

// in-place modification, vector and scalar
void array_add_scalar(data_t *data, data_t val, unsigned int len);
void array_sub_scalar(data_t *data, data_t val, unsigned int len);
void array_mult_scalar(data_t *data, data_t val, unsigned int len);
void array_div_scalar(data_t *data, data_t val, unsigned int len);
void array_set_to_constant(data_t *x, data_t value, unsigned int len);

// in-place modification, single vector
void array_cum_sum(data_t *data, unsigned int len);
void array_cum_Sxx(data_t *data, unsigned int len);
void array_cum_mean(data_t*data, unsigned int len);

// comparison and tests
short int array_equal(const data_t *x, const data_t *y, unsigned int len);
short int intarray_equal(const int *x, const int *y, unsigned int len);

// output
void array_print(const data_t *x, unsigned int len);
void intarray_print(const int *x, unsigned int len);
void bytearray_print(const uint8_t *x, unsigned int len);
void array_print_with_name(const data_t *x, unsigned int len, const char* name);
void intarray_print_with_name(const int *x, unsigned int len, const char* name);
void bytearray_print_with_name(const uint8_t *x, unsigned int len, const char* name);

#ifdef __cplusplus
}
#endif
	
#endif
