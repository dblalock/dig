//
//  array_utils.c
//  edtw
//
//  Created By <Anonymous> on 1/14/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "array_utils.h"

/** Fills the array with a sequence of values; equivalent to 
  * startVal:step:stopVal in MATLAB, seq(startVal, stopVal, step) in R,
  * etc. */
data_t* array_create_sequence(data_t startVal, data_t stopVal, data_t step) {
	if ( (stopVal - startVal) * step <= 0) {
		printf("ERROR: array_sequence: invalid args min=%.3f, max=%.3f, step=%.3f\n",
			   startVal, stopVal, step);
		return NULL;
	}
	
	//allocate a new array
	unsigned int len = (unsigned int) floor( (stopVal - startVal) / step ) + 1;
	data_t *data = (data_t*) malloc(len*sizeof(double));
	if (data == NULL) return NULL;
	
	data[0] = startVal;
	for (unsigned int i = 1; i < len; i++ ) {
		data[i] = data[i-1] + step;
	}
	return data;
}

data_t** array_reshape(const data_t* data, unsigned int len, unsigned int newNumDims) {
	unsigned int newArraysLen = len / newNumDims;
	if ( newArraysLen * newNumDims != len) {
		printf("WARNING: array_reshape: newNumDims %d is not factor of array length %d", newNumDims, len);
		return NULL;
	}
	
	size_t sample,dimension,readFrom=0;
	//initialize each array ptr and the array containing them; note
	//that the arrays are allocated as one contiguous block of memory
	data_t** arrays = (data_t**)malloc(newNumDims*sizeof(data_t*));
	data_t* arrayContents = (data_t*)malloc(len*sizeof(data_t));
	for (dimension = 0; dimension < newNumDims; dimension++) {
		arrays[dimension] = arrayContents + dimension*newArraysLen;
	}
	
	//copy the values from the 1D array to be reshaped
	for (sample = 0; sample < newArraysLen; sample++) {
		for (dimension = 0; dimension < newNumDims; dimension++, readFrom++) {
			arrays[dimension][sample] = data[readFrom];
		}
	}
	
	return arrays;
}

/** Returns the maximum value in data[0..len-1] */
data_t array_max(const data_t *data, unsigned int len) {
	data_t max = -INFINITY;
	for (unsigned int i = 0; i < len; i++) {
		if (data[i] > max) {
			max = data[i];
		}
	}
	return max;
}

/** Returns the minimum value in data[0..len-1] */
data_t array_min(const data_t *data, unsigned int len) {
	data_t min = INFINITY;
	for (unsigned int i = 0; i < len; i++) {
		if (data[i] < min) {
			min = data[i];
		}
	}
	return min;
}

/** Computes the sum of data[0..len-1] */
data_t array_sum(const data_t *data, unsigned int len) {
	data_t sum = 0;
	for (unsigned int i=0; i < len; i++) {
		sum += data[i];
	}
	return sum;
}

/** Computes the sum of data[0..len-1] */
long larray_sum(const long *data, unsigned int len) {
	long sum = 0;
	for (unsigned int i=0; i < len; i++) {
		sum += data[i];
	}
	return sum;
}

/** Computes the sum of data[i]^2 for i = [0..len-1] */
data_t array_sum_squared(const data_t *data, unsigned int len) {
	data_t sum = 0;
	for (unsigned int i=0; i < len; i++) {
		sum += data[i]*data[i];
	}
	return sum;
}

/** Computes the cumulative sum of the elements in data and writes them
  * in place; ie, data[i] = data[0] + data[1] + ... + data[i-1] */
void array_cum_sum(data_t *data, unsigned int len) {
	for (unsigned int i=1; i < len; i++) {
		data[i] += data[i-1];
	}
}

/** Computes the sum of squared differences from the mean of data[0..i] and 
  * stores it in data[i] for i = 0..len-1 */
void array_cum_Sxx(data_t *data, unsigned int len) {
	data_t sumX=0, sumX2=0;
	data_t x, xsq;
	for (unsigned int i = 0; i < len; i++) {
		x = data[i];
		xsq = x*x;
		sumX += x;
		sumX2 += xsq;
		data[i] = sumX2 - sumX*sumX / (i+1);
	}
}

/** Computes the mean of data[0..i] and stores it in data[i] for i = 0..len-1 */
void array_cum_mean(data_t*data, unsigned int len) {
	array_cum_sum(data, len);
	data_t *idxs = array_create_sequence(1, len, 1);
	array_div(idxs, data, len);
	free(idxs);
}

/** Computes the arithmetic mean of data[0..len-1] */
data_t array_mean(const data_t *data, unsigned int len) {
	return array_sum(data, len) / len;
}

/** Computes the mean of data[0..len-1] */
data_t larray_mean(const long *data, unsigned int len) {
	return larray_sum(data, len) / ((double) len);
}

/** Computes the population variance of data[0..len-1] */
data_t array_variance(const data_t *data, unsigned int len) {
	data_t sum = 0;
	data_t sumSq = 0;
	for (unsigned int i=0; i < len; i++) {
		sum += data[i];
		sumSq += data[i] * data[i];
	}
	data_t E_x = sum / len;
	data_t E_x2 = sumSq / len;
	return E_x2 - (E_x*E_x);
}

/** Computes the population variance of data[0..len-1] */
data_t larray_variance(const long *data, unsigned int len) {
	long sum = 0;
	long sumSq = 0;
	for (unsigned int i=0; i < len; i++) {
		sum += data[i];
		sumSq += data[i] * data[i];
	}
	data_t E_x = sum / ((double) len);
	data_t E_x2 = sumSq / ((double) len);
	return E_x2 - (E_x*E_x);
}

data_t array_variance_nd(const data_t *data, unsigned int dimLen, unsigned int nDims) {
	// NOTE: this function assumes that samples are lined up contiguously in the
	// array, such that adjacent points along a dimension are nDims apart. Note,
	// too, that we're returning E[(X-ux)^2] using different means for each
	// dimension, NOT the covariance matrix.
	
	// initialize arrays of averages across each dimension
	data_t avgs[nDims];
	for (unsigned int i=0; i < nDims; i++) {
		avgs[i] = 0;
	}
	
	// determine averages for each dimension by first finding the sums,
	// and then dividing by the total number of points in each dimension
	unsigned int nTotalValues = dimLen * nDims;
	for (unsigned int idx = 0; idx < nTotalValues; idx += nDims) {
		for (unsigned int dim = 0; dim < nDims; dim++) {
			data_t dataVal = data[idx + dim];
			avgs[dim] += dataVal;
		}
	}
	for (unsigned int i=0; i < nDims; i++) {
		avgs[i] /= dimLen;
	}
	
	// now calculate the total L2 dist of X from E[X], using the means
	// of each dimension. I.e., calculate SUM[ (X-ux)^2 ]
	data_t total_L2_dist = 0;
	for (unsigned int idx = 0; idx < nTotalValues; idx += nDims) {
		for (unsigned int dim = 0; dim < nDims; dim++) {
			data_t xMinusAvg = data[idx+dim] - avgs[dim];
			total_L2_dist += xMinusAvg*xMinusAvg;
		}
	}
	
	// return E[(X-ux)^2]
	return total_L2_dist / nTotalValues;
}

/** Returns the sum of the differences between x[0..len-1] and y[0..len-1]
  * squared; note that this does NOT take the square root of this value */
data_t array_euclidean_distance(const data_t *x, const data_t *y, unsigned int len) {
	data_t sum = 0;
	data_t diff;
	for (int i = 0; i < len; i++) {
		diff = x[i] - y[i];
		sum += diff*diff;
	}
	return sum;
}

/** Elementwise add src and dest, storing the result in dest */
void array_add(data_t *src, data_t *dest, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		dest[i] += src[i];
	}
}
/** Elementwise subtract src from dest, storing the result in dest */
void array_sub(data_t *src, data_t *dest, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		dest[i] -= src[i];
	}
}
/** Elementwise multiply src and dest, storing the result in dest */
void array_mult(data_t *src, data_t *dest, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		dest[i] *= src[i];
	}
}
/** Elementwise divide dest by src, storing the result in dest */
void array_div(data_t *src, data_t *dest, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		dest[i] /= src[i];
	}
}

/** Add the scalar val to each element in data[0..len-1] */
void array_add_scalar(data_t *data, data_t val, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		data[i] += val;
	}
}
/** Add the scalar val to each element in data[0..len-1] */
void array_sub_scalar(data_t *data, data_t val, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		data[i] -= val;
	}
}
/** Multiplies each element in data[0..len-1] by the scalar val */
void array_mult_scalar(data_t *data, data_t val, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		data[i] *= val;
	}
}
/** Multiplies each element in data[0..len-1] by the scalar val */
void array_div_scalar(data_t *data, data_t val, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		data[i] /= val;
	}
}

/** Copies src[0..len-1] to dest[0..len-1] */
void array_copy(const data_t *src, data_t *dest, unsigned int len) {
	//yes, we *could* use memcpy, but this is clearer
	for (int i = 0; i < len; i++) {
		dest[i] = src[i];
	}
}

/** Copies src[0..len-1] to dest[len-1..0] */
void array_copy_reverse(data_t *src, data_t *dest, unsigned int len) {
	int j = len - 1;
	for (int i = 0; i < len; i++, j--) {
		dest[i] = src[j];
	}
}

/** Sets each element of the array to the value specified */
void array_set_to_constant(data_t *x, data_t value, unsigned int len) {
	for (int i = 0; i < len; i++) {
		x[i] = value;
	}
}

/** Writes the elements of src to dest such that
 * dest[i] = src[ floor(i*srcLen/destLen) ]; note that this function does no
 * filtering of any kind */
void array_resample(const data_t *src, data_t *dest, unsigned int srcLen, unsigned int destLen) {
	unsigned int srcIdx;
	data_t scaleFactor = ((double)srcLen) / destLen;
	for( unsigned int i = 0; i < destLen; i++) {
		srcIdx = i * scaleFactor;
		dest[i] = src[srcIdx];
	}
}

/** Returns 1 if elements 0..(len-1) of x and y are equal, else 0 */
short int array_equal(const data_t *x, const data_t *y, unsigned int len) {
	for (int i = 0; i < len; i++) {
		if ( fabs(x[i] - y[i]) > .00001 ) return 0;
	}
	return 1;
}

/** Returns 1 if elements 0..(len-1) of x and y are equal, else 0 */
short int intarray_equal(const int *x, const int *y, unsigned int len) {
	for (int i = 0; i < len; i++) {
		if ( x[i] - y[i] != 0) return 0;
	}
	return 1;
}

/** Prints: "[ data[0] data[1] ... data[n] ]" on its own line */
void array_print(const data_t *x, unsigned int len) {
	printf("[");
	for (unsigned int i = 0; i < len; i++) {
		printf("%.3f ", x[i]);
	}
	printf("]\n");
}

/** Prints: "[ data[0] data[1] ... data[n] ]" on its own line */
void intarray_print(const int *x, unsigned int len) {
	printf("[");
	for (unsigned int i = 0; i < len; i++) {
		printf("%d ", x[i]);
	}
	printf("]\n");
}

/** Prints: "[ data[0] data[1] ... data[n] ]" on its own line */
void bytearray_print(const uint8_t *x, unsigned int len) {
	printf("[");
	for (unsigned int i = 0; i < len; i++) {
		printf("%d ", x[i]);
	}
	printf("]\n");
}

/** Prints: "<name>:\t[ data[0] data[1] ... data[n] ]" on its own line */
void array_print_with_name(const data_t *x, unsigned int len, const char* name) {
	printf("%s:\t",name);
	array_print(x, len);
}

/** Prints: "<name>:\t[ data[0] data[1] ... data[n] ]" on its own line */
void intarray_print_with_name(const int *x, unsigned int len, const char* name) {
	printf("%s:\t",name);
	intarray_print(x, len);
}

/** Prints: "<name>:\t[ data[0] data[1] ... data[n] ]" on its own line */
void bytearray_print_with_name(const uint8_t *x, unsigned int len, const char* name) {
	printf("%s:\t",name);
	bytearray_print(x, len);
}
