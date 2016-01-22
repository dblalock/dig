//
//  test_ucr_funcs.c
//  edtw
//
//  Created By <Anonymous> on 1/12/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "test_ucr_funcs.h"
#include "ucr_funcs.h"
#include "distance_utils.h"
#include "array_utils.h"
#include "flags.h"

///used for comparisons ignoring slight floating point errors
static const data_t EQUALITY_THRESHOLD = .00001;

void test_ucr_funcs_all() {
	printf("================================ Testing UCR Funcs\n");
	Znormalize_meanIsZero();
	Znormalize_varianceIsOne();
	
	Envelope_Warp0_Correct();
	Envelope_Warp2_Correct();
	
#ifdef TEST_UNIFORM_SCALING_ENVELOPE
	USEnvelope_noScaling_correct();
	USEnvelope_scaleDownOnly_correct();
	USEnvelope_scaleUpOnly_correct();
	USEnvelope_scaleUpAndDown_correct();
#endif
	
	EuclideanDist_FullComparison_correctDistance();
	EuclideanDist_EarlyAbandon_correctDistance();
	
	EuclideanSearch_BufferShorterThanQuery_ReturnsFailure();
	EuclideanSearch_QueryLenZero_ReturnsFailure();
	EuclideanSearch_QueryLenNegative_ReturnsFailure();
	EuclideanSearch_BufferLenZero_ReturnsFailure();
	EuclideanSearch_BufferLenNegative_ReturnsFailure();
	EuclideanSearch_NullQuery_ReturnsFailure();
	EuclideanSearch_NullBuffer_ReturnsFailure();
	EuclideanSearch_NullResult_ReturnsFailure();
	EuclideanSearch_EqualLenArrays_Correct();
	EuclideanSearch_DifferentLenArrays_Correct();
	
	USDist_OneLen_ReturnsEuclideanDist();
	USDist_MaxLenEqualsM_CorrectDistance();
	USDist_MinLenEqualsM_CorrectDistance();
	USDist_MinLessAndMaxGreater_CorrectDistance();
	
	DTWDist_NoWarpFullComparison_correctDistance();
	DTWDist_Warp1FullComparison_correctDistance();
	DTWDist_Warp2FullComparison_correctDistance();
	DTWDist_Warp1FullComparison_NeedsMoreWarp_correctDistance();
	DTWDist_NoWarpEarlyAbandon_correctDistance();
	
	DTWSearch_BufferShorterThanQuery_ReturnsFailure();
	DTWSearch_QueryLenZero_ReturnsFailure();
	DTWSearch_QueryLenNegative_ReturnsFailure();
	DTWSearch_BufferLenZero_ReturnsFailure();
	DTWSearch_BufferLenNegative_ReturnsFailure();
	DTWSearch_NullQuery_ReturnsFailure();
	DTWSearch_NullBuffer_ReturnsFailure();
	DTWSearch_NullResult_ReturnsFailure();
	DTWSearch_NegativeWarp_ReturnsFailure();
	DTWSearch_NoWarp_EqualLenArrays_Correct();
	DTWSearch_Warp_EqualLenArrays_Correct();
	DTWSearch_DifferentLenArrays_Correct();
	
	printf("================================ End UCR Funcs Test: Success\n");
}

//================================================================
// UTILITY FUNCTIONS
//================================================================

short int approxEqual(data_t a, data_t b) {
	return fabs(a - b) < EQUALITY_THRESHOLD;
}

//================================================================
// TEST SUPPORTING FUNCTIONS
//================================================================

void Znormalize_meanIsZero() {
	unsigned int len = 6;
	data_t x[] = {.7, 0, -.3, 11, -.6, 4};
	znormalize(x, len);
	data_t mean = array_mean(x, len);
	
	if (! approxEqual(mean, 0)) {
		printf("Error: z-normalized mean: %.3fnot 0\n", mean);
		assert(0);
	}
}

void Znormalize_varianceIsOne() {
	unsigned int len = 6;
	data_t x[] = {.7, 0, -.3, 11, -.6, 4};
	znormalize(x, len);
	data_t variance = array_variance(x, len);
	
	if (! approxEqual(variance, 1)) {
		printf("Error: z-normalized variance: %.3f not 1\n", variance);
		assert(0);
	}
}

void Envelope_Warp0_Correct() {
	float r = 0;
	int len = 17;
	data_t x[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};
	
	data_t lTest[len];
	data_t uTest[len];
	build_envelope(x,len,r,lTest,uTest);
	
	//warping width of 0 --> envelope = original sequence
	for (int i = 0; i < len; i++) {
		if (x[i] != uTest[i]) {
			printf("Error: upper envelope incorrect at index %d\n",i);
			array_print(uTest, len);
			assert(0);
		}
		if (x[i] != lTest[i]) {
			printf("Error: lower envelope incorrect at index %d\n",i);
			array_print(lTest, len);
			assert(0);
		}
	}
}

void Envelope_Warp2_Correct() {
	float r = 2;
	int len = 17;
	
	//u[i] = max( {x[i-r],x[i-r+1],...,x[i],...x[i+r-1],x[i+r]} )
	//l[i] = min( {x[i-r],x[i-r+1],...,x[i],...x[i+r-1],x[i+r]} )
	//	(but with indices outside array obviously not included)
	data_t x[] = {0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0};
	data_t l[] = {0, 0, 0, 0, 0, 1, 2, 1, 0, 0,-2,-3,-3,-3,-3,-3,-2};
	data_t u[] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, 0, 0};

	data_t lTest[len];
	data_t uTest[len];
	build_envelope(x,len,r,lTest,uTest);
	
	for (int i = 0; i < len; i++) {
		if (u[i] != uTest[i]) {
			printf("Error: upper envelope incorrect at index %d\n",i);
			array_print(uTest, len);
			assert(0);
		}
		if (l[i] != lTest[i]) {
			printf("Error: lower envelope incorrect at index %d\n",i);
			array_print(lTest, len);
			assert(0);
		}
	}
}

void USEnvelope_noScaling_correct() {
	unsigned int len = 6;
	data_t minScaling = 1;
	data_t maxScaling = 1;
	data_t x[] = {1,2,3,4,5,6};
	data_t lTruth[] = {1,2,3,4,5,6};
	data_t uTruth[] = {1,2,3,4,5,6};
	
	unsigned int minLen = minScaling*len;
	unsigned int maxLen = maxScaling*len;
//	znormalize(lTruth, minLen);
//	znormalize(uTruth, minLen);
	data_t l[minLen];
	data_t u[minLen];
	
	build_unifscale_envelope(x, len, minLen, maxLen, l, u);
	short int lCorrect = array_equal(l, lTruth, minLen);
	short int uCorrect = array_equal(u, uTruth, minLen);
	
	if (! lCorrect) {
		printf("TEST FAILED: lower US envelope, no scaling, incorrect\n");
		array_print_with_name(l, minLen, "lower");
		assert(0);
	}
	if (! uCorrect) {
		printf("TEST FAILED: upper US envelope, no scaling, incorrect\n");
		array_print_with_name(u, minLen, "upper");
		assert(0);
	}
}

void USEnvelope_scaleDownOnly_correct() {
	unsigned int len = 6;
	data_t minScaling = .5;
	data_t maxScaling = 1;
	data_t x[] = {7,2,3,4,5,6};
	data_t lTruth[] = {7,2,3};
	data_t uTruth[] = {7,3,5};

	unsigned int minLen = minScaling*len;
	unsigned int maxLen = maxScaling*len;
//	znormalize(lTruth, minLen);
//	znormalize(uTruth, minLen);
	data_t l[minLen];
	data_t u[minLen];
	
	printf("------------------\n");
	array_print_with_name(lTruth, minLen, "lTruth");
	array_print_with_name(uTruth, minLen, "uTruth");
	
	build_unifscale_envelope(x, len, minLen, maxLen, l, u);
	short int lCorrect = array_equal(l, lTruth, minLen);
	short int uCorrect = array_equal(u, uTruth, minLen);
	
	if (! lCorrect) {
		printf("TEST FAILED: lower US envelope, no scaling, incorrect\n");
		array_print_with_name(l, minLen, "lower");
		assert(0);
	}
	if (! uCorrect) {
		printf("TEST FAILED: upper US envelope, no scaling, incorrect\n");
		array_print_with_name(u, minLen, "upper");
		assert(0);
	}
}

void USEnvelope_scaleUpOnly_correct() {
	unsigned int len = 6;
	data_t minScaling = 1;
	data_t maxScaling = 2;
	data_t x[] = {7,2,3,4,6,5};
	data_t lTruth[] = {7, 2,2,2, 3,3};
	data_t uTruth[] = {7,7, 3, 4, 6,6};
	
	unsigned int minLen = minScaling*len;
	unsigned int maxLen = maxScaling*len;
	//	znormalize(lTruth, minLen);
	//	znormalize(uTruth, minLen);
	data_t l[minLen];
	data_t u[minLen];
	
	printf("------------------\n");
	array_print_with_name(lTruth, minLen, "lTruth");
	array_print_with_name(uTruth, minLen, "uTruth");
	
	build_unifscale_envelope(x, len, minLen, maxLen, l, u);
	short int lCorrect = array_equal(l, lTruth, minLen);
	short int uCorrect = array_equal(u, uTruth, minLen);
	
	if (! lCorrect) {
		printf("TEST FAILED: lower US envelope, no scaling, incorrect\n");
		array_print_with_name(l, minLen, "lower");
		assert(0);
	}
	if (! uCorrect) {
		printf("TEST FAILED: upper US envelope, no scaling, incorrect\n");
		array_print_with_name(u, minLen, "upper");
		assert(0);
	}
}

void USEnvelope_scaleUpAndDown_correct() {
	unsigned int len = 6;
	data_t minScaling = .5;
	data_t maxScaling = 3;
	data_t x[] = {7,2,3,4,6,5};
	data_t lTruth[] = {7,2,2};
	data_t uTruth[] = {7,7,7};
	
	unsigned int minLen = minScaling*len;
	unsigned int maxLen = maxScaling*len;
	//	znormalize(lTruth, minLen);
	//	znormalize(uTruth, minLen);
	data_t l[minLen];
	data_t u[minLen];
	
	printf("------------------\n");
	array_print_with_name(lTruth, minLen, "lTruth");
	array_print_with_name(uTruth, minLen, "uTruth");
	
	build_unifscale_envelope(x, len, minLen, maxLen, l, u);
	short int lCorrect = array_equal(l, lTruth, minLen);
	short int uCorrect = array_equal(u, uTruth, minLen);
	
	if (! lCorrect) {
		printf("TEST FAILED: lower US envelope, no scaling, incorrect\n");
		array_print_with_name(l, minLen, "lower");
		assert(0);
	}
	if (! uCorrect) {
		printf("TEST FAILED: upper US envelope, no scaling, incorrect\n");
		array_print_with_name(u, minLen, "upper");
		assert(0);
	}
}

//================================================================
// TEST EUCLIDEAN DISTANCE FUNCTION
//================================================================

void EuclideanDist_FullComparison_correctDistance() {
	unsigned int len = 6;
	
	data_t x[] = {0,5,1,4,2,3};
	znormalize(x, len);
	idx_t order[] = {0,1,2,3,4,5};
	data_t y[] = {0,1,2,-3,0,1};
	
	//compute z-normalized y
	data_t yNorm[len];
	array_copy(y, yNorm, len);
	znormalize(yNorm, len);
	
	//find distance between x and z-normalized y manually
	data_t dist = array_euclidean_distance(x, yNorm, len);
	
	data_t mean = array_mean(y, len);
	data_t std = sqrt(array_variance(y, len));
	
	data_t distTest = euclidean_dist_sq(x, y, len, mean, std, order, INFINITY);
	
	if (! approxEqual(dist, distTest)) {
		printf("Error: Euclidean distance %.3f not equal to expected dist %.3f.\n",distTest, dist);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

void EuclideanDist_EarlyAbandon_correctDistance() {
	unsigned int len = 6;
	unsigned int abandonAfter = 3;
	
	data_t x[] = {0,5,1,4,2,3};
	znormalize(x, len);
	idx_t order[] = {0,1,2,3,4,5};
	data_t y[] = {0,1,2,-3,0,1};
	
	//compute z-normalized y
	data_t yNorm[len];
	array_copy(y, yNorm, len);
	znormalize(yNorm, len);
	
	//find distance between x and z-normalized y manually
	data_t dist = array_euclidean_distance(x, yNorm, abandonAfter);
	
	data_t mean = array_mean(y, len);
	data_t std = sqrt(array_variance(y, len));
	
	data_t distTest = euclidean_dist_sq(x, y, len, mean, std, order, dist);
	
	if (! approxEqual(dist, distTest)) {
		printf("Error: Euclidean distance %.3f not equal to expected dist %.3f.\n",distTest, dist);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

//================================================================
// TEST EUCLIDEAN SEARCH FUNCTION
//================================================================

void EuclideanSearch_BufferShorterThanQuery_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 4;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_QueryLenZero_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 0;
	int n = 9;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_QueryLenNegative_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = -1;
	int n = 9;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_BufferLenZero_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 0;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_BufferLenNegative_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = -1;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_NullQuery_ReturnsFailure() {
	data_t *q = NULL;
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 9;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_NullBuffer_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t *buffer = NULL;
	int m = 5;
	int n = 9;
	Index result;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kFAILURE);
}
void EuclideanSearch_NullResult_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 9;
	Index *resultPtr = NULL;
	
	int retVal = euc_search(q, buffer, m, n, resultPtr);
	
	assert(retVal == kFAILURE);
}

void EuclideanSearch_EqualLenArrays_Correct() {
	data_t q[] = {1,2,3,4,5};	   // becomes [-1.414 -0.707 0.000 0.707 1.414 ]
	data_t buffer[] = {5,4,3,2,1}; // becomes [1.414 0.707 0.000 -0.707 -1.414 ]
	int m = 5;
	int n = 5;
	Index result;
	
	long correctStartLoc = 0;
	data_t correctDist = 20; // 1st and last: (sqrt(2) - (-sqrt(2)))^2	   = 8
							 // 2nd and 4th: (sqrt(2)/2 - (-sqrt(2)/2))^2  = 2
							 // 3rd = (0 - 0)^2 = 0
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kSUCCESS);
	
	if (correctStartLoc != result.index) {
		printf("Error: Incorrect start loc %ld; expected %ld\n",
			   result.index, correctStartLoc);
		assert(0);
	}
	if (! approxEqual(correctDist, result.value)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   result.value, correctDist);
		data_t *matchStartIdx = buffer + result.index;
		znormalize(q, m);
		znormalize(matchStartIdx, m);
		printf("query:\t");		array_print(q, m);
		printf("match:\t");		array_print(matchStartIdx, m);
		assert(0);
	}
}
void EuclideanSearch_DifferentLenArrays_Correct() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 9;
	Index result;

	long correctStartLoc = 2;
	data_t correctDist = 0;
	
	int retVal = euc_search(q, buffer, m, n, &result);
	
	assert(retVal == kSUCCESS);
	
	if (correctStartLoc != result.index) {
		printf("Error: Incorrect start loc %ld; expected %ld\n",
			   result.index, correctStartLoc);
		assert(0);
	}
	if (! approxEqual(correctDist, result.value)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   result.value, correctDist);
		data_t *matchStartIdx = buffer + result.index;
		znormalize(q, m);
		znormalize(matchStartIdx, m);
		printf("query:\t");		array_print(q, m);
		printf("match:\t");		array_print(matchStartIdx, m);
		assert(0);
	}
}

//================================================================
// TEST DTW FUNCTION ITSELF
//================================================================

void DTWDist_NoWarpFullComparison_correctDistance() {
	unsigned int len = 5;
	int r = 0;
	data_t x[] = {1,2,3,4,5};
	data_t y[] = {5,4,3,2,1};
	
	// initialize dummy LB_Keogh lower bound
	data_t cb[len];
	array_set_to_constant(cb, 0, len);

	//find distance between x and z-normalized y manually and via DTW
	data_t distTrue = array_euclidean_distance(x, y, len);
	data_t distTest = dtw(x, y, cb, len, r, INFINITY);
	
	if (! approxEqual(distTrue, distTest)) {
		printf("Error: DTW distance %.3f not equal to expected dist %.3f.\n",
			   distTest, distTrue);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

void DTWDist_Warp1FullComparison_correctDistance() {
	unsigned int len = 6;
	int r = 1;
	data_t x[] = {1,2,3,3,4,5};
	data_t y[] = {1,1,2,3,4,5};
	data_t distTrue = 0;
	
	// initialize dummy LB_Keogh lower bound
	data_t cb[len];
	array_set_to_constant(cb, 0, len);
	
	data_t distTest = dtw(x, y, cb, len, r, INFINITY);
	
	if (! approxEqual(distTrue, distTest)) {
		printf("Error: DTW distance %.3f not equal to expected dist %.3f.\n",
			   distTest, distTrue);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

void DTWDist_Warp2FullComparison_correctDistance() {
	unsigned int len = 6;
	int r = 2;
	data_t x[] = {5,1,3,3,3,4,5};
	data_t y[] = {5,1,3,4,4,4,5};
	data_t distTrue = 0;
	
	// initialize dummy LB_Keogh lower bound
	data_t cb[len];
	array_set_to_constant(cb, 0, len);
	
	data_t distTest = dtw(x, y, cb, len, r, INFINITY);
	
	if (! approxEqual(distTrue, distTest)) {
		printf("Error: DTW distance %.3f not equal to expected dist %.3f.\n",
			   distTest, distTrue);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

void DTWDist_Warp1FullComparison_NeedsMoreWarp_correctDistance() {
	unsigned int len = 7;
	int r = 1;
	data_t x[] = {5,1,3,3,3,4,5}; // if r =2, dist = 0
	data_t y[] = {5,1,3,4,4,4,5};
	data_t distTrue = 1; // x[5] with y[4] --> (3-4)^2 = 1
	
	// initialize dummy LB_Keogh lower bound
	data_t cb[len];
	array_set_to_constant(cb, 0, len);
	
	data_t distTest = dtw(x, y, cb, len, r, INFINITY);
	
	if (! approxEqual(distTrue, distTest)) {
		printf("Error: DTW distance %.3f not equal to expected dist %.3f.\n",
			   distTest, distTrue);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

void DTWDist_NoWarpEarlyAbandon_correctDistance() {
	unsigned int len = 5;
	unsigned int abandonAfter = 3;
	int r = 0;
	data_t x[] = {1,2,3,4,5};
	data_t y[] = {5,4,3,2,1};
	
	// initialize dummy LB_Keogh lower bound
	data_t cb[len];
	array_set_to_constant(cb, 0, len);
	
	//find distance between x and z-normalized y manually and via DTW
	data_t distTrue = array_euclidean_distance(x, y, abandonAfter);
	data_t distTest = dtw(x, y, cb, len, r, distTrue);
	
	if (! approxEqual(distTrue, distTest) || distTest < distTrue) {
		printf("Error: DTW distance %.3f not equal to expected dist %.3f.\n",
			   distTest, distTrue);
		array_print(x, len);
		array_print(y, len);
		assert(0);
	}
}

//================================================================
// TEST DTW SEARCH FUNCTIONS
//================================================================

void DTWSearch_BufferShorterThanQuery_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 4;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_QueryLenZero_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 0;
	int n = 9;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_QueryLenNegative_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = -1;
	int n = 9;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_BufferLenZero_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 0;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_BufferLenNegative_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = -1;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_NullQuery_ReturnsFailure() {
	data_t *q = NULL;
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 9;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_NullBuffer_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t *buffer = NULL;
	int m = 5;
	int n = 9;
	float r = .2;
	Index result;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_NullResult_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 9;
	float r = .2;
	Index *resultPtr = NULL;
	
	int retVal = dtw_search(q, buffer, m, n, r, resultPtr);
	
	assert(retVal == kFAILURE);
}
void DTWSearch_NegativeWarp_ReturnsFailure() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {1,0,-2,-1,0,1,2,4,3};
	int m = 5;
	int n = 9;
	float r = -.05;
	Index *resultPtr = NULL;
	
	int retVal = dtw_search(q, buffer, m, n, r, resultPtr);
	
	assert(retVal == kFAILURE);
}

void DTWSearch_NoWarp_EqualLenArrays_Correct() {
	data_t q[] = {1,2,3,4,5};	   // becomes [-1.414 -0.707 0.000 0.707 1.414 ]
	data_t buffer[] = {5,4,3,2,1}; // becomes [1.414 0.707 0.000 -0.707 -1.414 ]
	int m = 5;
	int n = 5;
	float r = 0;
	Index result;
	
	long correctStartLoc = 0;
	data_t correctDist = 20; // 1st and last: (sqrt(2) - (-sqrt(2)))^2	   = 8
							 // 2nd and 4th: (sqrt(2)/2 - (-sqrt(2)/2))^2  = 2
							 // 3rd = (0 - 0)^2 = 0
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kSUCCESS);
	
	if (correctStartLoc != result.index) {
		printf("Error: Incorrect start loc %ld; expected %ld\n",
			   result.index, correctStartLoc);
		assert(0);
	}
	if (! approxEqual(correctDist, result.value)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   result.value, correctDist);
		data_t *matchStartIdx = buffer + result.index;
		znormalize(q, m);
		znormalize(matchStartIdx, m);
		printf("query:\t");		array_print(q, m);
		printf("match:\t");		array_print(matchStartIdx, m);
		assert(0);
	}
}

void DTWSearch_Warp_EqualLenArrays_Correct() {
	data_t q[] = {1,2,3,4,4,5};
	data_t buffer[] = {1,2,2,3,4,5};
	int m = 6;
	int n = 6;
	float r = 1;
	Index result;

	long correctStartLoc = 0;
	data_t correctDist = 0;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kSUCCESS);
	
	if (correctStartLoc != result.index) {
		printf("Error: Incorrect start loc %ld; expected %ld\n",
			   result.index, correctStartLoc);
		assert(0);
	}
	if (! approxEqual(correctDist, result.value)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   result.value, correctDist);
		data_t *matchStartIdx = buffer + result.index;
		znormalize(q, m);
		znormalize(matchStartIdx, m);
		printf("query:\t");		array_print(q, m);
		printf("match:\t");		array_print(matchStartIdx, m);
		assert(0);
	}
}

void DTWSearch_DifferentLenArrays_Correct() {
	data_t q[] = {1,1,2,3,4,5};
	data_t buffer[] = {5,3,-4,-2,0,0,2,4,1,6};
	int m = 6;
	int n = 10;
	float r = 1;
	Index result;
	
	long correctStartLoc = 2;
	data_t correctDist = 0;
	
	int retVal = dtw_search(q, buffer, m, n, r, &result);
	
	assert(retVal == kSUCCESS);
	
	if (correctStartLoc != result.index) {
		printf("Error: Incorrect start loc %ld; expected %ld\n",
			   result.index, correctStartLoc);
		assert(0);
	}
	if (! approxEqual(correctDist, result.value)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   result.value, correctDist);
		data_t *matchStartIdx = buffer + result.index;
		znormalize(q, m);
		znormalize(matchStartIdx, m);
		printf("query:\t");		array_print(q, m);
		printf("match:\t");		array_print(matchStartIdx, m);
		assert(0);
	}
}

void USDist_OneLen_ReturnsEuclideanDist() {
	data_t q[] = {1,2,3,4,5};
	data_t buffer[] = {-3,-1,1,3,5};
	int m = 5;
	int minLen = m;
	int maxLen = m;
	data_t bsf = INFINITY;
	
	data_t mean = array_mean(buffer, minLen);
	data_t std = sqrt( array_variance(buffer, minLen) );
	
	us_query * query = us_query_new(q, m, minLen, maxLen);
	data_t dist = us_distance(query, buffer, mean, std, bsf);
	
	znormalize(q, minLen);
	znormalize(buffer, minLen);
	data_t correctDist = array_euclidean_distance(q, buffer, minLen);
	
	if ( ! approxEqual(correctDist, dist)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   dist, correctDist);
		array_print_with_name(q, minLen, "q");
		array_print_with_name(buffer, minLen, "x");
		assert(0);
	}
}

void USDist_MaxLenEqualsM_CorrectDistance() {
//	printf("--------USDist_MaxLenEqualsM_CorrectDistance--------\n");
	data_t q[]		= {1,2,3,4,5,6};
	data_t buffer[] = {5,3,4,1,2,3};	//last 3 match when normalized
	int m = 6;
	int n = 6;
	int minLen = 3;
	int maxLen = m;
	data_t bsf = INFINITY;
	
	data_t *minLenDataStart = buffer + maxLen - minLen;
	data_t mean = array_mean(minLenDataStart, minLen);
	data_t std = sqrt( array_variance(minLenDataStart, minLen) );
	
	us_query * query = us_query_new(q, m, minLen, maxLen);
	data_t dist = us_distance(query, buffer, mean, std, bsf);
	
	data_t *correctDataStart = buffer + maxLen - minLen;
	znormalize(q, minLen);
	znormalize(correctDataStart, minLen);
	data_t correctDist = array_euclidean_distance(q, correctDataStart, minLen);
	
	if ( ! approxEqual(correctDist, dist)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   dist, correctDist);
		array_print_with_name(q, minLen, "q");
		array_print_with_name(buffer, n, "x");
		assert(0);
	}
}

void USDist_MinLenEqualsM_CorrectDistance() {
//	printf("--------USDist_MinLenEqualsM_CorrectDistance--------\n");
	data_t q[]		= {3,1,2,4};
	data_t buffer[] = {13,13,13,11,11,11,12,12,12,14,14,14};	//last 3 match when normalized
	int m = 4;
	int n = 12;
	int minLen = m;
	int maxLen = 3*m;
	data_t bsf = INFINITY;
	
	data_t *minLenDataStart = buffer + maxLen - minLen;
	data_t mean = array_mean(minLenDataStart, minLen);
	data_t std = sqrt( array_variance(minLenDataStart, minLen) );
	
	us_query * query = us_query_new(q, m, minLen, maxLen);
	data_t dist = us_distance(query, buffer, mean, std, bsf);
	
	data_t correctDist = 0;
	
	if ( ! approxEqual(correctDist, dist)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   dist, correctDist);
		array_print_with_name(q, minLen, "q");
		array_print_with_name(buffer, n, "x");
		assert(0);
	}
}

void USDist_MinLessAndMaxGreater_CorrectDistance() {
//	printf("--------USDist_MinLenEqualsM_CorrectDistance--------\n");
	data_t q[]		= {3,1,2,4,5,7};
	data_t buffer[] = {5,3,16,12,4,9, 13,13,11,11,12,12,14,14,15,15,17,17};
	int m = 6;
	int n = 18;
	int minLen = 3;
	int maxLen = 3*m;
	data_t bsf = INFINITY;
	
	data_t *minLenDataStart = buffer + maxLen - minLen;
	data_t mean = array_mean(minLenDataStart, minLen);
	data_t std = sqrt( array_variance(minLenDataStart, minLen) );
	
	us_query * query = us_query_new(q, m, minLen, maxLen);
	data_t dist = us_distance(query, buffer, mean, std, bsf);
	
	data_t correctDist = 0;
	
	if ( ! approxEqual(correctDist, dist)) {
		printf("Error: Incorrect distance %.3f; expected %.3f\n",
			   dist, correctDist);
		array_print_with_name(q, minLen, "q");
		array_print_with_name(buffer, n, "x");
		assert(0);
	}
}
