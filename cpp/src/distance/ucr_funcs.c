
#include <assert.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <float.h>

#include "ucr_funcs.h"
#include "deque.h"
#include "array_utils.h"
#include "flags.h"
#include "global_test_vals.h"
#include "distance_utils.h"

// Many functions just need an array to hold tempororay data; allocate them
// ahead of time to avoid either allocating new memory all the time or passing
// in a large number of unnecessary array pointers. This *should* also make the
// window-based distance measures faster, since they can reuse one spot in
// memory instead of having to cache a different set of temporary arrays for
// each query. This structure would of course be a terrible idea in a
// multithreaded context, but we'll cross that bridge when we come to it.
static long tempArrayLen = -1;	// needs to be < 0
static data_t* datazTmp	= NULL;
static data_t* cb1Tmp	= NULL;
static data_t* cb2Tmp	= NULL;
static data_t* uTmp		= NULL;
static data_t* lTmp		= NULL;
static data_t* yTmp		= NULL;

/** Increases the size of an array to len if it's shorter than len */
static void ensureArrayLongEnough(data_t** v, long len) {
	if (len < 1) {
		printf("WARNING: tried to ensure array had len <=0\n");
		return;
	}
	if (*v==NULL) {
		*v = (data_t*) malloc(len*sizeof(data_t));
	} else {
		*v = (data_t*) realloc(*v, len*sizeof(data_t));
		if (*v == NULL) {
			printf("ERROR: failed to realloc\n");
			exit(1);
		}
	}
}

/** Increases the size of the temporary arrays used for distance computations
 * to at least len if they're shorter than len */
void ensureTempArraysLongEnough(unsigned int len) {
	if (len <= tempArrayLen) return;
	
	tempArrayLen = len;
	ensureArrayLongEnough(&datazTmp,	tempArrayLen);
	ensureArrayLongEnough(&cb1Tmp,		tempArrayLen);
	ensureArrayLongEnough(&cb2Tmp,		tempArrayLen);
	ensureArrayLongEnough(&uTmp,		tempArrayLen);
	ensureArrayLongEnough(&lTmp,		tempArrayLen);
	ensureArrayLongEnough(&yTmp,		tempArrayLen);
}

//================================================================
// UTILITY FUNCTIONS
//================================================================

#define min(x,y) ((x)<(y)?(x):(y)) //leave as macro to handle different types;
#define max(x,y) ((x)>(y)?(x):(y)) //since we can't overload in pure C

/** Computes (x-y)^2 */
static inline data_t dist(data_t x, data_t y) {
	global_dist_calls++;
	data_t diff = x - y;
	return diff * diff;
}

int32_t validateArgs(const data_t *query, const data_t *buffer, length_t m, length_t n, data_t bsf, Index *result) {
	if ( query == NULL) {
		printf("Error: could not compute distance; query NULL\n");
		return kFAILURE;
	}
	if ( buffer == NULL) {
		printf("Error: could not compute distance; buffer NULL\n");
		return kFAILURE;
	}
	if ( result == NULL) {
		printf("Error: could not compute distance; results struct NULL\n");
		return kFAILURE;
	}
	if ( m < 1) {
		printf("Error: could not compute distance; query length %d invalid\n", m);
		return kFAILURE;
	}
	if ( n < 1) {
		printf("Error: could not compute distance; buffer length %d invalid\n", n);
		return kFAILURE;
	}
	if (m > n) {
		printf("Error: could not compute distance; query length %d longer than buffer length %d\n", m, n);
		return kFAILURE;
	}
	if ( bsf < 0) {
		printf("Error: could not compute distance; best dist so far %.3f invalid\n", bsf);
		return kFAILURE;
	}
	return kSUCCESS;
}

void invalidate(Index *result) {
	if (result == NULL) return;
	result->index = kFAILURE;
	result->value = kFAILURE;
}

short int isInvalid(Index result) {
	return (result.index < 0 || result.value < 0 ||
		result.index == kFAILURE || result.value == kFAILURE);
}

//================================================================
// EUCLIDEAN DISTANCE FUNCTIONS
//================================================================

/** calculate the euclidean distance squared between a z-normalized and sorted
 * array A and an unprocessed vector B with the provided mean and standard
 * deviation */
data_t euclidean_dist_sq(const data_t* A, const data_t* B, length_t m, data_t mean, data_t std,
						 const idx_t* order, data_t bsf) {
    int i;
	data_t x;
    data_t sum = 0;
    for (i = 0 ; i < m && sum < bsf ; i++ ) {
        x = (B[order[i]]-mean)/std;
		sum += dist(x, A[i]);
//		diff = x-A[i];
//        sum += diff * diff;
    }
    return sum;
}

/** Search each subsequence in a buffer under the Euclidean distance */
int32_t euc_search(data_t* query, data_t* buffer, length_t m, length_t n, Index* result) {
	return euc_ongoing_search(query, buffer, m, n, INFINITY, result);
}

/** Search each subsequence in a buffer under the Euclidean distance, with
 * a pre-existing "best-so-far" distance for early abandoning */
int32_t euc_ongoing_search(data_t* query, const data_t* buffer, length_t m, length_t n,
						   data_t bsf, Index* result) {
	
	//sanity check args
	int validity = validateArgs(query, buffer, m, n, bsf, result);
	if (validity != kSUCCESS) {
		invalidate(result);
		return validity;
	}
	
	data_t sumx = 0;
	data_t sumx2 = 0;
	data_t x, oldx;
	data_t mean, std;
	data_t distance;
	const data_t *data_start;
	idx_t i, windowEnd;
	tick_t loc = -1;
	
	idx_t* order = (idx_t *)malloc(m * sizeof(idx_t));
	if (order == NULL) {
		printf("Error: couldn't allocate space to sort query");
		return kFAILURE;
	}
	
	//sanity check args
	if (n < m) {
		printf("Error: data length %d shorter than query lenth %d\n", n, m);
		free(order);
		return kFAILURE;
	}

	//z-normalize query
	znormalize(query, m);
	
	//compute order for early-abandoning query comparisons
    sort_abs_decreasing(query, order, m);
	
	//initialize sums for first n-1 points
	for (i = 0; i < m - 1; i++) {
		x = buffer[i];
		sumx += x;
		sumx2 += x*x;
	}
	
	//compute the euclidean distance for each subsequence
	data_start = buffer;
	for (i = m-1; i < n; i++, data_start++) {
		
		//update mean and std dev
		x = buffer[i];
		sumx += x;
		sumx2 += x*x;
		mean = sumx / m;
		std = sqrt(sumx2 / m - mean*mean);
		
		//check for new best match
		distance = euclidean_dist_sq(query, data_start, m, mean, std, order, bsf);
		if( distance < bsf ) {
			bsf = distance;
			loc = i - m + 1;
		}
		
		//remove point that's no longer in sliding window for next iteration
		windowEnd = i - m + 1;
		oldx = buffer[windowEnd];
		sumx -= oldx;
		sumx2 -= oldx*oldx;
	}
	
	result->index = loc;
	result->value = bsf;
	
	free(order);
	
	return kSUCCESS;
}

//================================================================
// UNIFORM SCALING FUNCTIONS
//================================================================


us_query *us_query_new(data_t* query, length_t m, length_t minLen, length_t maxLen) {
	int i=0, o=0, len;
	us_query *usq;
//	data_t *l, *u, *s, *envWidths;
	int numLens = maxLen - minLen + 1;
	
	data_t l[minLen];
	data_t u[minLen];
	data_t s[maxLen];
	data_t envWidths[minLen];
	
	usq = (us_query*)malloc(sizeof(us_query));
	if (usq == NULL)
		goto us_query_new_cleanup;
	usq->q = (data_t*)malloc(m*sizeof(data_t));
	if (usq->q == NULL)
		goto us_query_new_cleanup;
	usq->envOrder = (idx_t*)malloc(minLen*sizeof(idx_t));
	if (usq->envOrder == NULL)
		goto us_query_new_cleanup;
	usq->lo = (data_t*)malloc(minLen*sizeof(data_t));
	if (usq->lo == NULL)
		goto us_query_new_cleanup;
	usq->uo = (data_t*)malloc(minLen*sizeof(data_t));
	if (usq->uo == NULL)
		goto us_query_new_cleanup;
	usq->means = (data_t*)malloc(numLens*sizeof(data_t));
	if (usq->means == NULL)
		goto us_query_new_cleanup;
	usq->stds = (data_t*)malloc(numLens*sizeof(data_t));
	if (usq->stds == NULL)
		goto us_query_new_cleanup;
	usq->orders = (idx_t**)malloc(numLens*sizeof(idx_t*));
	if (usq->orders == NULL)
		goto us_query_new_cleanup;
	
	//allocate space to store the optimal ordering for every
	//possible scaling of the query
	i=0;
	for (len=minLen; len <= maxLen; len++, i++) {
		len = minLen+i;
		usq->orders[i] = (idx_t*)malloc(len*sizeof(idx_t));
		if (usq->orders[i] == NULL)
			goto us_query_new_cleanup;
	}
	
	usq->m = m;
	usq->minLen = minLen;
	usq->maxLen = maxLen;
	
	//store the (normalized) query data in the query struct
	for(i = 0; i < m; i++) {
        usq->q[i] = query[i];
    }
	znormalize(usq->q, m);
	
	// create the uniform scaling envelope for the query
	build_unifscale_envelope(usq->q, m, minLen, maxLen, l,u);
	
	// sort the envelope based on increasing width (so skinny sections get
	// compared first and fat sections get compared last; see UCR's "Atomic
	// Wedgie" (time series wedges) paper
	for(i = 0; i < minLen; i++) {
		envWidths[i] = u[i] - l[i];
	}
//	array_print_with_name(envWidths, minLen, "envWidthsUnsorted");
	sort_abs_increasing(envWidths, usq->envOrder, minLen);
//	array_print_with_name(envWidths, minLen, "envWidths");
//	intarray_print_with_name(usq->envOrder, minLen, "order");
	for(i = 0; i < minLen; i++) {
		o = usq->envOrder[i];
		usq->lo[i] = l[o];
        usq->uo[i] = u[o];
	}
	
	// precompute the means, standard deviations, and optimal
	// orders of comparison for every scaling of the query
	i = 0;
	for (len = minLen; len <= maxLen; len++) {
		array_resample(usq->q, s, m, len);
//		array_print_with_name(s, len, "s");
		usq->means[i] = array_mean(s, len);
		usq->stds[i] = sqrt( array_variance(s, len) );
		if (usq->stds[i] == 0) {
			usq->stds[i] = DBL_MIN; // don't divide by 0
		}
		
		znormalize(s, len);
		sort_abs_decreasing(s, usq->orders[i], len);
		
		i++;
	}
	
	return usq;

us_query_new_cleanup:
	us_query_free(usq);
	return NULL;
}

void us_query_free(us_query* query) {
	if (query == NULL) return;
	if (query->envOrder != NULL)
		free(query->envOrder);
	if (query->q != NULL)
		free(query->q);
	if (query->uo != NULL)
		free(query->uo);
	if (query->lo != NULL)
		free(query->lo);
	if (query->means != NULL)
		free(query->means);
	if (query->stds != NULL)
		free(query->stds);
	if (query->orders != NULL) {
		//free the order for every possible scaling
		int numLens = query->maxLen - query->minLen + 1;
		for (int i = 0; i < numLens; i++) {
			if (query->orders[i] != NULL)
				free(query->orders[i]);
		}
		free(query->orders);
	}
	free(query);
}

void build_unifscale_envelope(const data_t *x, length_t origLen, length_t minLen, length_t maxLen,
							  data_t *l, data_t *u) {
	data_t resampled[maxLen];
	
	//initialize max and min values to those of maximum upsampling
	array_resample(x, l, origLen, minLen);
#ifndef TEST_UNIFORM_SCALING_ENVELOPE
	znormalize(l, minLen);
#endif
	array_copy(l, u, minLen);
	
	//Find the highest and lowest value of the first minLen points in any scaling
	data_t val;
	length_t len, i;
	for (len = minLen+1; len <= maxLen; len++) {
		array_resample(x, resampled, origLen, len);
#ifndef TEST_UNIFORM_SCALING_ENVELOPE
		znormalize_prefix(resampled, len, minLen);
#endif
		for (i = 0; i < minLen; i++) {
			val = resampled[i];
			if (val > u[i]) {
				u[i] = val;
			} else if (val < l[i]) {
				l[i] = val;
			}
		}
	}
}

void build_swm_envelope(const data_t *x, length_t origLen, length_t minLen, length_t maxLen,
							  data_t r, data_t *l, data_t *u) {
	data_t xResampled[maxLen];
	data_t lResampled[maxLen];
	data_t uResampled[maxLen];
	data_t uVal, lVal;
	int rInt = round(r*minLen);
	
	short int noWarp = ((int)round(r*maxLen)) == 0;
	short int noScale = (minLen == maxLen);
//	printf("noWarp, noScale = %d, %d\n", noWarp, noScale);
	if (noWarp && noScale) {
		array_copy(x, xResampled, origLen);
		znormalize(xResampled, origLen);
		array_copy(xResampled, l, origLen);
		array_copy(xResampled, u, origLen);
		return;
	} else if (noWarp) {
		array_copy(x, xResampled, origLen);
		znormalize(xResampled, origLen);
		build_unifscale_envelope(xResampled, origLen, minLen, maxLen, l, u);
		return;
	} else if (noScale) {
		array_copy(x, xResampled, origLen);
		znormalize(xResampled, origLen);
		build_envelope(xResampled, origLen, rInt, l, u);
		return;
	}
	
	//initialize envelope to that of the maximum upsampling
	array_resample(x, xResampled, origLen, minLen);
	znormalize(xResampled, minLen);

	build_envelope(xResampled, minLen, rInt, l, u);
	
	//find an evelope that includes the LB_Keogh envelopes for all scalings
	length_t len, i;
	for (len = minLen+1; len <= maxLen; len++) {
		
		//scale the data to the specified length and build an envelope around it
		array_resample(x, xResampled, origLen, len);
		znormalize(xResampled, len);
		rInt = round(r*len);
		build_envelope(xResampled, len, rInt, lResampled, uResampled);
		
		//if any element of this envelope is outside the overall envelope,
		//expand the overall envelope to include it
		for (i = 0; i < minLen; i++) {
			lVal = lResampled[i];
			if (lVal < l[i]) {
				l[i] = lVal;
			}
			uVal = uResampled[i];
			if (uVal > u[i]) {
				u[i] = uVal;
			}
		}
	}
}

data_t lb_keogh_us_query(const idx_t* order, const data_t *t,
						  const data_t *uo, const data_t *lo,
						  length_t len, data_t mean, data_t std, data_t bsf) {
    data_t  lb = 0;
    data_t  x;
    int     i;
	
    for (i = 0; i < len && lb < bsf; i++) {
        x = (t[(order[i])] - mean) / std;
//		printf("x,uo,lo = %.3f,%.3f,%.3f\n",x,uo[i], lo[i]);
        if (x > uo[i]) {
            lb += dist(x, uo[i]);
        }
        else if (x < lo[i]) {
            lb += dist(x, lo[i]);
        }
    }
    return lb;
}

data_t euclidean_dist_nonnormalized(const data_t* Y, const data_t* X,
									const idx_t* order, int m,
									data_t meanY, data_t stdY,
									data_t meanX, data_t stdX, data_t bsf) {
    int i, idx;
	data_t x, y;
    data_t sum = 0;
    for (i = 0 ; i < m && sum < bsf ; i++ ) {
		idx = order[i];
        x = (X[idx]-meanX)/stdX;
		y = (Y[idx]-meanY)/stdY;
		sum += dist(x,y);
    }
    return sum;
}

data_t test_all_scalings(const data_t* query, const data_t* data,
						 const data_t* meansY, const data_t *stdsY,
						 idx_t**const orders,
						 length_t m, length_t minLen, length_t maxLen,
						 data_t meanX, data_t stdX, data_t bsf,
						 length_t* bestLen) {
	length_t len;
	idx_t i, j, idx;
	idx_t scaledIdx;
	data_t newXinWindow, delta;
	data_t d, x, y;
	data_t scaleFactor;
	data_t mf = (data_t) m;
	const data_t* dataStart = data + maxLen - minLen;
//	data_t Y[maxLen];
	data_t err = (stdX * stdX) * minLen;		// sigma^2 = err / n;
	data_t meanY = meansY[0];
	data_t stdY = stdsY[0];
	idx_t* order = orders[0];
	
//	ensureArrayLongEnough(&yTmp, maxLen);
	ensureTempArraysLongEnough(maxLen);
	
	// initially assume that the best scaling is the minimum-length scaling
	// and alter this assumption as better scalings are found
	*bestLen = minLen;
	
	// compute distance to minimum scaling; this is separate from the
	// main loop since we don't have to update the mean and standard
	// deviation
	array_resample(query, yTmp, m, minLen);
//	array_print_with_name(Y, minLen, "yMinLen");
//	array_print_with_name(dataStart, minLen, "xMinLen");
//	d = euclidean_dist_fixed_order(yTmp, dataStart, minLen,
//									  meanY, stdY,
//									  meanX, stdX,bsf);
	d = euclidean_dist_nonnormalized(yTmp, dataStart,
									 order, minLen,
									 meanY, stdY,
									 meanX, stdX,bsf);
//	printf("len, dist = %d, %.3f\n", minLen, dist);
//	printf("meanY, stdY = %.3f, %.3f\n",meanY, stdY);
	if (d < bsf) {
		bsf = d;
	}
	
	// calculate the distance to every possible scaling of the query,
	// abandoning early whenever possible. Note that we not only
	// z-normalize online, but also *resample* online for the greatest
	// possible performance
	i=1; // not 0 since we already tried the minimum scaling
	for (len = minLen+1; len <= maxLen; len++) {
		// pull out stats needed to resample and normalize y during the
		// distance calculation
		scaleFactor = mf / len;
		meanY = meansY[i];
		stdY = stdsY[i];
		
		// get new data val that gets added at this scaling
		dataStart--;
		newXinWindow = *dataStart;

		// update mean and std of data using Knuth's online algorithm
		delta	= newXinWindow - meanX;
		meanX	= meanX + delta / len;
		err		= err + delta * (newXinWindow - meanX);
		stdX	= sqrt(err / len);

		//compute early-abandoning euclidean distance for this scaling
		order = orders[i];
		d = 0;
		for (j = 0; j < len && d < bsf; j++) {
			idx = order[j];
			scaledIdx = idx * scaleFactor;
			x = (dataStart[idx] - meanX) / stdX;
			y = (query[scaledIdx] - meanY) / stdY;
			d += dist(x,y);
		}

		if (d < bsf) {
			bsf = d;
			*bestLen = len;
		}
		
		i++;
	}
	return bsf;
}

data_t us_distance_internal(const data_t *query, idx_t *envOrder,
							const data_t* data,
							const data_t* meansY, const data_t* stdsY,
							idx_t ** const orders,
							length_t m, length_t minLen, length_t maxLen,
							const data_t *uo, const data_t *lo,
							data_t mean, data_t std, data_t bsf,
							short int useEnvelope, length_t* bestLen) {

	//try to avoid testing individual scalings by computing an LB_Keogh distance
	if (useEnvelope) {
		const data_t* dataCompareStart = data + maxLen - minLen;
		data_t lb_k = lb_keogh_us_query(envOrder, dataCompareStart, uo, lo, minLen,
										mean, std, bsf);
		if (lb_k >= bsf) {
			return lb_k;
		}
	}
	
	return test_all_scalings(query, data, meansY, stdsY,
							 orders,
							 m, minLen, maxLen,
							 mean, std, bsf, bestLen);
}

data_t us_distance_and_len(const us_query *query, const data_t *buffer,
						   data_t mean, data_t std, data_t bsf, length_t* bestLen) {
	short int useEnvelope = 0; //false; no envelope
	return us_distance_and_len_envelope(query, buffer, mean, std, bsf, useEnvelope, bestLen);
}

//NOTE that the mean and std supplied need to be for the minimum length
//subsequence of the data
data_t us_distance(const us_query *query, const data_t *buffer,
				   data_t mean, data_t std, data_t bsf) {
	short int useEnvelope = 0; //false; no envelope
	return us_distance_envelope(query, buffer, mean, std, bsf, useEnvelope);
}

data_t us_distance_envelope(const us_query *query, const data_t *buffer,
							data_t mean, data_t std, data_t bsf,
							short int useEnvelope) {
	length_t dummy;
	return us_distance_and_len_envelope(query, buffer, mean, std, bsf, useEnvelope, &dummy);
}

data_t us_distance_and_len_envelope(const us_query *query, const data_t *buffer,
						   data_t mean, data_t std, data_t bsf,
						   short int useEnvelope, length_t* bestLen) {
	return us_distance_internal(query->q, query->envOrder, buffer,
								query->means, query->stds,
								query->orders,
								query->m, query->minLen, query->maxLen,
								query->uo, query->lo,
								mean, std, bsf,
								useEnvelope, bestLen);
}

//================================================================
// DTW PRECOMPUTATION FUNCTIONS
//================================================================

/// Finding the envelop of min and max value for LB_Keogh
/// Implementation idea is intoruduced by Danial Lemire in his paper
/// "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
void build_envelope(const data_t *t, tick_t len, length_t r, data_t *l, data_t *u) {
    deque du, dl;
    tick_t i = 0;
	length_t width = 2 * r + 1;
	
    deq_new(&du, width + 1);
    deq_new(&dl, width + 1);
	
    deq_push_back(&du, 0);
    deq_push_back(&dl, 0);
	
    for (i = 1; i < len; i++) {
        if (i > r) {
            u[i - r - 1] = t[deq_front(&du)];
            l[i - r - 1] = t[deq_front(&dl)];
        }
        if (t[i] > t[i - 1]) {
            deq_pop_back(&du);
            while (!deq_empty(&du) && t[i] > t[deq_back(&du)]) {
                deq_pop_back(&du);
            }
        } else {
            deq_pop_back(&dl);
            while (!deq_empty(&dl) && t[i] < t[deq_back(&dl)]) {
                deq_pop_back(&dl);
            }
        }
		
        deq_push_back(&du, i);
        deq_push_back(&dl, i);
        if (i == width + deq_front(&du)) {
            deq_pop_front(&du);
        } else if (i == width + deq_front(&dl)) {
            deq_pop_front(&dl);
        }
    }
	
    for (i = len; i < len + r + 1; i++) {
        u[i - r - 1] = t[deq_front(&du)];
        l[i - r - 1] = t[deq_front(&dl)];
        if (i - deq_front(&du) >= width) {
            deq_pop_front(&du);
        }
        if (i - deq_front(&dl) >= width) {
            deq_pop_front(&dl);
        }
    }
	
    deq_free(&du);
    deq_free(&dl);
}

dtw_query* dtw_query_new(data_t *query, length_t m, float r) {
    int32_t		i = 0, o = 0;
    dtw_query	*udq=NULL;
	data_t *l=NULL, *u=NULL;
	
    udq = (dtw_query*)malloc(sizeof(dtw_query));
    if(udq == NULL)
        goto query_new_cleanup;
	
    udq->q = (data_t *)malloc(sizeof(data_t) * m);
    if(udq->q == NULL)
        goto query_new_cleanup;
	
    udq->qo = (data_t *)malloc(sizeof(data_t) * m);
    if(udq->qo == NULL)
        goto query_new_cleanup;
	
    udq->lo = (data_t *)malloc(sizeof(data_t) * m);
    if(udq->lo == NULL)
        goto query_new_cleanup;
	
    udq->uo = (data_t *)malloc(sizeof(data_t) * m);
    if(udq->uo == NULL)
        goto query_new_cleanup;
	
    udq->order = (idx_t *)malloc(sizeof(idx_t) * m);
    if(udq->order == NULL)
        goto query_new_cleanup;
	
    l = (data_t *)malloc(sizeof(data_t) * m);
    if(l == NULL)
        goto query_new_cleanup;

    u = (data_t *)malloc(sizeof(data_t) * m);
    if(u == NULL)
        goto query_new_cleanup;
	
    // store query length
    udq->m = m;
	
	// store warping width as integer (convert if fraction of m)
    if (r <= 1) {
        udq->r = round(r * m);
    } else {
        udq->r = floor(r);
	}
	
	//store the raw query data in the query struct
	for(i = 0; i < m; i++) {
        udq->q[i] = query[i];
		udq->qo[i] = query[i];
    }
	
	//z-normalize the query in place
	znormalize(udq->q, udq->m);
	znormalize(udq->qo, udq->m);
	
    // Create envelope for the query: lower envelope, l, and upper envelope, u
    build_envelope(udq->q, udq->m, udq->r, l, u);
	
	// Sort the normalized query by decreasing absolute value in place
	sort_abs_decreasing(udq->qo, udq->order, udq->m);
	
    // also initialize arrays for keeping the sorted envelope
	for(i = 0; i < m; i++) {
		o = udq->order[i];
		udq->qo[i] = udq->q[o];
		udq->lo[i] = l[o];
        udq->uo[i] = u[o];
	}
	
	free(l);
	free(u);
	
    return udq;
	
query_new_cleanup:
	if (l != NULL) free(l);
	if (u != NULL) free(u);
    dtw_query_free(udq);
	
    return NULL;
}

void dtw_query_free(dtw_query* query) {
    if(query == NULL)
        return;
    if(query->q != NULL)
        free(query->q);
	if(query->qo != NULL)
        free(query->qo);
	if(query->uo != NULL)
        free(query->uo);
    if(query->lo != NULL)
        free(query->lo);
	if(query->order != NULL)
		free(query->order);
	free(query);
}

//================================================================
// DTW DISTANCE FUNCTIONS
//================================================================

/// Calculate quick lower bound
/// Usually, LB_Kim take time O(m) for finding top,bottom,fist and last.
/// However, because of z-normalization the top and bottom cannot give siginifant benefits.
/// And using the first and last points can be computed in constant time.
/// The prunning power of LB_Kim is non-trivial, especially when the query is not long, say in length 128.
data_t ucr_lb_kim(const data_t *t, const data_t *q, int len,
		   data_t mean, data_t std, data_t bsf) {
    /// 1 point at front and back
    data_t d, lb;
    data_t x0 = (t[0] - mean) / std;
    data_t y0 = (t[(len - 1)] - mean) / std;
	
//	printf("x[0] norm = %.3f, x[M-1] norm = %.3f\n", x0,y0);
	
    lb = dist(x0, q[0]) + dist(y0, q[len - 1]);
    if (lb >= bsf) {
        return lb;
    }
	
    /// 2 points at front
	if (len < 3) return lb;
    data_t x1 = (t[1] - mean) / std;
	
    d = min(dist(x1, q[0]), dist(x0, q[1]));
    d = min(d, dist(x1, q[1]));
    lb += d;
    if (lb >= bsf) {
        return lb;
    }
	
    /// 2 points at back
	if (len < 4) return lb;
    data_t y1 = (t[(len - 2)] - mean) / std;
	
    d = min(dist(y1, q[len - 1]), dist(y0, q[len - 2]));
    d = min(d, dist(y1, q[len - 2]));
    lb += d;
    if (lb >= bsf) {
        return lb;
    }
	
    /// 3 points at front
	if (len < 5) return lb;
    data_t x2 = (t[2] - mean) / std;
	
    d = min(dist(x0, q[2]), dist(x1, q[2]));
    d = min(d, dist(x2, q[2]));
    d = min(d, dist(x2, q[1]));
    d = min(d, dist(x2, q[0]));
    lb += d;
    if (lb >= bsf) {
        return lb;
    }
    /// 3 points at back
	if (len < 6) return lb;
    data_t y2 = (t[(len - 3)] - mean) / std;
	
    d = min(dist(y0, q[len - 3]), dist(y1, q[len - 3]));
    d = min(d, dist(y2, q[len - 3]));
    d = min(d, dist(y2, q[len - 2]));
    d = min(d, dist(y2, q[len - 1]));
    lb += d;
	
    return lb;
}

/// LB_Keogh 1: Create envelope for the query
/// Note that because the query is known, envelope can be created once at the begenining.
///
/// Variable Explanation,
/// order : sorted indices for the query.
/// uo, lo: upper and lower envelops for the query, which already sorted.
/// t     : a circular array keeping the current data.
/// j     : index of the starting location in t
/// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
data_t ucr_lb_keogh_query(const idx_t* order, const data_t *t,
						  const data_t *uo, const data_t *lo,
						  data_t *cb, length_t len,
						  data_t mean, data_t std, data_t bsf) {
    data_t  lb = 0;
    data_t  x, d;
    int     i;
	
    for (i = 0; i < len && lb < bsf; i++) {
        x = (t[(order[i])] - mean) / std;
        if (x > uo[i]) {
            d = dist(x, uo[i]);
        }
        else if(x < lo[i]) {
            d = dist(x, lo[i]);
        } else {
			d = 0;
		}
        lb += d;
        cb[order[i]] = d;
    }
    return lb;
}

data_t lb_keogh_fixed_order_normalizing(const data_t* t, data_t *tz,
							const data_t * u,const data_t* l,
							data_t* cb, length_t len,
							data_t mean, data_t std, data_t bsf) {
	data_t  lb = 0;
    data_t  x,d;
    int     i;
	
//	array_print_with_name(t, len, "lb_keogh x");
//	printf("lb_keogh meanX, stdX = %.4f, %.3f\n", mean, std);
	
    for (i = 0; i < len && lb < bsf; i++) {
        x = (t[i] - mean) / std;
		
        if (x > u[i]) {
            d = dist(x, u[i]);
        }
        else if(x < l[i]) {
            d = dist(x, l[i]);
        } else {
			d = 0;
		}
        lb += d;
        cb[i] = d;
		tz[i] = x;
//		printf("x%d = %.3f, u = %.3f, l = %.3f, tz = %.3f\n",i,x,u[i],l[i], tz[i]);
    }
    return lb;
}

data_t lb_keogh_fixed_order_normalized(data_t* xz,
									   const data_t* u, const data_t* l,
									   data_t *cb, int len, data_t bsf) {
	data_t  lb = 0;
    data_t  x, d;
    int     i;
	
    for (i = 0; i < len && lb < bsf; i++) {
        x = xz[i];
        if (x > u[i]) {
            d = dist(x, u[i]);
        }
        else if(x < l[i]) {
            d = dist(x, l[i]);
        } else {
			d = 0;
		}
        lb += d;
        cb[i] = d;
    }
    return lb;
}

/// LB_Keogh 2: Create Envelop for the data
/// Note that the envelops have been created (in main function) when each data point has been read.
///
/// Variable Explanation,
/// tz: Z-normalized data
/// qo: sorted query
/// cb: (output) current bound at each position. Used later for early abandoning in DTW.
/// l, u: lower and upper envelop of the current data
data_t ucr_lb_keogh_data(const idx_t* order, const data_t *qo, data_t *cb,
						 const data_t *l, const data_t *u,
						 length_t len, data_t mean, data_t std, data_t bsf) {
    data_t  lb = 0;
    data_t  uu, ll, d;
    int     i = 0;
	
    for (i = 0; i < len && lb < bsf; i++) {
        uu = (u[order[i]] - mean) / std;
        ll = (l[order[i]] - mean) / std;
        
        if (qo[i] > uu) {
            d = dist(qo[i], uu);
        }
        else if(qo[i] < ll) {
            d = dist(qo[i], ll);
        } else {
			d = 0;
		}
        lb += d;
        cb[order[i]] = d;
    }
    return lb;
}

/// Calculate Dynamic Time Warping distance
/// A,B: data and query, respectively
/// cb : cummulative bound used for early abandoning
/// r  : size of Sakoe-Chiba warpping band
data_t dtw(const data_t* A, const data_t* B, const data_t *cb, length_t m, length_t r, data_t bsf) {
    data_t  *cost;
    data_t  *cost_prev;
    data_t  *cost_tmp;
    int     i, j, k;
    data_t  x, y, z, min_cost;
	int final_idx = m - 1;
	int cum_bound_index;
	int width = 2 * r + 1;
	int two_times_r = 2 * r;
	
    /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
    cost = (data_t*)malloc(sizeof(data_t) * width);
    for(k = 0; k < width; k++) {
        cost[k] = INFINITY;
    }
	
    cost_prev = (data_t*)malloc(sizeof(data_t) * width);
    for(k = 0; k < width; k++) {
        cost_prev[k] = INFINITY;
    }
	
    for (i = 0; i < m; i++) {	//for each column
        k = max(0, r - i);	//first entry in costs array
        min_cost = INFINITY;
		
		int firstRowInWindow = max(0, i - r);
		int lastRowInWindow = min(final_idx, i + r);
        for(j = firstRowInWindow; j <= lastRowInWindow; j++, k++) {  //for each row in warping window
            /// Initialize first entry
            if ((i == 0) && (j == 0)) {
                cost[k] = dist(A[0], B[0]);
                min_cost = cost[k];
                continue;
            }
			
			short int inFirstRow		= j - 1 < 0;
			short int inFirstCol		= i - 1 < 0;
			short int inFirstWarpRow	= k - 1 < 0;
			short int inNewWarpRow		= k + 1 > two_times_r;
			
			if ( inFirstRow || inFirstWarpRow ) y = INFINITY;
            else                                y = cost[k - 1];		//vertical
            if ( inFirstCol || inNewWarpRow )	x = INFINITY;
            else                                x = cost_prev[k + 1];	//horizontal
            if ( inFirstCol || inFirstRow )     z = INFINITY;
            else                                z = cost_prev[k];		//diagonal
			
            /// Classic DTW calculation
            cost[k] = min(min(x, y), z) + dist(A[i], B[j]);
			
            /// Find minimum cost in row for early abandoning (possibly to use column instead of row).
            if (cost[k] < min_cost) {
                min_cost = cost[k];
            }
        }
		
        // We can abandon early if the current cummulative distace with lower
		// bound together are larger than bsf
		cum_bound_index = i + r + 1;
		if (cum_bound_index < m ) {
			data_t min_total_cost = min_cost + cb[cum_bound_index];
			if (min_total_cost >= bsf) {
				free(cost);
				free(cost_prev);
				return min_total_cost;
			}
		}
		
        /// Move current array to previous array.
        cost_tmp = cost;
        cost = cost_prev;
        cost_prev = cost_tmp;
    }
    k--;
	
    // the DTW distance is in the last cell in the matrix of size O(m^2),
	// which is in the middle of our warping window array.
    data_t final_dist = cost_prev[k];
    free(cost);
    free(cost_prev);
    return final_dist;
}

//NOTE that neither this nor the normal dtw_distance will actually
//degenerate to the Euclidean distance in *all* cases since lb_kimFL
//assumes a warping window of r >= 2
data_t dtw_distance_from_scratch(data_t* q, const data_t* x,
								 length_t r, length_t m,
								 data_t meanQ, data_t stdQ,
								 data_t meanX, data_t stdX,
								 data_t bsf) {
	idx_t i,k;
	ensureTempArraysLongEnough(m);
	
	// normalize first and last 3 points for lb_kim, making sure that
	// we handle short queries properly
	q[0]	= (q[0]   - meanQ) / stdQ;
	q[1]	= (q[1]   - meanQ) / stdQ;
	if (m > 2) {
		q[2]	= (q[2]   - meanQ) / stdQ;
		if (m > 3) {
			q[m-1]	= (q[m-1] - meanQ) / stdQ;
			if (m > 4) {
				q[m-2]	= (q[m-2] - meanQ) / stdQ;
				if (m > 5) {
					q[m-3]	= (q[m-3] - meanQ) / stdQ;
				}
			}
		}
	}
	
	if (r >= 2) {
		data_t lb_kim = ucr_lb_kim(x, q, m, meanX, stdX, bsf);
		if (lb_kim >= bsf || m <= 6)
			return lb_kim;
	}
	
	//z-normalize the query so we can build an envelope around it; note that
	//the call to lb_kim_normalizing already took care of the first and last
	//3 points
	for (i=3; i < m-3; i++) {
		q[i] = (q[i] - meanQ) / stdQ;
	}
	
	//build the envelope around the data and run early-abandoning lb_keogh;
	//note that we can't use the normal ucr_lb_keogh since we don't have the
	//query values sorted, since this would take nlg(n) and our whole objective
	//is to be O(n) here
	build_envelope(q, m, r, lTmp, uTmp);
	data_t lb_k = lb_keogh_fixed_order_normalizing(x,datazTmp,uTmp,lTmp,cb1Tmp,
												   m,meanX,stdX,bsf);
	if (lb_k >= bsf)
		return lb_k;

	build_envelope(datazTmp, m, r, lTmp, uTmp);	//recycle l and u to wrap data now
	data_t lb_k2 = lb_keogh_fixed_order_normalized(q,uTmp,lTmp,cb2Tmp,m,bsf);
	if (lb_k2 >= bsf)
		return lb_k2;
	
	// Choose better lower bound between lb_keogh around query
	// and lb_keogh around data to be used in early abandoning DTW;
	// cb is the cumulative lower bound from _end_ to _start_
	int final_idx = m-1;
	if (lb_k2 < lb_k) {
		cb1Tmp[final_idx] = cb2Tmp[final_idx];
		for(k = final_idx - 1; k >= 0; k--)
			cb1Tmp[k] = cb1Tmp[k + 1] + cb2Tmp[k];
	} else {
		for(k = final_idx - 1; k >= 0; k--)
			cb1Tmp[k] = cb1Tmp[k + 1] + cb1Tmp[k];
	}
	
	// compute DTW distance, abandoning early if possible
	return dtw(datazTmp, q, cb1Tmp, m, r, bsf);
}

data_t dtw_distance(const dtw_query* query,
					const data_t* t,
					const data_t* l_buff, const data_t* u_buff,
					data_t mean, data_t std,
					data_t bsf) {
	//unpack the query struct
	data_t* q		= query->q;
    length_t m		= query->m;
    length_t r		= query->r;
    idx_t* order	= query->order;
    data_t* qo		= query->qo;
    data_t* lo		= query->lo;
    data_t* uo		= query->uo;
	
	ensureTempArraysLongEnough(m);
	
	/// Use a O(1) lower bound to prune the obvious subsequences
	if (r >= 2) {
		data_t lb_kim = ucr_lb_kim(t, q, m, mean, std, bsf);
		if (lb_kim >= bsf || m <= 6)
			return lb_kim;
	}
	
	/// Use a linear time lower bound to prune; z_normalization
	/// of t will be computed on the fly.
	/// uo, lo are the sorted envelop of the query.
	data_t lb_k = ucr_lb_keogh_query(order, t, uo, lo, cb1Tmp, m, mean, std, bsf);
	if (lb_k >= bsf)
		return lb_k;
	
	/// Take another linear time to compute z_normalization of t.
	/// Note that for better optimization, this can merge with the previous function.
	idx_t k;
	for(k = 0; k < m; k++) {
		datazTmp[k] = (t[k] - mean) / std;
	}
	
	/// Use another lb_keogh (this time around the data) to prune
	/// qo is the sorted query. tz is unsorted z_normalized data.
	/// l_buff, u_buff are big envelopes for all data in this chunk
	data_t lb_k2 = ucr_lb_keogh_data(order, qo, cb2Tmp, l_buff, u_buff,
												m, mean, std, bsf);
	if (lb_k2 >= bsf)
		return lb_k2;
	
	// Choose better lower bound between lb_keogh around query
	// and lb_keogh around data to be used in early abandoning DTW;
	// cb is the cumulative lower bound from _end_ to _start_
	idx_t final_idx = m-1;
	if (lb_k2 < lb_k) {
		cb1Tmp[final_idx] = cb2Tmp[final_idx];
		for(k = final_idx - 1; k >= 0; k--)
			cb1Tmp[k] = cb1Tmp[k + 1] + cb2Tmp[k];
	} else {
		for(k = final_idx - 1; k >= 0; k--)
			cb1Tmp[k] = cb1Tmp[k + 1] + cb1Tmp[k];
	}
	
	// compute DTW distance, abandoning early if possible
	return dtw(datazTmp, q, cb1Tmp, m, r, bsf);
}

//================================================================
// DTW SEARCH FUNCTIONS
//================================================================

int32_t dtw_search_execute(const dtw_query *query, const dtw_buffer *buffer, Index *result) {
	return dtw_ongoing_search_execute(query, buffer, INFINITY, result);
}

int32_t dtw_ongoing_search_execute(const dtw_query *query, const dtw_buffer *buffer,
								   data_t bsf, Index *result) {
    data_t		d;
    data_t		sumx, sumx2, mean, std, distance;
    data_t		*t=NULL;//, *tz=NULL;
	data_t		*u_buff=NULL, *l_buff=NULL;
    int32_t		i = 0, j = 0, loc = -1;
    int32_t		I, r, m;
	
    if(buffer->last < 0)
        goto query_execute_cleanup;
	
    m = query->m;
    r = query->r;
	
    t = (data_t *)malloc(sizeof(data_t) * m * 2);
    if(t == NULL)
        goto query_execute_cleanup;

    u_buff = (data_t *)malloc(sizeof(data_t) * (buffer->len));
    if(u_buff == NULL)
        goto query_execute_cleanup;
	
    l_buff = (data_t *)malloc(sizeof(data_t) * (buffer->len));
    if(l_buff == NULL)
        goto query_execute_cleanup;
	
	// precompute warping envelope for whole buffer
    build_envelope(buffer->data, buffer->len, r, l_buff, u_buff);
	
    /// Do main task here
    sumx = sumx2 = 0;
    for(i = 0; i < buffer->len; i++) {	//for each datapoint in the buffer
        /// A bunch of data has been read and pick one of them at a time to use
        d = buffer->data[i];
		
        // update sum and sum squared
        sumx += d;
        sumx2 += d * d;
		
        /// t is a circular array for keeping current data
        t[i % m] = d;
		
        /// data_t the size for avoiding using modulo "%" operator
        t[(i % m) + m] = d;
		
        /// Start the task when there are more than m-1 points in the current chunk
        if( i >= m - 1 ) {
            mean = sumx / m;
            std = sumx2 / m;
            std = sqrt(std - mean * mean); // sqrt( E[X^2] - E[X]^2 )
			
            /// compute the start location of the data in the current circular array, t
            j = (i + 1) % m;
            /// the start location of the data in the current chunk
            I = i - (m - 1);
			distance = dtw_distance(query, t+j, l_buff+I, u_buff+I,
									mean, std, bsf);
			
            if( distance < bsf ) {   /// Update best-so-far distance
				/// loc is the real starting location of the nearest neighbor in the file
                bsf = distance;
                loc = i - m + 1;
            }
			
            /// Remove points from sum and sum squared that passed out of the
			/// sliding window
            sumx -= t[j];
            sumx2 -= t[j] * t[j];
        }
    }

query_execute_cleanup:
    if(t != NULL)
        free(t);
    if(l_buff != NULL)
        free(l_buff);
    if(u_buff != NULL)
        free(u_buff);

    if(loc == -1) {
        return loc;
    }
	
    result->index = loc;
    result->value = bsf;
	
    return kSUCCESS;
}

int32_t dtw_search(data_t *query, data_t *buffer, length_t m, length_t n, data_t r, Index *result) {
	return dtw_ongoing_search(query, buffer, m, n, r, INFINITY, result);
}

int32_t dtw_ongoing_search(data_t *query, data_t *buffer, length_t m, length_t n, data_t r,
						   data_t bsf, Index *result) {
	//if warping window narrow enough, degenerates to euclidean distance
	if ( round(r*m) == 0) {
		return euc_ongoing_search(buffer, query, m, n, bsf, result);
	}
	
	//check args valid
	int validity = validateArgs(query, buffer, m, n, bsf, result);
	if ( validity != kSUCCESS) {
		invalidate(result);
		return validity;
	}
	if ( r < 0) {
		printf("Error: could not compute distance; warping width %.3f < 0\n", r);
		invalidate(result);
		return kFAILURE;
	}
	
	dtw_buffer    *b = NULL;
    dtw_query     *q = NULL;
    int32_t              e;
	
    q = dtw_query_new(query, m, r);
    if(q == NULL)
        return kFAILURE;
	
    b = (dtw_buffer *)malloc(sizeof(dtw_buffer));
    if(b == NULL)
        return kFAILURE;
	
    b->data = buffer;
    b->len = n;
    b->last = n - 1;
	
    e = dtw_ongoing_search_execute(q, b, bsf, result);
    dtw_query_free(q);
	
	free(b);
	
    return e;
}

//================================================================
// Scaled Warped Matching (SWM) functions
//================================================================

swm_query* swm_query_new(data_t* query, length_t m, length_t minLen, length_t maxLen, float r) {
	int i=0, o=0, len;
	swm_query* sq = NULL;
	int numLens = maxLen - minLen + 1;
	
	data_t l[minLen];
	data_t u[minLen];
	data_t s[maxLen];
	data_t envWidths[minLen];
	
	sq = (swm_query*)malloc(sizeof(swm_query));
	if (sq == NULL) {
		goto swm_query_new_cleanup;
	}
	sq->q = (data_t*)malloc(m*sizeof(data_t));
	if (sq->q == NULL)
		goto swm_query_new_cleanup;
	sq->envOrder = (idx_t*)malloc(minLen*sizeof(idx_t));
	if (sq->envOrder == NULL)
		goto swm_query_new_cleanup;
	sq->lo = (data_t*)malloc(minLen*sizeof(data_t));
	if (sq->lo == NULL)
		goto swm_query_new_cleanup;
	sq->uo = (data_t*)malloc(minLen*sizeof(data_t));
	if (sq->uo == NULL)
		goto swm_query_new_cleanup;
	sq->dtwQueries = (dtw_query**)malloc(numLens*sizeof(dtw_query*));
	if (sq->dtwQueries == NULL)
		goto swm_query_new_cleanup;
	
	//store the query data in the query struct
	for(int i = 0; i < m; i++) {
        sq->q[i] = query[i];
    }
	znormalize(sq->q, m);
	
	// store the length + scaling + warping values
	sq->r = r;
	sq->m = m;
	sq->minLen = minLen;
	sq->maxLen = maxLen;

	// create the uniform scaling envelope for the query
	build_swm_envelope(sq->q, m,minLen,maxLen, r, l,u);

	// sort the envelope based on increasing width (so skinny sections get
	// compared first and fat sections get compared last; see UCR's "Atomic
	// Wedgie" (time series wedges) paper
	for(i = 0; i < minLen; i++) {
		envWidths[i] = u[i] - l[i];
	}
	sort_abs_increasing(envWidths, sq->envOrder, minLen);
	for(i = 0; i < minLen; i++) {
		o = sq->envOrder[i];
		sq->lo[i] = l[o];
        sq->uo[i] = u[o];
	}
	
	// create dtw queries for every possible scaling, since these
	// precompute envelopes and optimal orders that greatly improve
	// the speed of the computation
	i = 0;
	for (len = minLen; len <= maxLen; len++, i++) {
		array_resample(query, s, m, len);
		znormalize(s, len);
		if (len == m && !array_equal(sq->q, s, len)) {
			printf("ERROR: swm_query_new: resampling to same length ruined things!\n");
			exit(1);
		}
		sq->dtwQueries[i] = dtw_query_new(s, len, r);
	}
	
	return sq;
	
swm_query_new_cleanup:
	printf("cleaning up failed swm query construction\n");
	swm_query_free(sq);
	exit(1);
}

void swm_query_free(swm_query* query) {
	if (query == NULL) return;
	if (query->q != NULL)
		free(query->q);
	if (query->envOrder != NULL)
		free(query->envOrder);
	if (query->uo != NULL)
		free(query->uo);
	if (query->lo != NULL)
		free(query->lo);
	
	//free the dtw queries associated with each possible scaling
	if (query->dtwQueries != NULL) {
		int numLens = query->maxLen - query->minLen + 1;
		for (int i = 0; i < numLens; i++) {
			dtw_query_free(query->dtwQueries[i]);
		}
		free(query->dtwQueries);
	}
	free(query);
}

//TODO refactor duplicate code to avoid redundancy with test_all_scalings()
data_t swm_test_all_scalings(const data_t* query, const data_t* data,
							 dtw_query**const dtwQueries,
							 data_t r, length_t m, length_t minLen, length_t maxLen,
							 data_t meanX, data_t stdX,
							 data_t bsf, length_t* bestLen) {
	length_t len, i,j;
	idx_t scaledIdx;
	data_t newXinWindow, delta;
	data_t dist, y;
	data_t scaleFactor;
	data_t mf = (data_t) m;
	const data_t* dataStart = data + maxLen - minLen;
	data_t err = (stdX * stdX) * minLen;		// sigma^2 = err / n;
	data_t sumY, sumY2;
	data_t meanY, stdY;
	length_t rSteps;

	ensureTempArraysLongEnough(maxLen);
	
	// initially assume that the best scaling is the minimum-length scaling
	// and alter this assumption as better scalings are found
	*bestLen = minLen;
	
	// compute distance to minimum scaling; this is separate from the
	// main loop since we don't have to update the mean and standard
	// deviation
	scaleFactor = mf / minLen;
	sumY = 0;
	sumY2 = 0;
	for (j = 0; j < minLen; j++) {
		scaledIdx = j * scaleFactor;
		y = query[scaledIdx];
		yTmp[j] = y;
		sumY += y;
		sumY2 += y*y;
	}
	meanY = sumY / minLen;
	stdY = sqrt(sumY2 / minLen - meanY * meanY);
	if (stdY<=0) stdY = DBL_MIN;

	rSteps = round(r * minLen);
	dist = dtw_distance_from_scratch(yTmp, dataStart,
									 rSteps, minLen,
									 meanY, stdY, meanX, stdX,
									 bsf);
	if (dist < bsf) {
		bsf = dist;
	}

	// calculate the distance to every possible scaling of the query,
	// abandoning early whenever possible.
	i=1; // not 0 since we already tried the minimum scaling
	for (len = minLen+1; len <= maxLen; len++) {
		
		// get new data val that gets added at this scaling
		dataStart--;
		newXinWindow = *dataStart;
		
		// update mean and std of data using Knuth's online algorithm
		delta	= newXinWindow - meanX;
		meanX	= meanX + delta / len;
		err		= err + delta * (newXinWindow - meanX);
		stdX	= sqrt(err / len);
		
		//compute early-abandoning euclidean distance for this scaling
		sumY = 0;
		sumY2 = 0;
		scaleFactor = mf / len;
		for (j = 0; j < len; j++) {
			scaledIdx = j * scaleFactor;
			y = query[scaledIdx];
			yTmp[j] = y;
			sumY += y;
			sumY2 += y*y;
		}
		meanY = sumY / len;
		stdY = sqrt(sumY2 / len - meanY * meanY);
		if (stdY<=0) stdY = DBL_MIN;
		rSteps = round(r * len);
		dist = dtw_distance_from_scratch(yTmp, dataStart,
										 rSteps, len, meanY, stdY, meanX, stdX, bsf);
		if (dist < bsf) {
			bsf = dist;
			*bestLen = len;
		}
		
		i++;
	}
	return bsf;
}

data_t swm_test_all_scalings2(const data_t* query, const data_t* data,
							  dtw_query**const dtwQueries,
							  const data_t* lBuff, const data_t* uBuff,
							  data_t r, length_t m, length_t minLen, length_t maxLen,
							  data_t meanX, data_t stdX,
							  data_t bsf, length_t* bestLen) {
	length_t len, i;
	data_t newXinWindow, delta;
	data_t dist;
	dtw_query* dtwQuery = dtwQueries[0];
	
	data_t err = (stdX * stdX) * minLen; // variance *n, for running stdX
	
	idx_t startOffset = maxLen - minLen;
	const data_t* dataStart = data + startOffset;
	const data_t* lBuffStart = lBuff + startOffset;
	const data_t* uBuffStart = uBuff + startOffset;

	ensureTempArraysLongEnough(maxLen);
	
	// initially assume that the best scaling is the minimum-length scaling
	// and alter this assumption as better scalings are found
	*bestLen = minLen;
	
	// compute distance to minimum scaling; this is separate from the
	// main loop since we don't have to update the mean and standard
	// deviation
	dist = dtw_distance(dtwQuery, dataStart, lBuffStart, uBuffStart, meanX, stdX, bsf);
	if (dist < bsf) {
		bsf = dist;
	}

	// calculate the distance to every possible scaling of the query,
	// abandoning early whenever possible.
	i=1; // not 0 since we already tried the minimum scaling
	for (len = minLen+1; len <= maxLen; len++) {
		
		// get new data val that gets added at this scaling
		dataStart--;
		lBuffStart--;
		uBuffStart--;
		newXinWindow = *dataStart;
		
		// update mean and std of data using Knuth's online algorithm
		delta	= newXinWindow - meanX;
		meanX	= meanX + delta / len;
		err		= err + delta * (newXinWindow - meanX);
		stdX	= sqrt(err / len);
		
		dtwQuery = dtwQueries[i];
		dist = dtw_distance(dtwQuery, dataStart, lBuffStart, uBuffStart, meanX, stdX, bsf);
		
		if (dist < bsf) {
			bsf = dist;
			*bestLen = len;
		}
		i++;
	}
	return bsf;
}

data_t swm_distance(const swm_query *query, const data_t *data,
					const data_t* l_buff, const data_t* u_buff,
					data_t meanX, data_t stdX, data_t bsf) {
	short int useEnvelope = 0;
	return swm_distance_envelope(query, data, l_buff, u_buff, meanX, stdX, bsf, useEnvelope);
}

data_t swm_distance_and_len(const swm_query *query, const data_t *data,
							const data_t* l_buff, const data_t* u_buff,
							data_t meanX, data_t stdX, data_t bsf,
							length_t* bestLen) {
	short int useEnvelope = 0;
	return swm_distance_and_len_envelope(query, data, l_buff, u_buff, meanX, stdX, bsf, useEnvelope, bestLen);
}

data_t swm_distance_envelope(const swm_query *query, const data_t *data,
							 const data_t* l_buff, const data_t* u_buff,
							 data_t meanX, data_t stdX,
							 data_t bsf, short int useEnvelope) {
	length_t x; // dummy var, just to get a valid int*
	return swm_distance_and_len_envelope(query, data, l_buff, u_buff, meanX, stdX, bsf, useEnvelope, &x);
}

data_t swm_distance_and_len_envelope(const swm_query *query, const data_t *data,
									 const data_t* l_buff, const data_t* u_buff,
									 data_t meanX, data_t stdX,
									 data_t bsf, short int useEnvelope,
									 length_t* bestLen) {
	length_t maxLen = query->maxLen;
	length_t minLen = query->minLen;

	if (useEnvelope) {
		const data_t* dataCompareStart = data + maxLen - minLen;
		data_t lb_k = lb_keogh_us_query(query->envOrder, dataCompareStart,
										query->uo, query->lo, minLen,
										meanX, stdX, bsf);
		if (lb_k >= bsf) {
			return lb_k;
		}
	}
	
	return swm_test_all_scalings2(query->q, data, query->dtwQueries,
								  l_buff, u_buff,
								  query->r,query->m,query->minLen,query->maxLen,
								  meanX, stdX, bsf, bestLen);
}
