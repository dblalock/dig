//
//  ucr_funcs.h
//  edtw
//
//  Created By <Anonymous> on 1/4/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#ifndef timekit_ucr_funcs_h
#define timekit_ucr_funcs_h

#include <sys/types.h>
#include "type_defs.h"

#include "global_test_vals.h"

#ifdef __cplusplus
extern "C" {
#endif
	
//typedef int32_t length_t; // length of a time series in main memory; needs to be signed
//typedef int16_t idx_t;
//typedef int32_t tick_t;
//typedef double data_t;
	
//extern const int kINVALID;
//extern const int kSUCCESS;
//extern const int kFAILURE;
//
//extern const float kFLOATING_PT_ERROR;
	
//typedef struct Index {
//	data_t value;
//	tick_t index;
// } Index;
	
typedef struct us_query {
	length_t m, minLen, maxLen;
	idx_t *envOrder;
	idx_t **orders;
	data_t *q, *uo, *lo;
	data_t *means, *stds;
} us_query;

typedef struct dtw_buffer {
    tick_t     len;
    tick_t     last;
    data_t      *data;
} dtw_buffer;

typedef struct dtw_query {
    length_t         m, r;
    idx_t			*order;
    data_t          *q, *qo, *uo, *lo;
} dtw_query;

typedef struct swm_query {
	data_t *q, *uo, *lo;
	idx_t *envOrder;
	dtw_query** dtwQueries;
	data_t r;
	length_t m, minLen, maxLen;
} swm_query;

//================================================================
// Utility functions
//================================================================

void invalidate(Index *result);
short int isInvalid(Index result);

//================================================================
// Euclidean Distance (ED) functions
//================================================================

int32_t euc_search(data_t* query, data_t* buffer, length_t m, length_t n, Index* result);
int32_t euc_ongoing_search(data_t* query, const data_t* buffer, length_t m, length_t n,
						   data_t bsf, Index* result);

data_t euclidean_dist_sq(const data_t* query, const data_t* buffer, length_t m, data_t mean,
						 data_t std, const idx_t* order, data_t bsf);

//================================================================
// Uniform Scaling (US) functions
//================================================================

void build_unifscale_envelope(const data_t *x, length_t origLen,
							  length_t minLen, length_t maxLen,
							  data_t *l, data_t *u);

us_query *us_query_new(data_t* query, length_t m, length_t minLen, length_t maxLen);
void us_query_free(us_query* query);

data_t us_distance(const us_query *query, const data_t *buffer,
				   data_t mean, data_t std, data_t bsf);
data_t us_distance_and_len(const us_query *query, const data_t *buffer,
						   data_t mean, data_t std, data_t bsf, length_t* bestLen);
data_t us_distance_envelope(const us_query *query, const data_t *buffer,
				   data_t mean, data_t std, data_t bsf, short int useEnvelope);
data_t us_distance_and_len_envelope(const us_query *query, const data_t *buffer,
						   data_t mean, data_t std, data_t bsf,
						   short int useEnvelope, length_t* bestLen);

//================================================================
// Dynamic Time Warping (DTW) functions
//================================================================

void build_envelope(const data_t *t, tick_t len, length_t r, data_t *l, data_t *u);

dtw_query* dtw_query_new(data_t* query, length_t m, float r);
void dtw_query_free(dtw_query* query);

int32_t dtw_search_execute(const dtw_query *query, const dtw_buffer *buffer, Index *result);
int32_t dtw_ongoing_search_execute(const dtw_query *query, const dtw_buffer *buffer,
								   data_t bsf, Index *result);

int32_t dtw_search(data_t *query, data_t *buffer, length_t m, length_t n, data_t r, Index *result);
int32_t dtw_ongoing_search(data_t *query, data_t *buffer, length_t m, length_t n,
						   data_t r, data_t bsf, Index *result);

data_t dtw(const data_t* A, const data_t* B, const data_t *cb, length_t m, length_t r, data_t bsf);

data_t dtw_distance(const dtw_query* query,
					const data_t* t,
					const data_t* l_buff, const data_t* u_buff,
					data_t mean, data_t std,
					data_t bsf);

data_t dtw_distance_from_scratch(data_t* q, const data_t* x,
								 length_t r, length_t m,
								 data_t meanQ, data_t stdQ,
								 data_t meanX, data_t stdX,
								 data_t bsf);
	
//================================================================
// Scaled Warped Matching (SWM) functions
//================================================================

swm_query* swm_query_new(data_t* query, length_t m, length_t minLen, length_t maxLen, float r);
void swm_query_free(swm_query* query);

data_t swm_distance(const swm_query *query, const data_t *data,
					const data_t* l_buff, const data_t* u_buff,
					data_t meanX, data_t stdX, data_t bsf);
data_t swm_distance_and_len(const swm_query *query, const data_t *data,
							const data_t* l_buff, const data_t* u_buff,
							data_t meanX, data_t stdX, data_t bsf,
							length_t* bestLen);
data_t swm_distance_envelope(const swm_query *query, const data_t *data,
							 const data_t* l_buff, const data_t* u_buff,
							 data_t meanX, data_t stdX,
							 data_t bsf, short int useEnvelope);
data_t swm_distance_and_len_envelope(const swm_query *query, const data_t *data,
							const data_t* l_buff, const data_t* u_buff,
							data_t meanX, data_t stdX, data_t bsf,
							short int useEnvelope, length_t* bestLen);


#ifdef __cplusplus
}
#endif
	
#endif
