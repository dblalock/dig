//
//  distance_utils.cpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include "distance_utils.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "type_defs.h"

/** Z-normalize the first stopAfter elements of an array ar with length m */
void znormalize_prefix(data_t *ar, length_t m, length_t stopAfter) {
	data_t sumx = 0, sumx2 = 0;
	data_t x, mean, std;
	int i;

	//compute mean and standard deviation
	for (i = 0; i < m; i++) {
		x = ar[i];
		sumx += x;
		sumx2 += x*x;
	}
	mean = sumx / m;
	std = sqrt(sumx2 / m - mean*mean);
	if (std == 0) {
		std = DBL_MIN; // don't divide by 0
	}

	//z-normalize query
	for (i = 0; i < stopAfter; i++) {
		ar[i] = (ar[i] - mean) / std;
	}
}

/** Z-normalize the given array of length m in place */
void znormalize(data_t* ar, length_t m) {
	znormalize_prefix(ar, m, m);
}

/* Comparison function for sorting the query.
 * The query will be sorted by absolute z-normalization value,
 * |z_norm(Q[i])| from high to low. */
int index_comp(const void *a, const void* b) {
	Index* x = (Index*)a;
	Index* y = (Index*)b;
	return fabs(y->value) - fabs(x->value);
}

/* Comparison function for sorting the query.
 * The query will be sorted by absolute z-normalization value,
 * |z_norm(Q[i])| from low to high. */
int index_comp_increasing(const void *a, const void* b) {
	Index* x = (Index*)a;
	Index* y = (Index*)b;
	return fabs(x->value) - fabs(y->value);
}

/** Reorders the elements in a z-normalized array in descending order of
 *  absolute value and writes the ordering of indices from the original array. */
int32_t sort_abs_decreasing(data_t* normalizedAr, idx_t* order, length_t m) {

	//allocate a temporary array for sorting
	Index *Q_tmp = (Index *)malloc(sizeof(Index)*m);
	if (Q_tmp == NULL) {
		printf("Error: couldn't allocate space to sort array\n");
		return kFAILURE;
	}

	// bundle array values into an Index struct for sorting
	int i;
	for( i = 0 ; i < m ; i++ ) {
		Q_tmp[i].value = normalizedAr[i];
		Q_tmp[i].index = i;
	}

	//sort the Index structs
	qsort(Q_tmp, m, sizeof(Index), index_comp);

	//reorder the original array and record the order of elements
	for( i = 0; i < m; i++) {
		normalizedAr[i] = Q_tmp[i].value;
		order[i] = (int)Q_tmp[i].index; //safe to cast since query len << 2 bil
	}

	free(Q_tmp);
	return kSUCCESS;
}

int32_t sort_abs_increasing(data_t* normalizedAr, idx_t* order, length_t m) {

	//allocate a temporary array for sorting
	Index *Q_tmp = (Index *)malloc(sizeof(Index)*m);
	if (Q_tmp == NULL) {
		printf("Error: couldn't allocate space to sort array\n");
		return kFAILURE;
	}

	// bundle array values into an Index struct for sorting
	int i;
	for( i = 0 ; i < m ; i++ ) {
		Q_tmp[i].value = normalizedAr[i];
		Q_tmp[i].index = i;
	}

	//sort the Index structs
	qsort(Q_tmp, m, sizeof(Index), index_comp_increasing);

	//reorder the original array and record the order of elements
	for( i = 0; i < m; i++) {
		normalizedAr[i] = Q_tmp[i].value;
		order[i] = (int)Q_tmp[i].index; //safe to cast since query len << 2 bil
	}

	free(Q_tmp);
	return kSUCCESS;
}
