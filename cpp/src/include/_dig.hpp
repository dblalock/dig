//
//  dig.h
//  Dig
//
//  Created by DB on 7/28/14.
//  Copyright (c) 2014 D Blalock. All rights reserved.
//

#ifndef __Dig__dig__
#define __Dig__dig__

#include <vector>

#include "dig.h"

namespace tk {

//==================================================
// Data structures
//==================================================

template <typename data_t>
struct array_1D {
	data_t* startIdx;
	size_t length;
};

//==================================================
// Distance Measures
//==================================================

//-------------------------------
// Raw Arrays
//-------------------------------

//just call above with whichever of ED, DTW, US, SWM is necessary
//to achieve the params
template <typename data_t, typename dist_t>
inline dist_t distance(data_t* array1, data_t* array2,
				size_t len1, size_t len2,
				DistanceMeasureParams params);

//-------------------------------
// Vectors
//-------------------------------

//just call above with whichever of ED, DTW, US, SWM is necessary
//to achieve the params
template <typename data_t, typename dist_t>
inline dist_t distance(std::vector<data_t> array1, std::vector<data_t> array2,
				DistanceMeasureParams params) {
	return distance(&array1[0], &array2[0], array1.size(), array2.size(), params);
}

//==================================================
// Subsequence Search
//==================================================

//-------------------------------
// Raw Arrays
//-------------------------------

//TODO maybe add initial cutoff as param for all these so you can chain them
//together more easily
	//slash, just look at similaritySearch.cpp and see what it has

template <typename data_t>
array_1D<data_t> nearest_neighbor_subseq(data_t* query, data_t* buffer,
								size_t queryLen, size_t buffLen,
								DistanceMeasureParams p);

template <typename data_t>
std::vector<array_1D<data_t>> k_nearest_neighbor_subseqs(data_t* query, data_t* buffer,
								   size_t queryLen, size_t buffLen,
								   DistanceMeasureParams p,
								   unsigned int k, float maxOverlap=0);

template <typename data_t, typename dist_t>
std::vector<array_1D<data_t>> similar_subseqs(data_t* query, data_t* buffer,
								   size_t queryLen, size_t buffLen,
								   DistanceMeasureParams p,
								   dist_t threshold, float maxOverlap=0);

//use EDW/DDW for this one
template <typename data_t>
std::pair<array_1D<data_t>, array_1D<data_t>>
most_similar_subseq_fast(data_t* array1, data_t* array2,
						 size_t len1, size_t len2,
						 unsigned int minLen, unsigned int maxLen,
						 bool euclidean=false);

//-------------------------------
// Vectors
//-------------------------------

template <typename data_t>
array_1D<data_t> nearest_neighbor_subseq(std::vector<data_t> query, std::vector<data_t> buffer,
							   DistanceMeasureParams p);

template <typename data_t>
std::vector<array_1D<data_t>>
k_nearest_neighbor_subseqs(std::vector<data_t> query, std::vector<data_t> buffer,
						   DistanceMeasureParams p,
						   unsigned int k, float maxOverlap=0);

template <typename data_t, typename dist_t>
std::vector<array_1D<data_t>>
similar_subseqs(std::vector<data_t> query, std::vector<data_t> buffer,
				DistanceMeasureParams p,
				dist_t threshold, float maxOverlap=0);

template <typename data_t>
std::pair<array_1D<data_t>, array_1D<data_t>>
most_similar_subseq_fast(std::vector<data_t> array1, std::vector<data_t> array2,
						 unsigned int minLen, unsigned int maxLen,
						 bool euclidean=false);

//==================================================
// Database Search
//==================================================



//==================================================
// Discretization
//==================================================



//==================================================
// Anomaly Detection
//==================================================

template <typename data_t>
size_t leastSimilarSubseq(data_t* buffer, size_t buffLen,
								size_t minLen, size_t maxLen,
								DistanceMeasureParams p);



} // namespace dig
#endif /* defined(__Dig__dig__) */
