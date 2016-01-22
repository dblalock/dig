//
//  dtw.hpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TimeKit__dtw__
#define __TimeKit__dtw__

#include <stdlib.h>
#include <math.h>
#include <array>
#include <vector>
#include <valarray>

#include "lower_bounds.hpp"
#include "Lp.hpp"
#include "debug_utils.hpp"
#include "math_utils.h"
#include "distance_utils.hpp"

// Crap we need in here:
//	-a DTW Example object that can be used for insanely fast comparisons
//	-something to make LB_Keogh envelopes
//	-DTW with / without early abandoning threshold
//		-UCR suite for the latter (in implemenation file)
//			-we'll need stuff to handle non-floats differently
//
// Once this is all working well, have everything use it and stop doing
// anything with ucr_funcs.c/h

template <class data_t, class len_t, class dist_t>
class DTWTempStorage {
public:
	std::vector<dist_t> cb1;
	std::vector<dist_t> cb2;

	DTWTempStorage(len_t len):
		cb1(len),
		cb2(len)
	{}

	DTWTempStorage(const DTWTempStorage& other) = delete;

//	std::vector<dist_t>& getCumBound1() const { return cb1; }
};

template <class data_t, class len_t, class label_t=int>
class DTWExample {
private:
	std::vector<data_t> _data;
	std::vector<len_t>  _order;
	std::vector<data_t> _LBKeogh_u;
	std::vector<data_t> _LBKeogh_l;
	len_t _len;
	len_t _warp;
	label_t _label;
public:
	DTWExample(const data_t* x, len_t len, float warp, label_t label=0, bool znormalize=true):
		_data(x,x+len),
		_order(len),
		_LBKeogh_u(len),
		_LBKeogh_l(len),
		_len(len),
		_label(label)
	{
		assertf(len > 0, "DtwExample: cannot create instance with length %d", len);

		// store warping width as integer (convert if fraction of m)
	    if (warp < 1) {
			_warp = round(warp * len);
	    } else {
	        _warp = floor(warp);
	        assertf(_warp < len, "Warping width %d must be less than array length %d",
	        	_warp, len);
	        assertf(_warp >= 0, "Warping width %d nonnegative", _warp);
		}

		// znormalize data
		if (znormalize) {
			z_normalize(&_data[0], len);
		}

		// precompute LB_Keogh envelope
		build_envelope(&_data[0], len, _warp, &_LBKeogh_l[0], &_LBKeogh_u[0]);

		// compute best sorted order for envelope; we want to compute
		// distances for indices where the envelope is less likely to
		// enclose the point to which we're comparing
		std::vector<double> probabilities(len);
		for (len_t i = 0; i < len; i++) {
			probabilities[i] = normalCdf(_LBKeogh_u[i]) - normalCdf(_LBKeogh_l[i]);
		}
		bool increasing = true;
		_order = sorted_indices(probabilities,len,increasing);

		// resample the LB_Keogh envelopes in this order
		auto uTmp = _LBKeogh_u;
		auto lTmp = _LBKeogh_l;
		for(len_t i = 0; i < len; i++) {
			_LBKeogh_u[i] = uTmp[_order[i]];
			_LBKeogh_l[i] = lTmp[_order[i]];
		}
	}

	//getters
	const std::vector<data_t>& 	getData() 			const { return _data; }
	const std::vector<len_t>&  	getOrder() 			const { return _order; }
	const std::vector<data_t>& 	getUpperEnvelope() 	const { return _LBKeogh_u; }
	const std::vector<data_t>& 	getLowerEnvelope() 	const { return _LBKeogh_l; }
	len_t getLength() 			const { return _len; }
	len_t getWarp() 			const { return _warp; }
	label_t getLabel() 			const { return _warp; }
};

// ================================================================
// Functions

//TODO have it accept an arbitrary functor as an element-wise distance
//measure--we apparently can't pass it the name of a function or a function
//pointer if we want a default value
//
//function pointer arg: dist_t (*distFunc)(data_t x, data_t y),
//functor arg: ..., class F> ... F distFunc
//lambda arg: F&& distFunc=[](data_t x, data_t y) {return (x-y)*(x-y);})
template <class data_t, class len_t, class F>
data_t dtw_full_dist(const data_t *__restrict v1, const data_t *__restrict v2,
	len_t m, len_t r, F&& distFunc, short int stride=1) {

	if (v1 == v2) return 0;	//distance is 0 if referring to same data
	assertf(r <= m - 1, "Warping width %d must be less than array length %d", r, m);
	assertf(r >= 0, "Warping width %d nonnegative", r);

	// if r is 0, dtw degerenerates to euclidean distance squared
	if (r == 0) {
		return L2(v1, v2, m);
	}

	// shared vars
	data_t  x, y, z;
	bool inNewWarpRow;
	//vars in inner loop; probably unnecessary to pull these out, but can't hurt
	len_t k = r;	//first entry in costs array is initially at index r
	len_t final_idx = m - 1;

	// Due to warping constraint, only need to maintain two distance
	// arrays of width (2*r + 1), since warping window extends out r
	// elements away from the midline; we allocate arrays on the stack, but
	// need pointers so we can swap them after each iteration
	data_t cost_ar[2*r + 1];
	data_t cost_prev_ar[2*r + 1];
	data_t* cost = &cost_ar[0];
	data_t* cost_prev = &cost_prev_ar[0];

	//calculate the first column separately, since no decisions to be made
	len_t lastRowInWindow  = 0 + r;
	cost_prev[k] = diff_sq(v1[0], v2[0]);
	k++;
	for (int j = 1; j <= lastRowInWindow; j++, k++) {
		cost_prev[k] = cost_prev[k-1] + diff_sq(v1[0],v2[j*stride]);
	}

	// calculate the remaining columns
	for (int i = 1; i < m; i++) {
		k = std::max(0, r - i);						//first entry in costs array
		len_t firstRowInWindow = std::max(0, i - r);
		len_t lastRowInWindow = std::min(final_idx, i + r);

		//handle first row in warping window separately,
		//since it has no vertical component
		bool inFirstRow = (i - r) <= 0;
		data_t minPrevDist = cost_prev[k+1];				//horizontal distance
		if (!inFirstRow && cost_prev[k] < minPrevDist) {	//diagonal distance
			minPrevDist = cost_prev[k];
		}
		cost[k] = minPrevDist + distFunc(v1[i*stride], v2[firstRowInWindow*stride]);
		k++;

		//for each remaining row in warping window, do classic DTW calculation
		for(int j = firstRowInWindow+1; j <= lastRowInWindow; j++, k++) {
			inNewWarpRow	= k + 1 > 2*r;
			x = inNewWarpRow ? INFINITY : cost_prev[k+1];	//horizontal
			y = cost_prev[k];								//diagonal
			z = cost[k - 1];								//vertical
			cost[k] = std::min(std::min(x, y), z) + distFunc(v1[i*stride], v2[j*stride]);
		}

		// Set current array as previous array
		data_t* cost_tmp = cost;
		cost = cost_prev;
		cost_prev = cost_tmp;

	} // end for column

	// the DTW distance is in the last cell in the matrix of size O(m^2),
	// which is in the middle of our warping window array.
	return cost_prev[r];
}

template <class data_t, class len_t>
data_t dtw_full_dist(const data_t *__restrict v1, const data_t *__restrict v2,
	len_t m, len_t r, short int stride=1) {
	return dtw_full_dist(v1, v2, m, r, [](data_t x, data_t y) {return (x-y)*(x-y);}, stride);
}

template <class data_t, class len_t, class dist_t>
dist_t _dtw_with_cumbound(const data_t *__restrict v1,
	const data_t *__restrict v2, const dist_t* cb,
	len_t m, len_t r, dist_t thresh, short int stride=1) {

	if (v1 == v2) return 0;	//distance is 0 if referring to same data
	assertf(r <= m-1, "Warping width %d >= array length %d", r, m);
	assertf(r >= 0, "Warping width %d is negative", r);

	//shared vars
	data_t  x, y, z;
	bool inNewWarpRow;
	//vars in inner loop; probably unnecessary to pull these out, but can't hurt
	len_t k = r;	//first entry in costs array is initially at index r
	len_t final_idx = m - 1;

	// Due to warping constraint, only need to maintain two distance
	// arrays of width (2*r + 1), since warping window extends out r
	// elements away from the midline; we allocate arrays on the stack, but
	// need pointers so we can swap them after each iteration
	dist_t cost_ar[2*r + 1];
	dist_t cost_prev_ar[2*r + 1];
	dist_t* cost = &cost_ar[0];
	dist_t* cost_prev = &cost_prev_ar[0];

	//calculate the first column separately, since no decisions to be made
	len_t lastRowInWindow  = 0 + r;
	cost_prev[k] = diff_sq(v1[0], v2[0]);
	k++;
	for (len_t j = 1; j <= lastRowInWindow; j++, k++) {
		cost_prev[k] = cost_prev[k-1] + diff_sq(v1[0],v2[j*stride]);
	}

	// calculate the remaining columns
	for (len_t i = 1; i < m; i++) {
		k = std::max(0, r - i);						//first entry in costs array
		len_t firstRowInWindow = std::max(0, i - r);
		len_t lastRowInWindow = std::min(final_idx, i + r);

		//handle first row in warping window separately,
		//since it has no vertical component
		bool inFirstRow = (i - r) <= 0;
		data_t minPrevDist = cost_prev[k+1];				//horizontal distance
		if (!inFirstRow && cost_prev[k] < minPrevDist) {	//diagonal distance
			minPrevDist = cost_prev[k];
		}
		cost[k] = minPrevDist + diff_sq(v1[i*stride], v2[firstRowInWindow*stride]);
		dist_t min_cost = cost[k];
		k++;

		//for each remaining row in warping window, do classic DTW calculation
		for(len_t j = firstRowInWindow+1; j <= lastRowInWindow; j++, k++) {
			inNewWarpRow	= k + 1 > 2*r;
			x = inNewWarpRow ? INFINITY : cost_prev[k+1];	//horizontal
			y = cost_prev[k];								//diagonal
			z = cost[k - 1];								//vertical
			cost[k] = std::min(std::min(x, y), z) + diff_sq(v1[i*stride], v2[j*stride]);

			// Find minimum cost in this column for possible early abandoning
            if (cost[k] < min_cost) {
                min_cost = cost[k];
            }
		}

		// We can abandon early if the current cummulative distace with lower
		// bound together are larger than bsf
		len_t cum_bound_index = i + r + 1;
		if (cum_bound_index < m ) {
			data_t min_total_cost = min_cost + cb[cum_bound_index];
			if (min_total_cost >= thresh) {
				return min_total_cost;
			}
		}

		// Set current array as previous array
		dist_t* cost_tmp = cost;
		cost = cost_prev;
		cost_prev = cost_tmp;

	} // end for column

	// the DTW distance is in the last cell in the matrix of size O(m^2),
	// which is in the middle of our warping window array.
	return cost_prev[r];
}

// 1D dynamic time warping with early abandoning
// Note that if the warping constraint is different
template <class data_t, class len_t, class dist_t=float>
dist_t dtw_abandon(const DTWExample<data_t,len_t>& x,
	const DTWExample<data_t,len_t>& y,
	dist_t thresh,
	const DTWTempStorage<data_t, len_t, dist_t>* storage=nullptr,
	int stride=1) {

	//compute ts length and warping constraint; use narrower warping
	//constraint from the two examples so the bounds from the LB_Keogh
	//envelopes are valid
	len_t len = x.getLength();
	len_t warp = x.getWarp();
	if (y.getWarp() >=0 && y.getWarp() < warp) {
		warp = y.getWarp();
	}

	//validate input
	assertf(len == y.getLength(),
		"DTW: lengths unequal: %d, %d", x.getLength(), y.getLength());
	assertf(warp >= 0, "Warping width %d nonnegative", warp);

	//if r is 0, dtw degerenerates to euclidean distance squared
	if (warp == 0) {
		return L2_abandon(&x.getData()[0], &y.getData()[0], &x.getOrder()[0],
						  len, thresh);
	}

	//ensure that we have somewhere to store the temporary arrays
	//computed as part of the dtw computation
	if (storage == nullptr) {
		storage = new DTWTempStorage<data_t, len_t, dist_t>(len);
	}

	//compute LB_Keogh distance between x's envelope and y
	const data_t* data_y 	= &y.getData()[0];
	const len_t* order 		= &x.getOrder()[0];
	const data_t* u 		= &x.getUpperEnvelope()[0];
	const data_t* l 		= &x.getLowerEnvelope()[0];
	dist_t* cb1Tmp			= (dist_t*)&(storage->cb1[0]);
	dist_t lb_k = ucr_lb_keogh_with_cumbound(data_y, order, u, l, cb1Tmp, len, thresh);

	if (lb_k > thresh) return lb_k;

	//compute LB_Keogh distance between y's envelope and x
	const data_t* data_x 	= &x.getData()[0];
	order 					= &y.getOrder()[0];
	u 						= &y.getUpperEnvelope()[0];
	l 						= &y.getLowerEnvelope()[0];
	dist_t* cb2Tmp 			= (dist_t*)&(storage->cb2[0]);
	dist_t lb_k2 = ucr_lb_keogh_with_cumbound(data_x, order, u, l, cb2Tmp, len, thresh);

	if (lb_k > thresh) return lb_k;

	//initialize cumulative lower bound; these are the reverse-order cumulative
	//sums of the distances to the envelope that yielded the higher distance
	len_t final_idx = len - 1;
	if (lb_k2 < lb_k) {
		cb1Tmp[final_idx] = cb2Tmp[final_idx];
		for(len_t k = final_idx - 1; k >= 0; k--)
			cb1Tmp[k] = cb1Tmp[k + 1] + cb2Tmp[k];
	} else {
		for(len_t k = final_idx - 1; k >= 0; k--)
			cb1Tmp[k] = cb1Tmp[k + 1] + cb1Tmp[k];
	}

	//neither lower bound was high enough; compute DTW distance (but still
	//early abandoning if possible, with help of above cumulative lower bound)
	return _dtw_with_cumbound(data_x, data_y, cb1Tmp, len, warp, thresh);
}

template <class data_t, class len_t, class dist_t>
auto dtw_abandon(const data_t *__restrict v1, const data_t *__restrict v2,
				   len_t m, len_t r, dist_t thresh,
				   short int stride=1, bool znormalize=true)
				   -> decltype(v1[0]*thresh) {
	int label = -1;
	DTWExample<data_t, len_t> x(v1, m, r, label, znormalize);
	DTWExample<data_t, len_t> y(v2, m, r, label, znormalize);

	data_t one = 1.0;	//1.0 if data_t is {float,double}, 1 otherwise
	auto threshWithRightType = one * thresh;  //data = float -> thresh = float
	return dtw_abandon(x,y,threshWithRightType);
}

template <class data_t, class len_t>
data_t dtw(const data_t *__restrict v1, const data_t *__restrict v2,
		   len_t m, len_t r, short int stride=1) {
	return dtw_full_dist(v1,v2,m,r,stride);
}

#endif /* defined(__TimeKit__dtw__) */
