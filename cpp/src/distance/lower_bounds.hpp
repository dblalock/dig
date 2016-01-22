//
//  lower_bounds.hpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TimeKit__lower_bounds__
#define __TimeKit__lower_bounds__

#include "deque.h"
#include "distance_utils.hpp"

// Compute min and max envelope for DTW with sakoe-chiba band of width r
template <class data_t, class len_t>
void build_envelope(const data_t *t, len_t len, len_t r, data_t *l, data_t *u) {
    deque du, dl;
    len_t i = 0;
	len_t width = 2 * r + 1;
	
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

template <class data_t, class len_t, class dist_t>
dist_t ucr_lb_keogh_with_cumbound(const data_t* t,
					const len_t* order, 
					const data_t* uo,
					const data_t* lo,
					dist_t* cb, 
					len_t len, dist_t bsf) {
      
    dist_t  lb = 0;
    data_t  x, d;
    for (len_t i = 0; i < len && lb < bsf; i++) {
        x = t[order[i]];
        if (x > uo[i]) {
            d = diff_sq(x, uo[i]);
        }
        else if(x < lo[i]) {
            d = diff_sq(x, lo[i]);
        } else {
			d = 0;
		}
        lb += d;
        cb[order[i]] = d;
    }
    return lb;
}

#endif // __TimeKit__lower_bounds__
