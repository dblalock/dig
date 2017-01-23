//
//  lower_bounds.hpp
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __Dig_lower_bounds_hpp
#define __Dig_lower_bounds_hpp

#include <vector>
#include "immintrin.h"

#include "deque.h"
#include "distance_utils.hpp"

// ================================================================ l2

namespace dist {
namespace quantize {

static const auto kSplitHighLowIdxs = _mm256_set_epi8(
    31, 29, 27, 25, 23, 21, 19, 17,
    15, 13, 11, 9,  7,  5,  3,  1,
    30, 28, 26, 24, 22, 20, 18, 16,
    14, 12, 10,  8,  6,  4,  2, 0);
static const auto kZeros = _mm256_set1_epi8(0);
static const auto kOnes8 = _mm256_set1_epi8(1);

// NOTE: technically requires that X and y be 32B aligned, although Haswell
// and above will probably generate unaligned loads regardless because there's
// no performance penalty
template<int ncols, class idx_t=int32_t>
std::vector<idx_t> radius(const uint8_t* X, size_t nrows, const uint8_t* y,
    int32_t thresh)
{
    static_assert(__AVX2__, "AVX 2 is required!");
    static_assert(ncols % 32 == 0, "ncols must be a multiple of 32!");
    static_assert(ncols >= 32, "ncols must be at least 32!");

    auto y_ = _mm256_load_si256(reinterpret_cast<const __m256i*>(y));
    std::vector<idx_t> ret;
    for (int i = 0; i < nrows; i++) {
        // auto all_prods = kZeros;
        volatile auto all_prods = kZeros; // TODO rm after profile
        for (int j = 0; j <= (ncols - 32); j += 32) {
            // compute 16 partial dists; we do this by taking the difference
            // between the quantized reprs of each vector, subtracting one (but
            // clamping at 0), and then squaring the difference. maddubs is a
            // the multiplication instruction; it takes adjacent pairs of bytes
            // from both x and y, multiplies them elementwise, and sums the
            // two products horizontally, giving us 16 epi16s.
            auto addr = X + (ncols * i) + j;
            // volatile auto _DEBUG_ = *addr;
            auto x_ = _mm256_stream_load_si256(
                reinterpret_cast<const __m256i*>(addr));    // latency, thruput
            auto diffs = _mm256_subs_epi8(x_, y_);                      // 1, -
            auto abs_diffs = _mm256_abs_epi8(diffs);                    // 1, -
            auto lb_diffs = _mm256_subs_epu8(abs_diffs, kOnes8);        // 1, -
            auto prods = _mm256_maddubs_epi16(abs_diffs, lb_diffs);     // 5, 1
            all_prods = _mm256_adds_epu16(all_prods, prods);            // 1, -
        }
        // Note: just the above loop, with no reduction, can search thru an
        // N x D matrix in 1s with N * D = 51M. I.e., ~19GFLOPS, not that
        // they're actually floats
        // With the reduction, it's about 18GFLOPS with N, D = 100k, 128
        // With D = 64, no reduction is still like 19.5 GFLOPS, while with
        // reduction (but still no append to list) is like 13.9

        // reduce; we shuffle the epi16s so that the most significant 8
        // bytes are in the upper half and the least significant bytes are
        // in the lower half; we then use sad_epu8 to reduce groups of 8
        // bytes into the lower bits of the corresponding 64-bit elements;
        // we then add the adjacent pairs of 64-bit elements to get the
        // sums for the upper and lower bits; the overall sum is the sum
        // from the lower bits by the sum from the upper bits << 8.
        // auto shuffled = _mm256_shuffle_epi8(prods, kSplitHighLowIdxs); //1,1
        // auto shuffled = _mm256_shuffle_epi8(all_prods, kSplitHighLowIdxs);//1,1
        // auto sums = _mm256_sad_epu8(shuffled, kZeros);                  // 5, 1
        // auto sums_shft = _mm256_srli_si256(sums, 8);                    // 1, -
        // auto dists = _mm256_add_epi64(sums, sums_shft);                 // 1, .5
        // auto high_dist = _mm256_extract_epi64(dists, 2);                // NA
        // auto low_dist = _mm256_extract_epi64(dists, 0);                 // NA
        // volatile int64_t dist = low_dist + (high_dist << 8);                     // ?
        // // if (UNLIKELY(dist < thresh)) {
        // //     ret.push_back(static_cast<idx_t>(dist));
        // // }
        // // TODO uncomment after profiling
    }
    return ret;
}

} // namespace lb
} // namespace dist

// ================================================================ dtw

// TODO put stuff below in namespace dist also

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

#endif // __Dig_lower_bounds_hpp
