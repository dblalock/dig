//
//  bolt.hpp
//  Dig
//
//  Created by DB on 2017-2-3
//  Copyright (c) 2016 DB. All rights reserved.
//


#ifndef __BOLT_HPP
#define __BOLT_HPP

// #include <iostream> // TODO rm

#include <assert.h>
#include <sys/types.h>
#include "immintrin.h"

//#include "macros.hpp"
// #include "multi_codebook.hpp"



static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");

namespace {

// // see Eigen/src/Core/arch/AVX/PacketMath.h
// float predux_max(const __m256& a)
// {
//     // 3 + 3 cycles
//     auto tmp = _mm256_max_ps(a, _mm256_permute2f128_ps(a,a,1));
//     // 3 + 3 cycles (_MM_SHUFFLE is a built-in macro that generates constants)
//     tmp = _mm256_max_ps(tmp, _mm256_shuffle_ps(tmp,tmp,_MM_SHUFFLE(1,0,3,2)));
//     // 1 cycle + 3 cycles + 1 cycle
//     return pfirst(_mm256_max_ps(tmp, _mm256_shuffle_ps(tmp,tmp,1)));
// }

// // see Eigen/src/Core/arch/AVX/PacketMath.h
// float pfirst(const __m256& a) {
//   return _mm_cvtss_f32(_mm256_castps256_ps128(a));
// }

// returns a * b + c, elementwise; see eigen/src/Core/arch/AVX/PacketMath.h
inline __m256 fma(__m256 a, __m256 b, __m256 c) {
    __m256 res = c;
    __asm__("vfmadd231ps %[a], %[b], %[c]" : [c] "+x" (res) : [a] "x" (a), [b] "x" (b));
    return res;
}

template<class T>
static int8_t msb_idx_u32(T x) {
    return 8*sizeof(uint32_t) - 1 - __builtin_clzl((uint32_t)x);
}

template<class T>
inline __m256i load_si256i(T* ptr) {
    return _mm256_load_si256((__m256i *)ptr);
}

template<class T>
inline __m256i stream_load_si256i(T* ptr) {
    return _mm256_stream_load_si256((__m256i *)ptr);
}

//
//inline __m256i load_256f(float* ptr) {
//    return _mm256_load_ps((__m256 *)ptr);
//}


/**
 * @brief Encode a matrix of floats using Bolt.
 * @details [long description]
 *
 * @param X Congtiguous, row-major matrix whose rows are the vectors to encode
 * @param nrows Number of rows in X
 * @param ncols Number of columns in X; must be divisible by 2 * NBytes
 * @param centroids A set of 16 * 2 * NBytes centroids in contiguous vertical
 *  layout, as returned by bolt_encode_centroids.
 * @param out Array in which to store the codes; must be of length
 *  nrows * NBytes
 * @tparam NBytes Byte length of Bolt encoding for each row
 */
template<int NBytes>
void bolt_encode(const float* X, int64_t nrows, int ncols,
    const __m256* centroids, uint8_t* out)
{
    static constexpr int lut_sz = 16;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = 2 * NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = ncols / ncodebooks;
    const int trailing_subvect_len = ncols % ncodebooks;
    assert(trailing_subvect_len == 0); // TODO remove this constraint

    __m256 accumulators[lut_sz / packet_width];
    alignas(32) float argmin_storage[16];

    for (int64_t n = 0; n < nrows; n++) { // for each row of X
        auto x_ptr = X + n * ncols;
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_ps();
        }

        for (int m = 0; m < NBytes * 2; m++) { // for each codebook
            // compute distances to each of the centroids, which we assume
            // are in column major order; this takes 2 packets per col
            auto centroids_ptr = centroids;
            for (int j = 0; j < subvect_len; j++) { // for each encoded dim
                auto x_j_broadcast = _mm256_set1_ps(*x_ptr);
                for (int i = 0; i < nstripes; i++) { // for upper and lower 8
                    auto centroids_half_col = _mm256_load_ps((float*)centroids_ptr);
                    // centroids_ptr += packet_width;
                    centroids_ptr++;
                    auto diff = _mm256_sub_ps(x_j_broadcast, centroids_half_col);
                    auto prods = fma(diff, diff, accumulators[i]);
                    accumulators[i] = _mm256_add_ps(accumulators[i], prods);

                    _mm256_store_ps(
                        argmin_storage + packet_width * i, accumulators[i]);
                }
                x_ptr++;
            }

            // compute argmax; we just write out the array and iterate thru
            // it because max reduction, then broadcast, then movemask, then
            // clz, is like 25 cycles and uses the ymm registers
            //
            // TODO see if SIMD version is faster
            //
            // TODO try hybrid; reduce to 8f, then shuffle / check within
            // 128bit lanes, then check get max directly from the 2 possible
            // vals in low floats of each 128bit lane
            //
            // TODO try storing val < min_val in a bit, with
            // minval = min(minval, val) to get min, then find last 1
            auto min_val = argmin_storage[0];
            uint8_t min_idx = 0;
            // uint32_t indicators = 1;
            for (int i = 1; i < lut_sz; i++) {
                auto val = argmin_storage[i];
                // min_val = min_val < val ? val : min_val;
                // indicators |= static_cast<uint64_t>(val < min_val) << i;
                if (val < min_val) {
                    min_val = val;
                    min_idx = i;
                }
            }
            // uint8_t min_idx = msb_idx_u32(indicators);

            if (m % 2) {
                out[m] |= min_idx << 4; // odds -> store in upper 4 bits
            } else {
                // TODO pretty sure we don't actually need to mask
                out[m] = min_idx & 0x0F; // evens -> store in lower 4 bits
            }
        }
        out += NBytes;
    }
}

/**
 * @brief Create a lookup table (LUT) containing the distances from the
 * query q to each of the centroids.
 *
 * @details Centroids must be in vertical layout, end-to-end, as returned by
 * bolt_encode_centroids. Eg, if we only had two centroids instead of 16, and
 * they were [1, 2, 3, 4], [11,12,13,14], with two subvectors, memory should
 * be column-major and look like:
 *
 * 1    2   3   4
 * 11   12  13  14
 *
 * The LUT will also be written in column major order. If the query were
 * [0 0 0 0], the LUT resulting LUT would be:
 *
 * (1^2 + 2^2)      (3^2 + 4^2)
 * (11^2 + 12^2)    (13^2 + 14^2)
 *
 * @param q The (not-necessarily aligned) query vector for which to compute
 *  the LUT. Elements of the query must be contiguous.
 * @param len The length of the query, measured as the number of elements.
 * @param centroids A set of 16 * 2 * NBytes centroids in contiguous vertical
 *  layout, as returned by bolt_encode_centroids.
 * @param out 32B-aligned storage in which to write the look-up table. Must
 *  be of length at least 16 * 2 * NBytes
 * @tparam NBytes Byte length of Bolt encoding
 */
template<int NBytes>
void bolt_lut_l2(const float* q, int len, const __m256* centroids, __m256i* out) {
    static constexpr int lut_sz = 16;
    static constexpr int packet_width = 8; // objs per simd register
    static constexpr int nstripes = lut_sz / packet_width;
    static constexpr int ncodebooks = 2 * NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = len / ncodebooks;
    const int trailing_subvect_len = len % ncodebooks;
    assert(trailing_subvect_len == 0); // TODO remove this constraint

    __m256 accumulators[nstripes];
    __m256i dists_uint16_0 = _mm256_undefined_si256();

    for (int m = 0; m < 2 * NBytes; m++) { // for each codebook
        for (int i = 0; i < nstripes; i++) {
            accumulators[i] = _mm256_setzero_ps();
        }
        for (int j = 0; j < subvect_len; j++) { // for each dim in subvect
            auto q_broadcast = _mm256_set1_ps(q[j]);
            for (int i = 0; i < nstripes; i++) { // for upper 8, lower 8 floats
                auto centroids_col = _mm256_load_ps((float*)centroids);
                centroids += packet_width;

                auto diff = _mm256_sub_ps(q_broadcast, centroids_col);
                auto prods = fma(diff, diff, accumulators[i]);
                accumulators[i] = _mm256_add_ps(accumulators[i], prods);
            }
        }
        // nevermind, second pack undoes first one's shuffling
        // static constexpr __m256i shuffle_idxs = _mm256_set_epi8(
        //     31, 30, 29, 28, 27, 26, 25, 24,
        //     15, 14, 13, 12, 11, 10, 9,  8,
        //     23, 22, 21, 20, 19, 18, 17, 16,
        //     7,  6,  5,  4,  3,  2,  1,  0);

        // convert the floats to ints
        auto dists_int32_low = _mm256_cvtps_epi32(accumulators[0]);
        auto dists_int32_high = _mm256_cvtps_epi32(accumulators[1]);

        // because we saturate to uint8s, we only get 32 objs to write after
        // two 16-element codebooks
        if (m % 2) {
            // if odd-numbered codebook, combine dists from previous codebook
            // with these dists
            auto dists_uint16_1 = _mm256_packus_epi32(dists_int32_low, dists_int32_high);
            auto dists_uint8 = _mm256_packus_epi16(dists_uint16_0, dists_uint16_1);
            _mm256_store_si256((__m256i*)out, dists_uint8);
            // out += packet_width;
            out++; // just increment by one because type is __mm26i*
        } else {
            // if even-numbered codebook, just store these dists to be combined
            // when we look at the next codebook
            dists_uint16_0 = _mm256_packus_epi32(dists_int32_low, dists_int32_high);
        }
    }
}

// basically just a transpose with known centroid sizes
// note that this doesn't have to be fast because we do it once after training
template<int NBytes>
void bolt_encode_centroids(const float* centroids, int ncols, float* out) {
    static constexpr int lut_sz = 16;
    static constexpr int ncodebooks = 2 * NBytes;
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    const int subvect_len = ncols / ncodebooks;
    const int trailing_subvect_len = ncols % ncodebooks;
    assert(trailing_subvect_len == 0); // TODO remove this constraint

    for (int m = 0; m < 2 * NBytes; m++) {
        // for each codebook, just copy rowmajor to colmajor, then offset
        // the starts of the centroids and out array
        for (int i = 0; i < lut_sz; i++) { // for each centroid
            auto in_row_ptr = centroids + subvect_len * i;
            for (int j = 0; j < subvect_len; j++) { // for each dim
                auto in_ptr = in_row_ptr + j;
                auto out_ptr = out + (16 * j) + i;
            }
        }
        centroids += 16 * subvect_len;
        out += 16 * subvect_len;
    }
}

template<int NBytes>
inline void bolt_scan_l2(const uint8_t* codes,
    const uint8_t* luts, uint8_t* dists_out, int64_t nblocks)
{
    static_assert(NBytes > 0, "Code length <= 0 is not valid");
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    // unpack 16B luts into 32B registers
    __m256i luts_ar[NBytes * 2];
    auto lut_ptr = luts;
    for (uint8_t j = 0; j < NBytes; j++) {
        auto both_luts = load_si256i(lut_ptr);
        lut_ptr += 32;
        auto lut0 = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
        auto lut1 = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));
        luts_ar[2 * j] = lut0;
        luts_ar[2 * j + 1] = lut1;
    }

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        for (uint8_t j = 0; j < NBytes; j++) {
            // auto x_col = load_si256i(codes);
            auto x_col = stream_load_si256i(codes);
            codes += 32;

            auto lut_low = luts_ar[2 * j];
            auto lut_high = luts_ar[2 * j + 1];

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_shft = _mm256_srli_epi16(x_col, 4);
            auto x_high = _mm256_and_si256(x_shft, low_4bits_mask);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);
        }
        // _mm256_store_si256((__m256i*)dists_out, totals);
        _mm256_stream_si256((__m256i*)dists_out, totals); // "non-temporal memory hint"
        dists_out += 32;
    }
}

} // anon namespace

// namespace dist {


// } // namespace dist
#endif // __BOLT_HPP