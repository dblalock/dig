//
//  multi_codebook.hpp
//  Dig
//
//  Created by DB on 2017-1-22
//  Copyright (c) 2016 DB. All rights reserved.
//


#ifndef __MULTI_CODEBOOK_HPP
#define __MULTI_CODEBOOK_HPP

#include <sys/types.h>
#include "immintrin.h"

#include "macros.hpp"

namespace dist {

static const uint8_t mask_low4b = 0x0F;

template<class T>
inline __m256i load_si256i(T* ptr) {
    return _mm256_load_si256((__m256i *)ptr);
}



// experimental version to see what bottlenecks are
inline void incorrect_block_lut_dists_32x8B_4b(const uint8_t* codes, const uint8_t* luts,
    uint8_t* dists_out, int64_t nblocks)
{
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        // auto x_col = load_si256i(codes);
        // auto both_luts = load_si256i(luts);
//        __m256i both_luts;
        __m256i four_luts = _mm256_undefined_si256();
        // __m256i luts0, luts1;
        // __m256i lut_low, lut_high;
        for (uint8_t j = 0; j < 8; j++) {
            auto x_col = load_si256i(codes);

            __m256i both_luts;
            if (j % 2 == 0) {
                four_luts = load_si256i(luts);
                luts += 32;
                both_luts = _mm256_permute2x128_si256(four_luts, four_luts, 0 + (0 << 4));
            } else {
                both_luts = _mm256_permute2x128_si256(four_luts, four_luts, 1 + (1 << 4));
            }
            // unpack lower and upper 4 bits into luts for lower and upper 4
            // bits of codes
            auto lut_low = _mm256_and_si256(both_luts, low_4bits_mask);
            auto lut_high = _mm256_srli_epi16(both_luts, 4);
            lut_high = _mm256_and_si256(lut_high, low_4bits_mask);

            // auto both_luts = load_si256i(luts);
            // auto lut_low = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            // auto lut_high = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_high = _mm256_srli_epi16(x_col, 4);
            x_high = _mm256_and_si256(x_high, low_4bits_mask);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            // TODO uncomment after debug
            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);

            codes += 32;
        }
        _mm256_store_si256((__m256i*)dists_out, totals);
        luts -= 4 * 32;
        dists_out += 32;
    }
}

// version that unpacks 16B LUTs into 32B to cut out half the loads
inline void block_lut_dists_32x8B_4b_unpack(const uint8_t* codes,
    const uint8_t* luts, uint8_t* dists_out, int64_t nblocks)
{
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        // auto x_col = load_si256i(codes);
        // auto both_luts = load_si256i(luts);
        for (uint8_t j = 0; j < 8; j++) {
            auto x_col = load_si256i(codes);

            // unpack lower and upper 16B into two 32B luts
            // NOTE: cast + broadcast seems no faster, and more complicated
            auto both_luts = load_si256i(luts);
            // auto lower_128 = _mm256_castsi256_si128(both_luts);
            // auto lut_low = _mm256_broadcastsi128_si256(lower_128);
            auto lut_low = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            auto lut_high = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

            // auto lut_high = both_luts;
            // auto lut_low = load_si256i(luts);
            // auto lut_high = load_si256i(luts + 32);

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_high = _mm256_srli_epi16(x_col, 4);
            x_high = _mm256_and_si256(x_high, low_4bits_mask);
            // auto x_low = x_col;
            // auto x_high = _mm256_srli_epi16(x_col, 4);

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            // TODO uncomment after debug
            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);

            codes += 32;
            // luts += 64;
            luts += 32;
        }
        _mm256_store_si256((__m256i*)dists_out, totals);
        // _mm256_stream_si256((__m256i*)dists_out, totals); // "non-temporal memory hint"
        // luts -= 8 * 64;
        luts -= 8 * 32;
        dists_out += 32;
    }
}

inline void block_lut_dists_32x8B_4b(const uint8_t* codes, const uint8_t* luts,
    uint8_t* dists_out, int64_t nblocks)
{
    static const __m256i low_4bits_mask = _mm256_set1_epi8(0x0F);

    for (int64_t i = 0; i < nblocks; i++) {
        auto totals = _mm256_setzero_si256();
        for (uint8_t j = 0; j < 8; j++) {
            auto x_col = load_si256i(codes);
            auto lut_low = load_si256i(luts);
            auto lut_high = load_si256i(luts + 32);
            // auto both_luts = load_si256i(luts);
            // auto lut_low = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
            // auto lut_high = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));

            // compute distances via lookups; we have one table for the upper
            // 4 bits of each byte in x, and one for the lower 4 bits; the
            // shuffle instruction always looks at the lower 4 bits, so we
            // have to shift x to look at its upper 4 bits; also note that
            // we have to mask out the upper bit because the shuffle
            // instruction will zero the corresponding byte if this bit is set
            auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
            auto x_high = _mm256_srli_epi16(x_col, 4);
            x_high = _mm256_and_si256(x_high, low_4bits_mask);

            // TODO try 16B LUTs + broadcast upper and lower

            auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
            auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

            totals = _mm256_adds_epu8(totals, dists_low);
            totals = _mm256_adds_epu8(totals, dists_high);

            codes += 32;
            luts += 64;
            // luts += 32;
        }
        _mm256_store_si256((__m256i*)dists_out, totals);
        // _mm256_stream_si256((__m256i*)dists_out, totals); // "non-temporal memory hint"
       luts -= 8 * 64;
        // luts -= 8 * 32;
        dists_out += 32;
    }
}

// for debugging; should have same behavior as above func
inline void naive_block_lut_dists_32x8B_4b(const uint8_t* codes, const uint8_t* luts,
    uint8_t* dists_out, int64_t nblocks)
{
    for (int64_t b = 0; b < nblocks; b++) {
        // auto totals = _mm256_setzero_si256();
        for (uint8_t i = 0; i < 32; i++) {
            dists_out[i] = 0;
        }

        for (uint8_t j = 0; j < 8; j++) {
            for (uint8_t i = 0; i < 32; i++) {
                auto code = codes[i];
                auto low_bits = code & mask_low4b;
                auto high_bits = code >> 4;

                // what if we just use what the luts *should* be?
                // dists_out[i] += popcount(low_bits ^ (j << 1));
                // dists_out[i] += popcount(high_bits ^ ((j << 1) + 1));

                auto offset = 16 * (i >= 16); // look in 2nd 16B of lut
                auto lut_low = luts + offset;
                auto lut_high = luts + 32 + offset;
                dists_out[i] += lut_low[low_bits];
                dists_out[i] += lut_high[high_bits];
            }
            codes += 32;
            luts += 64;
        }
        luts -= 8 * 64;
        dists_out += 32;
    }
}

// luts must be of size [16][16]
template<class dist_t>
inline void lut_dists_8B_4b(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        auto lut_ptr = luts;
//        auto lut_ptr_right = luts + 16;
        for (uint8_t j = 0; j < 8; j++) {
            uint8_t code_high = static_cast<uint8_t>(codes[j] >> 4);
            uint8_t code_low = static_cast<uint8_t>(codes[j] & mask_low4b);

            // // TODO rm after debug
            // dists_out[i] += popcount(code_left ^ j);
            // dists_out[i] += popcount(code_right ^ j);

            auto lut_low = lut_ptr;
            auto lut_high = lut_low + 16;
            dists_out[i] += lut_low[code_low];
            dists_out[i] += lut_high[code_high];
            auto dist = lut_low[code_low] + lut_high[code_high];
            // std::cout << "---- " << (int)j << "\n";
            // std::cout << "code: " << (int)codes[j] << "\n";
            // std::cout << "code_low: " << (int)code_low << "\n";
            // std::cout << "code_high: " << (int)code_high << "\n";
            // std::cout << "dist: " << (int)dist << "\n";
            // std::cout <i< "dist: " << (int)dist << "\n";
            lut_ptr += 32;
        }
        codes += 8;
    }
}

template<int NBytes, class dist_t>
inline void lut_dists_8b(const uint8_t* codes, const dist_t* luts,
    dist_t* dists_out, int64_t N)
{
    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < NBytes; j++) {
            dists_out[i] += luts[j][codes[j]];
        }
        codes += NBytes;
    }
}

template<class dist_t>
inline void lut_dists_8B_8b_stride4b(const uint8_t* codes,
    const dist_t* luts, dist_t* dists_out, int64_t N)
{
    // sum LUT distances both along byte boundaries and shifted by 4 bits
    // note that the shifted lookups assume that the corresponding LUT entries
    // are after the 8 entries for the non-shifted lookups
    static const int NBytes = 8;
    for (int64_t i = 0; i < N; i++) {
        dists_out[i] = 0;
        for (int j = 0; j < NBytes; j++) {
            dists_out[i] += luts[j][codes[j]];
        }
        // TODO possibly shift 4 of these at once using SIMD instrs
        auto codes_as_uint = reinterpret_cast<const uint64_t*>(codes);
        auto shifted = (*codes_as_uint) >> 4;
        auto shifted_codes = reinterpret_cast<const uint8_t*>(shifted);
        for (int j = 0; j < NBytes - 1; j++) {
            dists_out[i] += luts[j + NBytes][shifted_codes[j]];
        }
        codes += NBytes;
    }
}

inline void popcount_8B(const uint8_t* codes, const uint64_t q,
    uint8_t* dists_out, int64_t N)
{
    for (int64_t i = 0; i < N; i++) {
        auto row_ptr = reinterpret_cast<const uint64_t*>(codes + (8 * i));
        dists_out[i] = __builtin_popcountll((*row_ptr) ^ q);
    }
}

} // namespace dist
#endif // __MULTI_CODEBOOK_HPP
