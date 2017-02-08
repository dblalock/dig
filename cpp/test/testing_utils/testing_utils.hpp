//
//  testing_utils.hpp
//  Dig
//
//  Created by DB on 10/22/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef __TESTING_UTILS_HPP
#define __TESTING_UTILS_HPP

#include <iostream>
#include <random>
#include <stdint.h>

#include "memory.hpp"
#include "timing_utils.hpp" // for profiling

// TODO rm these functions
///used for comparisons ignoring slight floating point errors
short int approxEq(double a, double b);
double rnd(double a);

template<class DistT>
double prevent_optimizing_away_dists(DistT* dists, int64_t N) {
    // count how many dists are above a random threshold; let's see you
    // optimize this away, compiler
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(0, 255);
    uint8_t thresh = static_cast<uint8_t>(d(gen));

    volatile int64_t count = 0;
    for (int64_t n = 0; n < N; n++) { count += dists[n] > thresh; }
    volatile double frac = static_cast<double>(count) / N;
    // printf("(%d%%) ", static_cast<int>(frac * 100));
    return frac;
}

static inline void print_dist_stats(const std::string& name, int64_t N,
    double t_ms)
{
    printf("%s: %.2f (%.1fM/s)\n", name.c_str(), t_ms, N / (1e3 * t_ms));
}

template<class dist_t>
static inline void print_dist_stats(const std::string& name,
    const dist_t* dists, int64_t N, double t_ms)
{
    if (dists != nullptr) {
        // prevent_optimizing_away_dists(dists, N);
        // if (N < 100) {
        //     auto printable_ar = ar::add(dists, N, 0);
        //     ar::print(printable_ar.get(), N);
        // }
    }
    print_dist_stats(name, N, t_ms);
}

#define PROFILE_DIST_COMPUTATION(NAME, NITERS, DISTS_PTR, NUM_DISTS, EXPR)  \
    do {                                                                \
        double __t_min = std::numeric_limits<double>::max();            \
        for (int __i = 0; __i < NITERS; __i++) {                        \
            double __t = 0;                                             \
            {                                                           \
                EasyTimer _(__t);                                       \
                EXPR;                                                   \
            }                                                           \
            __t_min = __t < __t_min ? __t : __t_min;                    \
            prevent_optimizing_away_dists(DISTS_PTR, NUM_DISTS);        \
        }                                                               \
        print_dist_stats(                                               \
            NAME " (best of " #NITERS ")",                              \
            NUM_DISTS, __t_min);                                        \
    } while (0);


// TODO put this func in array_utils
template <class data_t, class len_t>
static inline void randint_inplace(data_t* data, len_t len,
                                   data_t min=std::numeric_limits<data_t>::min(),
                                   data_t max=std::numeric_limits<data_t>::max())
{
    assert(data != nullptr);
    assert(len > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> d(min, max);

    for (len_t i = 0; i < len; i++) {
        data[i] = static_cast<data_t>(d(gen));
    }
}
template <class data_t, class len_t>
static inline void rand_inplace(data_t* data, len_t len,
                                data_t min=0, data_t max=1)
{
    assert(data != nullptr);
    assert(len > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(min, max);

    for (len_t i = 0; i < len; i++) {
        data[i] = static_cast<data_t>(d(gen));
    }
}

template<class data_t>
static inline data_t* aligned_random_ints(int64_t len) {
    data_t* ptr = aligned_alloc<data_t>(len);
    randint_inplace(ptr, len);
    return ptr;
}

template<class data_t>
static inline data_t* aligned_random(int64_t len) {
    data_t* ptr = aligned_alloc<data_t>(len);
    rand_inplace(ptr, len);
    return ptr;
}

#endif
