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

// TODO rm these functions
///used for comparisons ignoring slight floating point errors
short int approxEq(double a, double b);
double rnd(double a);

template<class DistT>
void prevent_optimizing_away_dists(DistT* dists, int64_t N) {
    volatile int64_t count = 0;
    for (int64_t n = 0; n < N; n++) { count += dists[n] > 3 * 42; }
    std::cout << "(" << count << "/" << N << ")\t";
}

// TODO put this func in array_utils
template <class data_t, class len_t>
static inline void randint_inplace(data_t* data, len_t len,
                                   data_t min=std::numeric_limits<data_t>::min(),
                                   data_t max=std::numeric_limits<data_t>::max())
{
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
