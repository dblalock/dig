//  neighbor_testing_utils.hpp
//
//  Dig
//
//  Created by DB on 9/15/16
//  Copyright © 2016 D Blalock. All rights reserved.

#ifndef __NEIGHBOR_TESTING_UTILS_HPP
#define __NEIGHBOR_TESTING_UTILS_HPP

//void require_neighbors_same(const Neighbor& nn, const Neighbor& trueNN) {
//    REQUIRE(nn.idx == trueNN.idx);
//    REQUIRE(std::abs(nn.dist - trueNN.dist) < .0001);
//}

// macro so that, if it fails, failing line is within the test
#define REQUIRE_NEIGHBORS_SAME(nn, trueNN) \
    REQUIRE(nn.idx == trueNN.idx); \
    REQUIRE(std::abs(nn.dist - trueNN.dist) < .0001);

#endif // __NEIGHBOR_TESTING_UTILS_HPP
