//
//  test_distance_utils.cpp
//  TimeKit
//
//  Created by DB on 10/24/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include "catch.hpp"
#include "testing_utils.hpp"

#include "distance_utils.hpp"
#include "array_utils.h"

TEST_CASE( "z-normalization", "[distance_utils]" ) {
	
	SECTION("double") {
		typedef double data_t;
		
		unsigned int len = 6;
		data_t x[] = {.7, 0, -.3, 11, -.6, 4};
		znormalize(x, len);
		data_t mean = array_mean(x, len);
		
		REQUIRE(rnd(mean) == 0);
	}
}
