//
//  test_dtw.cpp
//  TimeKit
//
//  Created by DB on 10/22/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include "catch.hpp"
#include "testing_utils.hpp"
#include "dtw.hpp"

TEST_CASE( "dtw with and without early abandoning", "[dtw]" ) {
	
	SECTION("double") {
		constexpr int len = 5;
		double a[len] = {5,2,2,3,5.1};
		double b[len] = {5,2,3,3,4};
		short stride = 1;
		bool znormalize = false;
		
		SECTION("full distance") {

			SECTION("warp 1") {
				int warp = 1;
				double ans = 1.1*1.1;
				
				double dist = dtw_full_dist(&a[0], &b[0], len, warp);
				REQUIRE( rnd(dist) == rnd(ans) );
				
				dist = dtw(&a[0], &b[0], len, warp);
				REQUIRE( rnd(dist) == rnd(ans) );
			}
		}
		
		SECTION("early abandon") {
			SECTION("warp1") {
				int warp = 1;
				SECTION("warp 1, no abandon") {
					int warp = 1;
					double ans = 1.1*1.1;
					int thresh = 99;		//note that this is an int, so has to infer
					double dist = dtw_abandon(&a[0], &b[0], len, warp, thresh, stride, znormalize);
					REQUIRE( rnd(dist) == rnd(ans) );
				}

				SECTION("warp 1, abandon") {
					double c[len] = {5,2.1,2,3,5.1};
					double d[len] = {5,2,3.1,3,4};
					int warp = 1;
					double fullDist = 1.1*1.1;
					double thresh = .1;
					double dist = dtw_abandon(&a[0], &b[0], len, warp, thresh, stride, znormalize);
					REQUIRE( dist >= thresh );
					REQUIRE( dist < fullDist);
				}
			}
		}
	}
	
	SECTION("int") {
		constexpr int len = 5;
		double a[len] = {5,2,2,3,5};
		double b[len] = {5,2,3,3,4};
		short stride = 1;
		bool znormalize = false;
		
		SECTION("full distance") {
			
			SECTION("warp 1") {
				int warp = 1;
				double ans = 1;
				
				double dist = dtw_full_dist(&a[0], &b[0], len, warp);
				REQUIRE( rnd(dist) == rnd(ans) );
				
				dist = dtw(&a[0], &b[0], len, warp);
				REQUIRE( rnd(dist) == rnd(ans) );
			}
		}
		
		SECTION("early abandon") {
			SECTION("warp1") {
				int warp = 1;
				
				SECTION("warp 1, no abandon") {
					double ans = 1;
					int thresh = 99;		//note that this is an int, so has to infer
					double dist = dtw_abandon(&a[0], &b[0], len, warp, thresh, stride, znormalize);
					REQUIRE( rnd(dist) == rnd(ans) );
				}
				
				SECTION("warp 1, abandon") {
					int c[len] = {5,3,2,3,5};
					int d[len] = {5,2,3,3,4};
					
					double fullDist = 2;
					double thresh = .1;
					double dist = dtw_abandon(&a[0], &b[0], len, warp, thresh, stride, znormalize);
					REQUIRE( dist >= thresh );
					REQUIRE( dist < fullDist);
				}
			}
		}
	}
}
