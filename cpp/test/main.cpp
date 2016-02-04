//
//  main.cpp
//  Dig
//
//  Created by DB on 1/20/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include <stdio.h>

//#include <vector> // TODO remove
//#include <unordered_map> // TODO remove

// unit tests magic
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char *const argv[]) {
	printf("running tests...\n");
	
//	std::vector<int16_t>v;
//	std::unordered_map<int16_t, int64_t>m;
//	printf("vect size, capacity = %ld, %ld\n", v.size(), v.capacity());
//	printf("map size, buckets = %ld, %ld\n", m.size(), m.bucket_count());
//	printf("vect sizeof, map sizeof = %ld, %ld\n", sizeof(v), sizeof(m));
//	// ^ interesting: container sizes independent of element sizes
//	
//	v.reserve(8);
//	printf("after reserve vect size, capacity = %ld, %ld\n", v.size(), v.capacity());
//	// ^ excellent; this only allocates how much resize asked for
	
	return Catch::Session().run( argc, argv );
}
