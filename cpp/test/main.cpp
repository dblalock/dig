//
//  main.cpp
//  Dig
//
//  Created by DB on 1/20/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include <stdio.h>

// unit tests magic
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

//#include <map> // TODO remove
#include <memory>

#include "intmap.hpp"

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
	
	printf("size of unique_ptr: %ld\n", sizeof(std::unique_ptr<double*>));
	printf("size of raw ptr: %ld\n", sizeof(double*));
	
	return Catch::Session().run(argc, argv);
//
//	std::map<int,char> example = {{1,'a'},{2,'b'},{3,'c'},{4,'d'}};
// 
//	auto it = example.lower_bound(2);
//	std::map<int,char>::reverse_iterator itr(it);
//	
//	int i = 0;
//	while (it != example.end()) {
//		std::cout << it->first << ": " << it->second << '\n';
//		std::cout << "\t" << itr->first << ": " << itr->second << '\n';
//		++it;
//		i++;
//		if (i > 1) {
//			it = example.end(); // succesfully breaks the loop
//		}
//	}
//	std::cout << "================================\n";
//	// so reverse itr starts pointing at element *before* forward iterator;
//	// that's weird. Also unaffected by any modification of forward iter, as
//	// it should be.
//	while (itr != example.rend()) {
//		std::cout << itr->first << ": " << itr->second << '\n';
//		++itr;
//	}
	
	
	return 0;
}
