//
//  main.cpp
//  Dig
//
//  Created by DB on 1/20/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

// unit tests magic
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main(int argc, char *const argv[]) {

	// run Catch unit tests--it would be better of all of the above
	// had been done via Catch, but alas, I was a noob.
	int result = Catch::Session().run( argc, argv );

	return 0;
}
