//
//  test_bitops.cpp
//  Dig
//
//  Created by DB on 2/14/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "bit_ops.hpp"

#include <iostream>
#include "catch.hpp"

#include "array_utils.hpp"

template<class T>
void isBitSetTest() {
	int numBits = sizeof(T) * 8;
	// int maxIdx = numBits - 1;
	T x = 0;
	// no bits set when x is 0
	for (int i = 0; i < numBits; i++) {
		REQUIRE_FALSE(isBitSet(x, i));
	}
	// appropriate bit set when x nonzero; other bits not set
	for (int idx = 0; idx < numBits; idx++) {
		x = oneAtIdx<T>(idx);
		CAPTURE(x);
		CAPTURE(idx);
		if (!isBitSet(x, idx)) {
			dumpBits(x);
			T mask = oneAtIdx<T>(idx);
			dumpBits(mask);
			// dumpBits(x & m)
		}
		REQUIRE(isBitSet(x, idx));
		for (uint8_t j = 0; j < numBits; j++) {
			if (j == idx) {
				continue;
			}
			CAPTURE((int)j);
			REQUIRE_FALSE(isBitSet(x, j));
		}
	}
}
TEST_CASE("isBitSet", "bit_ops") {
	isBitSetTest<int8_t>();
	isBitSetTest<int16_t>();
	isBitSetTest<int32_t>();
	isBitSetTest<int64_t>();
	isBitSetTest<uint8_t>();
	isBitSetTest<uint16_t>();
	isBitSetTest<uint32_t>();
	isBitSetTest<uint64_t>();
}

template<class T>
void isBitUnsetTest() {
	int numBits = sizeof(T) * 8;
	for (int idx = 0; idx < numBits; idx++) {
		T x = oneAtIdx<T>(idx);
		// set some other bits so it can't pass by always
		//  acting like x is just 0
		auto idxs = ar::rand_ints(0, numBits-1, 3);
		for (auto idx : idxs) {
			x |= oneAtIdx<T>(idx);
		}
		CAPTURE(x);
		CAPTURE(idx);
		REQUIRE(isBitSet(x, idx));
		uint64_t y = unsetBit(x, idx);
		REQUIRE_FALSE(isBitSet(y, idx));
	}
}
TEST_CASE("isBitUnset", "bit_ops") {
	isBitUnsetTest<int8_t>();
	isBitUnsetTest<int16_t>();
	isBitUnsetTest<int32_t>();
	isBitUnsetTest<int64_t>();
	isBitUnsetTest<uint8_t>();
	isBitUnsetTest<uint16_t>();
	isBitUnsetTest<uint32_t>();
	isBitUnsetTest<uint64_t>();
}


template<class T>
void popcountTest() {
	int numBits = sizeof(T) * 8;
	T x = 0;
	REQUIRE(popcount(x) == 0);
	for (int count = 1; count <= numBits; count++) {
		x = 0;
		auto idxs = ar::rand_ints(0, numBits-1, count);
		REQUIRE(idxs.size() == count);
		for (auto idx : idxs) {
			x = setBit(x, idx);
		}
		if (popcount(x) != count) {
			printf("it's gon break!\n");
			dumpBits(x);
		}
		REQUIRE((int)popcount(x) == count);
	}
}
TEST_CASE("popcount", "bit_ops") {
	popcountTest<int8_t>();
	popcountTest<int16_t>();
	popcountTest<int32_t>();
	popcountTest<int64_t>();
	popcountTest<uint8_t>();
	popcountTest<uint16_t>();
	popcountTest<uint32_t>();
	popcountTest<uint64_t>();
}

template<class T>
void msbTest() {
	int numBits = sizeof(T) * 8;
	T x = 0;
	REQUIRE(popcount(x) == 0);
	REQUIRE(msbIdx(x) == -1);
	for (int count = 1; count <= numBits; count++) {
		// just set ith bit
		x = 0;
		x = setBit(x, count-1);
		CAPTURE(numBits);
		REQUIRE((int)msbIdx(x) == count-1);

		// set `count` random bits
		CAPTURE(count);
		x = 0;
		auto idxs = ar::rand_ints(0, numBits-1, count);
		REQUIRE(idxs.size() == count);
		REQUIRE(ar::unique(idxs).size() == count); // no replacement
		ar::sort_inplace(idxs);
		for (auto idx : idxs) {
			x = setBit(x, idx);
		}
		if (popcount(x) != count) {
			dumpBits(x);
		}
		REQUIRE(popcount(x) == count);
		REQUIRE((int)msbIdx(x) == idxs[count-1]);
	}
}
TEST_CASE("msbIdx", "bit_ops") {
	msbTest<int8_t>();
	msbTest<int16_t>();
	msbTest<int32_t>();
	msbTest<int64_t>();
	msbTest<uint8_t>();
	msbTest<uint16_t>();
	msbTest<uint32_t>();
	msbTest<uint64_t>();
}

template<class T>
void lsbTest() {
	int numBits = sizeof(T) * 8;
	T x = 0;
	REQUIRE(popcount(x) == 0);
	REQUIRE(msbIdx(x) == -1);
	for (int count = 1; count <= numBits; count++) {
		CAPTURE(count);
		x = 0;
		auto idxs = ar::rand_ints(0, numBits-1, count);
		REQUIRE(idxs.size() == count);
		REQUIRE(ar::unique(idxs).size() == count); // no replacement
		ar::sort_inplace(idxs);
		for (auto idx : idxs) {
			x = setBit(x, idx);
		}
		REQUIRE(popcount(x) == count);
		REQUIRE((int)lsbIdx(x) == idxs[0]);
	}
}

TEST_CASE("lsbIdx", "bit_ops") {
	lsbTest<int8_t>();
	lsbTest<int16_t>();
	lsbTest<int32_t>();
	lsbTest<int64_t>();
	lsbTest<uint8_t>();
	lsbTest<uint16_t>();
	lsbTest<uint32_t>();
	lsbTest<uint64_t>();
}
