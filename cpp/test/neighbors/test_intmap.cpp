//
//  test_intmap.cpp
//  Dig
//
//  Created by DB on 2/14/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include "intmap.hpp"

#include <iostream>
#include "catch.hpp"

#include "array_utils.hpp"

//using namespace ar;


TEST_CASE("construct intmap", "intmap") {
	intmap64<double> m{};
}

TEST_CASE("isBitSet", "intmap") {
	uint64_t x = 0;
	// no bits set when x is 0
	for (int i = 0; i < 64; i++) {
		REQUIRE_FALSE(isBitSet(x, i));
	}
	// appropriate bit set when x nonzero; other bits not set
	for (int idx = 0; idx < 64; idx++) {
		x = oneAtIdx(idx);
		CAPTURE(x);
		CAPTURE(idx);
		REQUIRE(isBitSet(x, idx));
		for (uint8_t j = 0; j < 64; j++) {
			if (j == idx) {
				continue;
			}
			CAPTURE((int)j);
			REQUIRE_FALSE(isBitSet(x, j));
		}
	}
}

TEST_CASE("unsetBit", "intmap") {
	for (int idx = 0; idx < 64; idx++) {
		uint64_t x = oneAtIdx(idx);
		// set some other bits so it can't pass by always
		//  acting like x is just 0
		auto idxs = ar::rand_ints(0, 63, 3);
		for (auto idx : idxs) {
			x |= oneAtIdx(idx);
		}
		CAPTURE(x);
		CAPTURE(idx);
		REQUIRE(isBitSet(x, idx));
		uint64_t y = unsetBit(x, idx);
		REQUIRE_FALSE(isBitSet(y, idx));
	}
}

TEST_CASE("insert", "intmap") {
	intmap64<double> m{};
	
	m.put(0, 7.);
	REQUIRE(m.get(0) == 7.);
	
	vector<double> trueVals;
	for (int idx = 0; idx < 64; idx++) {
		trueVals.push_back(100. - idx);
	}
	for (int idx = 0; idx < 64; idx++) {
		m.put(idx, trueVals[idx]);
		for (int j = 0; j <= idx; j++) {
			REQUIRE(m.get(idx) == trueVals[idx]);
		}
	}
}

TEST_CASE("put+get, value", "intmap") {
	intmap64<double> m{};
	
	// knows it's empty
	for (int i = 0; i < 64; i++) {
		REQUIRE_FALSE(m.contains(i));
	}
	
	for (int i = 0; i < 64; i+=2) {
		m.put(i, i * 3);
	}
	// contains what we put in
	for (int i = 0; i < 64; i+=2) {
		REQUIRE(m.contains(i));
		REQUIRE(m.get(i) == i * 3);
	}
	// doesn't contain anything we didn't put in
	for (int i = 1; i < 64; i+=2) {
		REQUIRE_FALSE(m.contains(i));
	}
}

TEST_CASE("put+get, pointer", "intmap") {
	intmap64<std::unique_ptr<double>> m{};
	
	// knows it's empty
	for (int i = 0; i < 64; i++) {
		REQUIRE_FALSE(m.contains(i));
	}
	
	for (int i = 0; i < 64; i+=2) {
		m.put(i, std::unique_ptr<double>(new double(i * 3)));
	}
	// contains what we put in
	for (int i = 0; i < 64; i+=2) {
		REQUIRE(m.contains(i));
		REQUIRE(*m.get(i) == i * 3);
	}
	// doesn't contain anything we didn't put in
	for (int i = 1; i < 64; i+=2) {
		REQUIRE_FALSE(m.contains(i));
	}
}

TEST_CASE("put+get+erase, value", "intmap") {
	intmap64<float> m{};
	
	for (int i = 0; i < 64; i+=2) {
		m.put(i, i * 3.);
	}
	// contains what we put in
	for (int i = 0; i < 64; i+=2) {
		REQUIRE(m.contains(i));
		REQUIRE(m.get(i) == i * 3.);
	}
	for (int i = 0; i < 64; i++) {
		m.erase(i);
		REQUIRE_FALSE(m.contains(i));
	}
	// doesn't contain anything now
	for (int i = 1; i < 64; i++) {
		REQUIRE_FALSE(m.contains(i));
	}
	// haven't left it in a bad state or anything
	for (int i = 0; i < 64; i+=2) {
		m.put(i, i * 5);
		REQUIRE(m.contains(i));
		REQUIRE(m.get(i) == i * 5);
	}
}

TEST_CASE("put+get+erase, pointer", "intmap") {
	intmap64<std::unique_ptr<double>> m{};
	
	for (int i = 0; i < 64; i+=2) {
		m.put(i, std::unique_ptr<double>(new double(i * 3)));
	}
	// contains what we put in
	for (int i = 0; i < 64; i+=2) {
		REQUIRE(m.contains(i));
		REQUIRE(*m.get(i) == i * 3);
	}
	for (int i = 0; i < 64; i++) {
		m.erase(i);
		REQUIRE_FALSE(m.contains(i));
	}
	// doesn't contain anything now
	for (int i = 1; i < 64; i++) {
		REQUIRE_FALSE(m.contains(i));
	}
	// haven't left it in a bad state or anything
	for (int i = 0; i < 64; i+=2) {
		m.put(i, std::unique_ptr<double>(new double(i * 5)));
		REQUIRE(m.contains(i));
		REQUIRE(*m.get(i) == i * 5);
	}
}
