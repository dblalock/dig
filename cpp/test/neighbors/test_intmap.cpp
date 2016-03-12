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

TEST_CASE("popcount", "intmap") {
	uint64_t x = 0;
	REQUIRE(popcount(x) == 0);
	for (int count = 1; count < 64; count++) {
		x = 0;
		auto idxs = ar::rand_ints(0, 63, count);
		for (auto idx : idxs) {
//			std::cout << idx << ", ";
			x = setBit(x, idx);
		}
//		std::cout << std::endl;
//		dumpBits(x);
		REQUIRE(popcount(x) == count);
	}
}

TEST_CASE("msbIdx", "intmap") {
	uint64_t x = 0;
	REQUIRE(popcount(x) == 0);
	REQUIRE(msbIdx(x) == -1);
	for (int count = 1; count < 64; count++) {
		CAPTURE(count);
		x = 0;
		auto idxs = ar::rand_ints(0, 63, count);
		ar::sort_inplace(idxs);
		for (auto idx : idxs) {
			x = setBit(x, idx);
		}
		REQUIRE(popcount(x) == count);
		REQUIRE((int)msbIdx(x) == idxs[count-1]);
	}
}
TEST_CASE("lastKeyBefore", "intmap") {
	{
		intmap64<int64_t> m{};
		REQUIRE(m.lastKeyBefore(0) == -1);
		REQUIRE(m.lastKeyBefore(1) == -1);
		REQUIRE(m.lastKeyBefore(63) == -1);
	}
	
	for (int count = 1; count < 64; count++) {
		CAPTURE(count);
		
		intmap64<int64_t> m{};
		REQUIRE(m.size() == 0);

		auto idxs = ar::rand_ints(0, 63, count);
		ar::sort_inplace(idxs);
		for (auto idx : idxs) {
//			std::cout << idx << ", ";
			m.put(idx, 7);
		}
		
		REQUIRE(m.size() == count);
//		std::cout << std::endl << "occ:\t";
//		m._dumpOccupiedArray();
		
		for (int i = 1; i < count; i++) {
			REQUIRE(m.lastKeyBefore(idxs[i]) == idxs[i-1]);
		}
		CAPTURE(idxs[0]);
		
//		uint64_t lft = m._occupiedArray() << 64;
//		std::cout << "left:\t";
//		dumpBits(lft);
//		std::cout << "right:\t";
//		dumpBits(lft >> 64);
//		
//		std::cout << "shftd:\t";
//		uint8_t shft = 64 - idxs[0];
//		auto x = (m._occupiedArray() << shft) >> shft;
//		dumpBits(x);
		
		REQUIRE(m.lastKeyBefore(idxs[0]) == -1);
	}
}

TEST_CASE("firstKey[AtOr]After", "intmap") {
	{
		intmap64<int64_t> m{};
		REQUIRE(m.firstKeyAfter(0) == -1);
		REQUIRE(m.firstKeyAfter(1) == -1);
		REQUIRE(m.firstKeyAfter(63) == -1);
		REQUIRE(m.firstKeyAtOrAfter(0) == -1);
		REQUIRE(m.firstKeyAtOrAfter(1) == -1);
		REQUIRE(m.firstKeyAtOrAfter(63) == -1);
	}
	
	for (int count = 1; count < 64; count++) {
		CAPTURE(count);
		
		intmap64<int64_t> m{};
		REQUIRE(m.size() == 0);
		
		auto idxs = ar::rand_ints(0, 63, count);
		ar::sort_inplace(idxs);
		for (auto idx : idxs) {
			m.put(idx, 7);
		}
		REQUIRE(m.size() == count);
		
		for (int i = 0; i < count-1; i++) {
			REQUIRE(m.firstKeyAfter(idxs[i]) == idxs[i+1]);
		}
		for (int i = 0; i < count; i++) {
			REQUIRE(m.firstKeyAtOrAfter(idxs[i]) == idxs[i]);
		}
		CAPTURE(idxs[count-1]);
		REQUIRE(m.firstKeyAfter(idxs[count-1]) == -1);
		REQUIRE(m.firstKeyAtOrAfter(idxs[count-1]) == idxs[count-1]);
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

TEST_CASE("put+get, uniq_ptr", "intmap") {
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

TEST_CASE("put+get+erase, uniq_ptr", "intmap") {
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
