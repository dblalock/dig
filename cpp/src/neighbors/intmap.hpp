//  intmap.hpp
//
//  Dig
//
//  Created by DB on 1/24/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include <assert.h>
#include <array>


//static constexpr uint8_t UINTS = {0,1,2,3,4,5,6,7,8,9,
//								 10,11,12,13,14,15,16,17,18,19,
//								 20,21,22,23,24,25,26,27,28,29,
//								 30,31,32,33,34,35,36,37,38,39,
//								 40,41,42,43,44,45,46,47,48,49,
//								 50,51,52,53,54,55,56,57,58,59,
//								 60,61,62,63};


// see http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2010/index.htm#
//const uint64_t deBruijn = 0x022fdd63cc95386d;
//const int convert[64] = {
//	0, 1, 2,53, 3, 7, 54, 27,
//	4, 38, 41,  8, 34, 55, 48, 28,
//	62,  5, 39, 46, 44, 42, 22,  9,
//	24, 35, 59, 56, 49, 18, 29, 11,
//	63, 52,  6, 26, 37, 40, 33, 47,
//	61, 45, 43, 21, 23, 58, 17, 10,
//	51, 25, 36, 32, 60, 20, 57, 16,
//	50, 31, 19, 15, 30, 14, 13, 12
//};
//static inline uint8_t idxOfNonzeroBit(uint64_t x) {
//	return convert[(x * deBruijn) >> 58];
//}

// constants for popcount
static const uint64_t M5 = ~(((uint64_t)-1) << 32); // first half of bits 1
static const uint64_t M4 = M5 ^ (M5 << 16);
static const uint64_t M3 = M4 ^ (M4 << 8);
static const uint64_t M2 = M3 ^ (M3 << 4);
static const uint64_t M1 = M2 ^ (M2 << 2);
static const uint64_t M0 = M1 ^ (M1 << 1); // every other bit 1

static inline uint8_t popcount(uint64_t x) {
	x = ((x >> 1) & M0) + (x & M0); // right term to avoid overflow
	x = ((x >> 2) & M1) + (x & M1); // right term to avoid overflow
	x = ((x >> 4) + x) & M2;
	x = ((x >> 8) + x) & M3;
	x = ((x >> 16) + x) & M4;
	return ((x >> 32) + x) & M5;
}

static inline uint64_t zeroAllButLeastSignificantOne(uint64_t x) {
	return x & (-x);
}

static inline uint8_t msbIdx(uint64_t x) {
	// set all bits below MSB
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
	// count how many bits are set
	return popcount(x);
}

static inline uint8_t lastSetBitBeforeIdx(uint64_t x, uint8_t idx) {
	assert(idx < 64);
	int8_t shft = 64 - idx;
	x = (x << shft) >> shft; // zero bits above idx, including idx
	if (!x) {
		return -1;
	}
	return msbIdx(idx);
}

static inline int8_t firstSetBitAfterIdx(uint64_t x, uint8_t idx) {
	assert(idx < 64);
	x = (x >> (idx+1)) << (idx+1); // zero bits below idx, including idx
	if (!x) { // no more nonzeros after idx
		return -1;
	}
	x = zeroAllButLeastSignificantOne(x);
	return msbIdx(x);
//	return idxOfNonzeroBit(x)
}

static inline int8_t firstSetBitAtOrAfterIdx(uint64_t x, uint8_t idx) {
	assert(idx < 64);
	x = (x >> idx) << idx; // zero bits below idx, not including idx
	if (!x) { // no more nonzeros after idx
		return -1;
	}
	x = zeroAllButLeastSignificantOne(x);
	return msbIdx(x);
}

static inline uint64_t oneAtIdx(uint8_t idx) {
	return ((uint64_t)1) << idx; // *must* treat 1 as a 64-bit type
}

static inline bool isBitSet(uint64_t x, uint8_t idx) {
	uint64_t mask = oneAtIdx(idx);
	return (x & mask) > 0;
}

static inline uint64_t unsetBit(uint64_t x, uint8_t idx) {
	uint64_t bit = oneAtIdx(idx);
	return x & (~bit);
}

// specialized map for storing objs with uint8 keys ranging from 0 to 63 ->
template<class V>
class intmap64 {
private:
	std::array<V, 64> _ar;
	uint64_t _occupied;

public:
	intmap64(): _ar(), _occupied(0) {}

	// basic map functionality
	inline V& get(uint8_t key) {
		return _ar[key];
	}
	inline void put(uint8_t key, V value) {
		std::swap(_ar[key], value);
		_occupied |= oneAtIdx(key);
	}
	inline void erase(uint8_t key) {
		V val;
		std::swap(_ar[key], val);
		_occupied = unsetBit(_occupied, key);
	}
	inline bool contains(uint8_t key) {
		return isBitSet(_occupied, key);
	}

	// determining next keys to look at
	inline int8_t firstKeyAtOrAfter(uint8_t key) {
		return firstSetBitAtOrAfterIdx(_occupied, key);
	}
	inline int8_t firstKeyAfter(uint8_t key) {
		return firstSetBitAfterIdx(_occupied, key);
	}
	inline int8_t lastKeyBefore(uint8_t key) {
		return lastSetBitBeforeIdx(_occupied, key);
	}
};
