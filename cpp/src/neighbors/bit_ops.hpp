//  bit_ops.hpp
//
//  Dig
//
//  Created by DB on 4/29/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#ifndef __BIT_OPS_HPP
#define __BIT_OPS_HPP

#include <assert.h>
#include <stdint.h>
// #include <type_traits> // just for static_assert
#include <iostream>
#include <string.h> // for ffs()

namespace { // anonymous namespace

// ------------------------ debugging

template<class T>
inline void dumpBits(T x) {
	for (int i = 0; i < sizeof(x); i++) {
		std::cout << " ";
		for (int j = 0; j < 8; j++) {
			uint64_t mask = ((uint64_t)1) << (8*i + j);
			uint64_t masked = mask & x;
			std::cout << (bool)masked;
		}
	}
	std::cout << std::endl;
}

// ------------------------ popcount

template<class T, int N>
struct _popcount;

template<class T>
struct _popcount<T, 1> {
	static int8_t count(T x) { return __builtin_popcount((uint8_t)x); }
};
template<class T>
struct _popcount<T, 2> {
	static int8_t count(T x) { return __builtin_popcount((uint16_t)x); }
};
template<class T>
struct _popcount<T, 4> {
	static int8_t count(T x) { return __builtin_popcountl((uint32_t)x); }
};
template<class T>
struct _popcount<T, 8> {
	static int8_t count(T x) { return __builtin_popcountll((uint64_t)x); }
};

template<class T>
int8_t popcount(T x) {
	return _popcount<T, sizeof(T)>::count(x);
}

// template<class T, REQUIRE_SIZE(1, T)>
// inline int8_t popcount(T x) {
// 	return __builtin_popcount(x);
// }
// template<class T, REQUIRE_SIZE(2, T)>
// inline int8_t popcount(T x) {
// 	return __builtin_popcount(x);
// }
// template<class T, REQUIRE_SIZE(4, T)>
// inline int8_t popcount(T x) {
// 	return __builtin_popcountl(x);
// }
// template<class T, REQUIRE_SIZE(8, T)>
// inline int8_t popcount(T x) {
// 	return __builtin_popcountll(x);
// }

// template<class T>
// inline int8_t popcount(T x) {
// //	static_assert(false, "not specialized for provided type!");
// 	std::cout << "ERROR: called invalid popcount impl!\n";
// 	return -99; // TODO better err catching
// }

// template<>
// inline int8_t popcount(uint8_t x) {
// template<class T, REQUIRE_SIZE(1, T)>
// inline int8_t popcount(T x) {
// 	return __builtin_popcount(x); // uncomment below if func not availble

	// relies on 64-bit instructions; see
	// https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSet64
	// return (x * 0x200040008001ULL & 0x111111111111111ULL) % 0xf;

	// alternative that doesn't rely on 64-bit instructions
	// const uint8_t ONES_8 = ~((uint8_t)0);
	// const uint8_t M2 = ~(ONES_8 << 4); // 00001111
	// const uint8_t M1 = M2 ^ (M2 << 2); // 00110011
	// const uint8_t M0 = M1 ^ (M1 << 1); // 01010101

	// x = ((x >> 1) & M0) + (x & M0); // right term to avoid overflow
	// x = ((x >> 2) & M1) + (x & M1); // right term to avoid overflow
	// return ((x >> 4) + x) & M2;
// }
// template<>
// inline int8_t popcount(int8_t x) {
// 	return __builtin_popcount(x); // uncomment below if func not availble
// }
// template<>
// inline int8_t popcount(uint64_t x) {
// 	return __builtin_popcountll(x); // uncomment below if func not available

// 	// const uint64_t ONES_64 = ~((uint64_t)0);
// 	// const uint64_t M5 = ~(ONES_64 << 32); // second half of bits 1
// 	// const uint64_t M4 = M5 ^ (M5 << 16);
// 	// const uint64_t M3 = M4 ^ (M4 << 8);
// 	// const uint64_t M2 = M3 ^ (M3 << 4);
// 	// const uint64_t M1 = M2 ^ (M2 << 2);
// 	// const uint64_t M0 = M1 ^ (M1 << 1); // every other bit 1

// 	// x = ((x >> 1) & M0) + (x & M0); // right term to avoid overflow
// 	// x = ((x >> 2) & M1) + (x & M1); // right term to avoid overflow
// 	// x = ((x >> 4) + x) & M2;
// 	// x = ((x >> 8) + x) & M3;
// 	// x = ((x >> 16) + x) & M4;
// 	// return ((x >> 32) + x) & M5;
// }
// template<>
// inline int8_t popcount(int64_t x) {
// 	return __builtin_popcountll(x); // uncomment below if func not availble
// }

// ------------------------ msbIdx

// note that we have to use sizeof() for the types that __builtin_clz*
// operates on, not the types we pass in, when inferring where the msb is;
// the reason is that the values we pass in will get padded with zero bytes
// as necessary, which will inflate the leading zero counts (which is what
// __builtin_clz() returns)

template<class T, int N>
struct _msbIdx;

template<class T>
struct _msbIdx<T, 1> {
	static int8_t func(T x) {
		return 8*sizeof(unsigned int) - 1 - __builtin_clz((uint8_t)x);
	}
};
template<class T>
struct _msbIdx<T, 2> {
	static int8_t func(T x) {
		return 8*sizeof(unsigned int) - 1 - __builtin_clz((uint16_t)x);
	}
};
template<class T>
struct _msbIdx<T, 4> {
	static int8_t func(T x) {
		return 8*sizeof(unsigned long) - 1 - __builtin_clzl((uint32_t)x);
	}
};
template<class T>
struct _msbIdx<T, 8> {
	static int8_t func(T x) {
		return 8*sizeof(unsigned long long) - 1 - __builtin_clzll((uint64_t)x);
	}
};

template<class T>
int8_t msbIdxNoCheck(T x) {
	return _msbIdx<T, sizeof(T)>::func(x);
}

// template<class T>
// inline int8_t msbIdxNoCheck(T x) {
// 	//	static_assert(false, "not specialized for provided type!");
// 	return -99; // TODO better err catching
// }

// template<>
// inline int8_t msbIdxNoCheck(uint8_t x) {
// 	int y = x;
// 	return (8 * sizeof(x) - 1) - __builtin_clz(y);
// }
// template<>
// inline int8_t msbIdxNoCheck(uint32_t x) {
// 	return (8 * sizeof(x) - 1) - __builtin_clzl(x);

// 	// set all bits below MSB
// 	// x |= x >> 1;
// 	// x |= x >> 2;
// 	// x |= x >> 4;
// 	// // count how many bits are set
// 	// return popcount(x) - 1;
// }
// template<>
// inline int8_t msbIdxNoCheck(uint64_t x) {
// 	return (8 * sizeof(x) - 1) - __builtin_clzll(x);

// 	// set all bits below MSB
// 	// x |= x >> 1;
// 	// x |= x >> 2;
// 	// x |= x >> 4;
// 	// x |= x >> 8;
// 	// x |= x >> 16;
// 	// x |= x >> 32;
// 	// // count how many bits are set
// 	// return popcount(x) - 1;
// }


template<class T>
inline int8_t msbIdx(T x) {
	if (!x) { // no more nonzeros after idx
		return -1;
	}
	return msbIdxNoCheck(x);
}

// ------------------------ lsbIdx

template<class T, int N>
struct _lsbIdx;

template<class T>
struct _lsbIdx<T, 1> {
	static int8_t func(T x) { return ffs(x) - 1; }
};
template<class T>
struct _lsbIdx<T, 2> {
	static int8_t func(T x) { return ffs(x) - 1; }
};
template<class T>
struct _lsbIdx<T, 4> {
	static int8_t func(T x) { return ffsl(x) - 1; }
};
template<class T>
struct _lsbIdx<T, 8> {
	static int8_t func(T x) { return ffsll(x) - 1; }
};

template<class T>
int8_t lsbIdx(T x) {
	return _lsbIdx<T, sizeof(T)>::func(x);
}

// ------------------------ misc

template<class T>
inline T zeroAllButLeastSignificantOne(T x) {
	return x & (-x);
}

template<class T>
inline T unsetMSB(T x) {
	return unsetBit(x, msbIdxNoCheck(x));
}
template<class T>
inline T unsetLSB(T x) {
	T lsbMask = zeroAllButLeastSignificantOne(x);
	return x & ~lsbMask;
}

template<class T>
inline T oneAtIdx(uint8_t idx) {
	return ((T)1) << idx; // *must* treat 1 as the appropriate type
}

template<class T>
inline bool isBitSet(T x, uint8_t idx) {
	T mask = oneAtIdx<T>(idx);
	return (x & mask);
}

template<class T>
inline T setBit(T x, uint8_t idx) {
	return x | oneAtIdx<T>(idx);
}

template<class T>
inline T unsetBit(T x, uint8_t idx) {
	T bit = oneAtIdx<T>(idx);
	return x & (~bit);
}

template<class T>
inline int8_t lastSetBitBeforeIdx(T x, uint8_t idx) {
	assert(idx < 8*sizeof(T));
	if (x == 0 || idx == 0) {
		return -1;
	}
	T shft = 64 - idx;
	x = (x << shft) >> shft; // zero bits above idx, including idx

	return msbIdx(x);
}

template<class T>
inline int8_t firstSetBitAtOrAfterIdx(T x, uint8_t idx) {
	assert(idx < 8*sizeof(T));
	x = (x >> idx) << idx; // zero bits below idx, not including idx
	return lsbIdx(x);
//	x = zeroAllButLeastSignificantOne(x);
//	return msbIdx(x);
}
template<class T>
inline int8_t firstSetBitAfterIdx(T x, uint8_t idx) {
	assert(idx < 8*sizeof(T));
	if (x == 0 || idx == (8*sizeof(T)-1)) {
		return -1;
	}
	x = (x >> (idx+1)) << (idx+1); // zero bits below idx, including idx
	return lsbIdx(x);
//	x = zeroAllButLeastSignificantOne(x);
//	return msbIdx(x);
	// return msbIdxNoCheck(x);
//	return idxOfNonzeroBit(x)
}

} // anonymous namespace
#endif // include guard
