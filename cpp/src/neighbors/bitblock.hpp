//  bitblock.hpp
//
//  Dig
//
//  Created by DB on 5/1/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//


// #include <array>

#ifndef __BITARRAY_HPP
#define __BITARRAY_HPP

#include <assert.h>

#include "bit_ops.hpp" // TODO uncomment after test

// ------------------------ main class

template<class T>
class const_forward_iterator;
template<class T>
class const_backward_iterator;

// TODO force T to be an unsigned integral type of at most 8 bytes
template<class T>
class bitblock { // backed by a single integral type; eg, int32_t
public:
	// typedef T base_type;
	enum {
		kEndVal = -1,
		capacity_bytes = sizeof(T),
		capacity = 8 * sizeof(T)
	};

	T bits() 					{ return _bits; }
	inline uint8_t count() 		{ return popcount(_bits); }
	inline bool any() 			{ return T; } // (likely) faster than count == 0
	inline void print() const 	{ dumpBits(_bits); }
	inline void clear() 		{ T = 0; }

	inline void set(uint8_t idx) 		{ _bits = setBit(_bits, idx); }
	inline void unset(uint8_t idx) 		{ _bits = unsetBit(_bits, idx); }
	inline bool get(uint8_t idx) const 	{ return isBitSet(_bits, idx); }

	inline void firstOneAtOrAfter(uint8_t idx) const {
		return firstSetBitAtOrAfterIdx(_bits, idx);
	}
	inline int8_t firstOneAfter(uint8_t idx) const {
		return firstSetBitAfterIdx(_bits, idx);
	}
	inline int8_t lastOneBefore(uint8_t idx) const {
		return lastSetBitBeforeIdx(_bits, idx);
	}

private:
	T _bits;
};

// ------------------------ iterators

// notes:
// -these are not remotely STL compliant
// -they can only be iterated thru once
// -both will start returning bitblock<T>::kEndVal (which is -1) when there
// are no more 1s in their respective directions (forward or backward)
// -don't initialize them with indices < 0 if you're using pairs of them;
// the idx will get clipped at 0 and it will start here, so if both start
// at 0, you'll iterate over the same bit twice
// -initializing the forward iterator with an idx > the max possible idx is
// fine though; it will just clear out its whole (internal) bit vector and
// return -1 for next() immediately
// 	  -still, it's probably better to just check the idx bounds before
// 	  constructing instances of these iterators and not even construct them
// 	  and/or start to iterate thru them if the idx would be outside the bounds
//	-EDIT: actually, removed safety checks on idx; it's now undefined behvior
//  to pass in indices outside the range [0, 8*sizeof(T)-1]

template<class T>
class const_forward_iterator {
public:

	explicit const_forward_iterator(const bitblock<T>& b, int8_t idx) {
		// _idx = max( min(idx, kMaxIdx+1), 0); // idx must be in valid range
		_idx = idx; // idx must be in valid range
		// clear bits below idx (bit at idx is not cleared)
		_iter_bits = (b.bits() >> idx) << idx;
	}

	uint8_t next() {
		if (!_iter_bits) { return -1; };
		auto ret = lsbIdx(_iter_bits);
		_iter_bits = unsetBit(_iter_bits, ret);
		_idx++;
		return ret;
	}
};
private:
	enum { kMaxIdx = (8 * sizeof(T)) - 1 };
	T _iter_bits;
	int8_t _idx;
};

template<class T>
class const_backward_iterator {
public:
	enum { kEndVal = -1 };

	explicit const_backward_iterator(const bitblock<T>& b, int8_t idx) {
		// _idx = max( min(idx, kMaxIdx), 0); // idx must be in valid range
		_idx = idx;
		int8_t shft = kMaxIdx - idx;

		// clear bits above idx (bit at idx is not cleared)
		_iter_bits = (b.bits() << shft) >> shft;
	}

	uint8_t next() {
		if (!_iter_bits) { return -1; };
		auto ret = msbIdxNoCheck(_iter_bits);
		_iter_bits = unsetBit(_iter_bits, ret);
		_idx--;
		return ret;
	}
private:
	enum { kMaxIdx = (8 * sizeof(T)) - 1 };
	T _iter_bits;
	int8_t _idx;
};


#endif // include guard
