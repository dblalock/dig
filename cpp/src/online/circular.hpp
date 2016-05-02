

#ifndef _CIRCULAR_HPP
#define _CIRCULAR_HPP

#include <array>
#include <stdint.h>

// helped by http://accu.org/index.php/journals/389

template<class T, int Capacity=256, class Alloc=std::allocator<T> >
class circular_array {
public:

	// ------------------------------------------------
	// compile-time constants
	// ------------------------------------------------

	enum {
		version_major = 0,
		version_minor = 0
	};

	// required typedefs for STL compliance
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef typename Alloc::size_type size_type;
	typedef typename Alloc::reference reference;
	typedef typename Alloc::const_reference const_reference;
	// TODO {const,reverse, normal} iterator

	// other typdefs mirroring STL vector
	typedef Alloc allocator_type;
	typedef typename Alloc::difference_type difference_type;

	// misc typedefs
//	typedef circular_array<T, Alloc> self_type;
	typedef int32_t index_type;

	// default capacity
	enum { capacity = Capacity };

	// ------------------------------------------------
	// accessors
	// ------------------------------------------------

	pointer last() { return _ar.data() + _last_idx; }
	const_pointer last() const { return _ar.data() + _last_idx; }

	pointer end() const { return last() + 1; } // stl convention of 1 past end
	const_pointer end() const { return last() + 1; }

	size_type size() const { return _size; }
	pointer data() { return last() - _size + 1; }
	const_pointer data() const { return last() - _size + 1; }

	value_type back() const { return _ar[_last_idx]; }
	value_type front() const { return *(data()); }

	// ------------------------------------------------
	// other functions
	// ------------------------------------------------

	explicit circular_array(): _last_idx(Capacity-1), _size(0) {}

	void clear() {
		_last_idx = Capacity - 1;
		_size = 0;
	}

	void push_back(const value_type& item) {
		++_last_idx;
		if (_last_idx > lastPossibleIdx) {
			_last_idx -= Capacity;
		}
		_ar[_last_idx] = item;
		_ar[_last_idx - Capacity] = item;
		if (_size < Capacity) {
			++_size;
		}
	}

private:
	enum { lastPossibleIdx = 2 * Capacity - 1 };

	std::array<T, 2*Capacity> _ar; // double capacity so always contiguous
	index_type _last_idx;
	size_type _size;
	// index_type _first_idx;


};

#endif // _CIRCULAR_HPP
