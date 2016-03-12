//
//  array_utils.hpp
//
//  Created By Davis Blalock on 1/14/14.
//  Copyright (c) 2014 Davis Blalock. All rights reserved.
//

#ifndef __ARRAY_UTILS_EIGEN_HPP
#define __ARRAY_UTILS_EIGEN_HPP


#include <algorithm>

using std::min;

namespace ar {

template<class F, class Derived1, class Derived2>
static inline auto copy(const F&& func, const EigenBase<Derived1>& in,
	EigenBase<Derived2>& out, size_t len=-1) {
	len = min(len, in.size())
	assert(len <= out.size())

	for (size_t i = 0; i < len; i++) {
		out(i) = in(i);
	}
}

// template<class F, class Derived1, class Derived2>
// static inline auto copy(const F&& func, const EigenBase<Derived1>& in,
// 	EigenBase<Derived2>& out, size_t len=-1) {
// 	len = min(len, in.size())
// 	assert(len <= out.size())

// 	for (size_t i = 0; i < len; i++) {
// 		out(i) = in(i);
// 	}
// }


} // namespace ar

#endif
