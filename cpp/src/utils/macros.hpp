//
//  macros.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __MACROS_HPP
#define __MACROS_HPP

// adapted from http://stackoverflow.com/a/5948101/1153180
#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
	#define RESTRICT __restrict__
#elif defined(__clang__)
	#define RESTRICT __restrict__
#elif defined(_MSC_VER) && _MSC_VER >= 1400
	#define RESTRICT __restrict
#else
	#define RESTRICT
#endif

#ifdef __cplusplus
	#include <type_traits>

	// put these in function bodies to statically assert that appropriate types
	// have been passed in as template params; prefer using the below type
	// constraint macros, however
	#define ASSERT_TRAIT(TRAIT, T, MSG) static_assert(std::TRAIT<T>::value, MSG)
	#define ASSERT_INTEGRAL(T) ASSERT_TRAIT(is_integral, T, "Type not integral!")

	// put these as extra template params to enforce constraints
	// on previous template params
	#define REQUIRE_TRAIT(TRAIT, T) \
		typename std::enable_if<std::TRAIT<T>::value, T>::type = 0
	#define REQUIRE_INT(T) REQUIRE_TRAIT(is_integral, T)
	#define REQUIRE_NUM(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_PRIMITIVE(T) REQUIRE_TRAIT(is_arithmetic, T)
#endif

#endif
