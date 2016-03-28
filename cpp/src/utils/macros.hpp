//
//  macros.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __MACROS_HPP
#define __MACROS_HPP

// ------------------------ restrict keyword
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

// ------------------------ type traits macros
#ifdef __cplusplus
	#include <type_traits>

	// put these in function bodies to statically assert that appropriate types
	// have been passed in as template params; prefer using the below type
	// constraint macros, however
	#define ASSERT_TRAIT(TRAIT, T, MSG) static_assert(std::TRAIT<T>::value, MSG)
	#define ASSERT_INTEGRAL(T) ASSERT_TRAIT(is_integral, T, "Type not integral!")

	// put these as extra template params to enforce constraints
	// on previous template params; e.g.:
	//
	// template<class T, REQUIRE_INT(T)> T foo(T arg) { return arg + 1; }
	//
	#define REQUIRE_TRAIT(TRAIT, T) \
		typename = typename std::enable_if<std::TRAIT<T>::value, T>::type

	#define REQUIRE_NOT_TRAIT(TRAIT, T) \
		typename = typename std::enable_if<!std::TRAIT<T>::value, T>::type

	#define REQUIRE_IS_A(BASE, T) \
		typename = typename std::enable_if<std::is_base_of<BASE, T>::value, T>::type

	#define REQUIRE_IS_NOT_A(BASE, T) \
		typename = typename std::enable_if<!std::is_base_of<BASE, T>::value, T>::type

	#define REQUIRE_INT(T) REQUIRE_TRAIT(is_integral, T)
	#define REQUIRE_NUM(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_FLOAT(T) REQUIRE_TRAIT(is_floating_point, T)
	#define REQUIRE_PRIMITIVE(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_NOT_PTR(T) REQUIRE_NOT_TRAIT(is_pointer, T)
#endif // __cplusplus

#endif // __MACROS_HPP
