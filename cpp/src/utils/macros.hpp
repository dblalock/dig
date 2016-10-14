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

// count the number of arguments in a varargs list
#define VA_NUM_ARGS(...) _VA_NUM_ARGS_IMPL(__VA_ARGS__, \
	16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _VA_NUM_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, \
	 _13, _14, _15, _16, N, ...) N

#ifdef __cplusplus
// ------------------------ type traits macros
	#include <type_traits>
	// #include "hana.hpp"

	// #define SELF_TYPE \
	// 	typename std::remove_reference<decltype(*this)>::type

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

	// ------------------------ require that some constexpr be true

	#define REQ(EXPR) \
		typename = typename std::enable_if<EXPR, void>::type

	// have to wrap EXPR in a local template param for enable_if to work on
	// a class method where EXPR is a class template param
	#define _METHOD_REQ(EXPR, NAME) \
		bool NAME = EXPR, typename = typename std::enable_if<NAME, void>::type

	#define METHOD_REQ0(EXPR) \
		_METHOD_REQ(EXPR, __expr0__)

	#define METHOD_REQ1(EXPR) \
		_METHOD_REQ(EXPR, __expr1__)

	#define METHOD_REQ2(EXPR) \
		_METHOD_REQ(EXPR, __expr2__)

	#define METHOD_REQ3(EXPR) \
		_METHOD_REQ3(EXPR, __expr3__)

	#define METHOD_REQ4(EXPR) \
		_METHOD_REQ(EXPR, __expr4__)

	#define METHOD_REQ5(EXPR) \
		_METHOD_REQ(EXPR, __expr5__)

	#define METHOD_REQ6(EXPR) \
		_METHOD_REQ(EXPR, __expr6__)

	#define METHOD_REQ(EXPR) \
		_METHOD_REQ(EXPR, __expr__)

		// bool __expr__ = EXPR, typename = typename std::enable_if<__expr__, void>::type

		// typename = typename std::enable_if<EXPR, T>::type
	#define REQUIRE_TRAIT(TRAIT, T) \
		typename = typename std::enable_if<std::TRAIT<T>::value, T>::type

	#define REQUIRE_NOT_TRAIT(TRAIT, T) \
		typename = typename std::enable_if<!std::TRAIT<T>::value, T>::type

	#define REQUIRE_IS_A(BASE, T) \
		typename = typename std::enable_if<std::is_base_of<BASE, T>::value, T>::type

	#define REQUIRE_IS_NOT_A(BASE, T) \
		typename = typename std::enable_if<!std::is_base_of<BASE, T>::value, T>::type

		// REQ(!std::is_base_of<BASE, T>::value)

	#define REQUIRE_INT(T) REQUIRE_TRAIT(is_integral, T)
	#define REQUIRE_NUM(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_FLOAT(T) REQUIRE_TRAIT(is_floating_point, T)
	#define REQUIRE_PRIMITIVE(T) REQUIRE_TRAIT(is_arithmetic, T)
	#define REQUIRE_NOT_PTR(T) REQUIRE_NOT_TRAIT(is_pointer, T)

	// ------------------------ is_valid; requires C++14
	//  inspired by https://gist.github.com/Jiwan/7a586c739a30dd90d259

	template <typename T> struct _valid_helper {
	private:
	    template <typename Param> constexpr auto _is_valid(int _)
		    // type returned by decltype is last type in the list (here,
			// std::true_type), but previous types must be valid
	    	-> decltype(std::declval<T>()(std::declval<Param>()),
	    		std::true_type())
	    {
	        return std::true_type();
	    }

	    template <typename Param> constexpr std::false_type _is_valid(...) {
	        return std::false_type();
	    }

	public:
	    template <typename Param> constexpr auto operator()(const Param& p) {
	        // The argument is forwarded to one of the two overloads.
	        // The SFINAE on the 'true_type' will come into play to dispatch.
	        return _is_valid<Param>(int(0));
	    }
	};

	template <typename T> constexpr auto is_valid(const T& t) {
	    return _valid_helper<T>();
	}

	// #define IS_VALID(EXPR)) \
	// 	hana::is_valid([](auto&& x) -> decltype(EXPR) { })

	#define CREATE_TEST(OBJNAME, EXPR) \
		is_valid([](auto&& OBJNAME) -> decltype(EXPR) { })

	#define CREATE_TEST_X(EXPR) \
		is_valid([](auto&& x) -> decltype(EXPR) { })

	#define TEST_FOR_METHOD(INVOCATION) \
		is_valid([](auto&& x) -> decltype(x. INVOCATION) { })

	#define PASSES_TEST(OBJ, TEST) \
		decltype(TEST(OBJ))::value

	#define TYPE_PASSES_TEST(T, TEST) \
		decltype(TEST(std::declval<T>()))::value

	#define REQ_TYPE_PASSES(T, TEST) \
		REQ(TYPE_PASSES_TEST(T, TEST))

	#define ENABLE_IF(EXPR, T) \
		typename std::enable_if<EXPR, T>::type


// ------------------------ TYPES(...) convenience macro for template args
#define TYPES_1(A) template<typename A>
#define TYPES_2(A, B) \
	template<typename A, typename B>
#define TYPES_3(A, B, C) \
	template<typename A, typename B, typename C>
		// TYPES_10(A, B, C, D, E, F=int, G=int, H=int, I=int, J=int)
#define TYPES_4(A, B, C, D) \
	template<typename A, typename B, typename C, typename D>
		// TYPES_10(A, B, C, D, E=int, F=int, G=int, H=int, I=int, J=int)
#define TYPES_5(A, B, C, D, E) \
	template<typename A, typename B, typename C, typename D, typename E>
		// TYPES_10(A, B, C, D, E, F=int, G=int, H=int, I=int, J=int)
#define TYPES_6(A, B, C, D, E, F) \
	template<typename A, typename B, typename C, typename D, typename E, \
		typename F>
		// TYPES_10(A, B, C, D, E, F, G=int, H=int, I=int, J=int)
#define TYPES_7(A, B, C, D, E, F, G) \
	template<typename A, typename B, typename C, typename D, typename E, \
		typename F, typename G>
		// TYPES_10(A, B, C, D, E, F, G, H=int, I=int, J=int)
#define TYPES_8(A, B, C, D, E, F, G, H) \
	template<typename A, typename B, typename C, typename D, typename E, \
		typename F, typename G, typename H>
		// TYPES_10(A, B, C, D, E, F, G, H, I=int, J=int)
#define TYPES_9(A, B, C, D, E, F, G, H, I) \
	template<typename A, typename B, typename C, typename D, typename E, \
		typename F, typename G, typename H, typename I>
		// TYPES_10(A, B, C, D, E, F, G, H, I, J=int)
#define TYPES_10(A, B, C, D, E, F, G, H, I, J) \
	template<typename A, typename B, typename C, typename D, typename E, \
		typename F, typename G, typename H, typename I, typename J

// define a top level TYPES macro that automatically calls the macro with
// the proper number of arguments
// #define _TYPES_IMPL2(count, ...) TYPES_ ## count (__VA_ARGS__)
// #define _TYPES_IMPL(count, ...) _TYPES_IMPL2(count, __VA_ARGS__)
// #define TYPES(...) _TYPES_IMPL(VA_NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

// TODO if this works, get rid of above 3 loc and move outside of cpp block
#define _WRAP_VARIADIC_IMPL2(name, count, ...) \
		name ## count (__VA_ARGS__)
#define _WRAP_VARIADIC_IMPL(name, count, ...) \
		_WRAP_VARIADIC_IMPL2(name, count, __VA_ARGS__)
#define WRAP_VARIADIC(name, ...) \
		_WRAP_VARIADIC_IMPL(name, VA_NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

#define TYPES(...) WRAP_VARIADIC(TYPES, __VA_ARGS__)

// #undef _TYPES_IMPL
// #undef _TYPES_IMPL2
#undef _WRAP_VARIADIC_IMPL
#undef _WRAP_VARIADIC_IMPL2
#undef VA_NUM_ARGS
#undef _VA_NUM_ARGS_IMPL

// ------------------------ static size assertions from Eigen

#define _PREDICATE_SAME_MATRIX_SIZE(TYPE0,TYPE1) \
    ( \
        (int(TYPE0::SizeAtCompileTime)==0 && int(TYPE1::SizeAtCompileTime)==0) \
    || (\
          (int(TYPE0::RowsAtCompileTime)==Eigen::Dynamic \
        || int(TYPE1::RowsAtCompileTime)==Eigen::Dynamic \
        || int(TYPE0::RowsAtCompileTime)==int(TYPE1::RowsAtCompileTime)) \
      &&  (int(TYPE0::ColsAtCompileTime)==Eigen::Dynamic \
        || int(TYPE1::ColsAtCompileTime)==Eigen::Dynamic \
        || int(TYPE0::ColsAtCompileTime)==int(TYPE1::ColsAtCompileTime))\
       ) \
    )

#define STATIC_ASSERT_SAME_SHAPE(TYPE0,TYPE1) \
	static_assert(_PREDICATE_SAME_MATRIX_SIZE(TYPE0,TYPE1), \
		YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES)

#undef _PREDICATE_SAME_MATRIX_SIZE


#define PRINT_STATIC_TYPE(X) \
	static_assert(decltype(X)::__debug__, #X);

#endif // __cplusplus




#endif // __MACROS_HPP
