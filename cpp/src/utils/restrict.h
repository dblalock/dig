//
//  restrict.h
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __RESTRICT_H
#define __RESTRICT_H

// adapted from http://stackoverflow.com/a/5948101/1153180
#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#   define RESTRICT __restrict__
#elif defined(__clang__)
#	define RESTRICT __restrict__
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#   define RESTRICT __restrict
#else
#   define RESTRICT
#endif

#endif
