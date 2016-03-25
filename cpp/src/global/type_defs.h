//
//  type_defs.h
//  edtw
//
//  Created By <Anonymous> on 1/14/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#ifndef timekit_type_defs_h
#define timekit_type_defs_h

#include <sys/types.h>

typedef double data_t;
//typedef float data_t;
//typedef int	int_t;
//typedef short int short_int_t;
// typedef int32_t length_t;		//length of a time series in main memory; needs to be signed
typedef int32_t length_t;		//length of a time series in main memory; needs to be signed
typedef int16_t idx_t;			//indices in a time series in main memory; needs to be signed
typedef long tick_t;			// indices in an arbitrary-length time series; needs to be signed
typedef unsigned short int steps_t;
typedef unsigned int dims_t;
// typedef long long dist_calls_t;	///< number of times the innermost distance function has been called

extern const int kINVALID;
extern const int kSUCCESS;
extern const int kFAILURE;

extern const float kFLOATING_PT_ERROR;

// /** Struct of index + value at that index */
typedef struct Index {
	data_t value;
	tick_t index;
} Index;

#endif
