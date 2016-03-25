//
//  global_dist_calls.c
//  edtw
//
//  Created By <Anonymous> on 2/11/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#include "type_defs.h"

typedef long dist_calls_t;
dist_calls_t global_dist_calls = 0;

double gAbandonAbovePruningPower = 10.0;

long optimal_dist_time = 0;
long inherit_time = 0;
long shift_time = 0;
long invalid_time = 0;
