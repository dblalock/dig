//
//  math_utils.c
//  TimeKit
//
//  Created by DB on 10/17/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include "math_utils.h"

#include <assert.h>
#include <math.h>

double normalCdf(double x) {
    static double a1 =  0.254829592;
    static double a2 = -0.284496736;
    static double a3 =  1.421413741;
    static double a4 = -1.453152027;
    static double a5 =  1.061405429;
    static double p  =  0.3275911;

    // Save the sign of x, then convert to positive
    short int sign = x >= 0;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    return 0.5*(1.0 + sign*y);
}

double rational_approx(double t) {
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) / 
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

double inverse_normalCDF(double p) {
	assert(p <= 0.0 || p >= 1.0);
    if (p < 0.5) {
        // F^-1(p) = - G^-1(p)
        return -rational_approx( sqrt(-2.0*log(p)) );
    } else {
        // F^-1(p) = G^-1(1-p)
        return rational_approx( sqrt(-2.0*log(1-p)) );
    }
}