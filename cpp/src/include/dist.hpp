//
//  Dist.h
//  Dig
//
//  Created by DB on 10/2/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef Dig_Dist_h
#define Dig_Dist_h

//==================================================
// Constants
//==================================================

typedef enum DistanceMeasure {
	EUCLIDEAN = 0,
	ED = EUCLIDEAN,
	DYNAMIC_TIME_WARPING = 1,
	DTW = DYNAMIC_TIME_WARPING,
	UNIFORM_SCALING = 2,
	US = UNIFORM_SCALING,
	SCALED_WARPED_MATCHING = 3,
	SWM = SCALED_WARPED_MATCHING
} DistanceMeasure;

//in general, automatically pick distance measure based on these
typedef struct DistanceMeasureParams {
	float timeWarping;
	float timeScaling;

	DistanceMeasureParams(float warp=0, float scaling=0) {
		timeWarping = warp;
		timeScaling = scaling;
	}
} DistanceMeasureParams;

typedef enum SubseqReportStrategy {
	AGGRESSIVE,
	MODERATE,
	CAUTIOUS
} SubseqReportStrategy;

//==================================================
// Distance Measures
//==================================================

// ------------------------------- L1 distance

int dist_L1(const int* v1, const int* v2, int n);
int dist_L1(const int* v1, int m, const int* v2, int n);
double dist_L1(const double* v1, const double* v2, int n);
double dist_L1(const double* v1, int m, const double* v2, int n);

// ------------------------------- L2 distance

int dist_L2(const int* v1, const int* v2, int n);
int dist_L2(const int* v1, int m, const int* v2, int n);
double dist_L2(const double* v1, const double* v2, int n);
double dist_L2(const double* v1, int m, const double* v2, int n);

// ------------------------------- DTW distance

int dist_dtw(const int* v1, const int* v2, int n, int r);
int dist_dtw(const int* v1, int m, const int* v2, int n, int r);
double dist_dtw(const double* v1, const double* v2, int n, int r);
double dist_dtw(const double* v1, int m, const double* v2, int n, int r);

// ------------------------------- Uniform Scaling distance

// double dist_scaling(const int* v1, const int* v2, int n, int r);
// double dist_scaling(const int* v1, int m, const int* v2, int n, int r);
// double dist_scaling(const double* v1, const double* v2, int n, int r);
// double dist_scaling(const double* v1, int m, const double* v2, int n, int r);

// ------------------------------- distance function wrapper

// // top-level function that computes distance with any amount of warping and time scaling
// int dist(const int* v1, const int* v2, int n, DistanceMeasureParams p);
// int dist(const int* v1, int m, const int* v2, int n, DistanceMeasureParams p);
// double dist(const double* v1, const double* v2, int n, DistanceMeasureParams p);
// double dist(const double* v1, int m, const double* v2, int n, DistanceMeasureParams p);



#endif //
