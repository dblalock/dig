//
//  Dig.h
//  Dig
//
//  Created by DB on 10/2/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#ifndef Dig_Dig_h
#define Dig_Dig_h

#include <memory>	// for unique_ptr

// #ifdef __cplusplus
// extern "C" {
// #endif

//==================================================
// Testing 	//TODO remove once unit tests in place
//==================================================

int swigTest(int x);
double swigArrayTest(const double* x, int len);

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

//==================================================
// Classification
//==================================================

typedef enum ClassificationAlgorithm {
	NN_L1,
	NN_L2,
	NN_DTW,
	LOGICAL_SHAPELET,
	// LEARNED_SHAPELET, 	//TODO
	AdjustedConfidence,
} ClassificationAlgorithm;

class TSClassifier {
private:
	class impl;
	std::unique_ptr<impl> _pimpl;
public:
	TSClassifier(ClassificationAlgorithm algo=NN_L2);
	~TSClassifier();

	void setAlgorithm(ClassificationAlgorithm algo);
	void addExample(const double* X, int m, int n, int label);
	int classify(const double* X, int m, int n);
};

//==================================================
// Discretization		//TODO most of this should be internal
//==================================================

//TODO we really want a more general TS interface that has
//arbitrary cardinality (including float32/float64) at different
//positions, as well as, ideally, variable-length characters
	//prolly just an OrderedReals class

class OrderedReals {	//maybe have this take 'd','i', etc in ctor to specify type
public:
	virtual ~OrderedReals() = 0;
	virtual std::shared_ptr<double> getRawData() = 0;	//TODO don't mandate double by templating and instantiating template
	virtual int getBitsForPosition(int idx) = 0;
	virtual double distanceTo(const OrderedReals& other, double thresh=-1) = 0;		//TODO specify distance measure:
};

// class SAXWord: public OrderedReals {
// private:
// 	class impl;
// 	std::unique_ptr<impl> _pimpl;
// public:
// 	SAXWord(const double* v, int len, int wordLen, int bits);
// 	std::shared_ptr<double> getRawData();
// 	int getWordLen();
// 	int getBits();	//TODO might vary by position--how should we deal with this?
// 	int getBitsForPosition(int idx);
// 	double distanceTo(const OrderedReals& other, double thresh=-1);
// };

//TODO a bunch of template functions that compute dists between
//SAX chars xor quantized chars based on each of their lengths
//and cardinalities

//==================================================
// Search
//==================================================

typedef struct SubsequenceLocation {
	int start;
	int end;
	double dist;
} SubsequenceLocation;

// ------------------------------- int
// SubseqLocation findNearestNeighbor(const int* query, int qLen,
// 	int* x, int xLen, double thresh=-1);
// std::vector<SubseqLocation> findKNearestNeighbors(const int* query, int qLen,
// 	int* x, int xLen, int k, double thresh=-1);

// ------------------------------- double
// SubseqLocation findNearestNeighbor(const double* query, int qLen,
// 	double* x, int xLen, double thresh=-1);
// std::vector<SubseqLocation> findKNearestNeighbors(const double* query, int qLen,
// 	double* x, int xLen, int k, double thresh=-1);

//==================================================
// Data pipelining
//==================================================

//so I have no idea how to make this compile...
	//pretty sure this top-level class needs to not be templated
// template <class INPUT, class OUTPUT>
// class Operation<INPUT,OUTPUT> {	//TODO also int types
// private:
// 	class impl;
// 	std::unique_ptr<impl> _pimpl;
// public:
// 	bool addReceiver(const Operation& other);
// 	//subclasses need to implement this, but we don't want to specify
// 	//the type here because we want the compiler to tell us if what
// 	//this spits out and what the other one takes in aren't the same
// 	void receive(INPUT input);
// 	void broadcast();
// };

// #ifdef __cplusplus
// }
// #endif

#endif
