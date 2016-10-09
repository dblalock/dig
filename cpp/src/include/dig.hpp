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

//==================================================
// Testing 	//TODO remove once unit tests in place
//==================================================

int swigTest(int x);
double swigArrayTest(const double* x, int len);


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

#endif
