//
//  Flock.hpp
//  Dig
//
//  Created by DB on 3/9/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef DIG_FLOCK_HPP
#define DIG_FLOCK_HPP

#include <memory>

#include "Dense" // just for ArrayXd...

// #include "pimpl.hpp"
// #include "pimpl_impl.hpp" // include in header for SWIG; TODO does this fix it?

#include "shape_features.hpp"


// using Eigen::MatrixXd;
// using Eigen::ArrayXXd;
using Eigen::Dynamic;
using Eigen::ColMajor;

typedef Eigen::Array<double, Dynamic, Dynamic, ColMajor> FArray;

// duplicate typedefs here (not just in shape_features) to make SWIG happy
// #include <vector>
// using std::vector;
// #include "type_defs.h"
// using Eigen::Matrix;
// using Eigen::MatrixXd;
// using Eigen::Dynamic;
// using Eigen::RowMajor;
// typedef Matrix<double, Dynamic, Dynamic, RowMajor> CMatrix;
// // typedef Matrix<double, Dynamic, Dynamic> FMatrix;
// typedef MatrixXd FMatrix;

vector<double> dotProdsForSeed(const FMatrix& Phi, const FMatrix& Phi_blur,
	int seed, int windowLen);
vector<length_t> candidatesFromDotProds(const vector<double>& dotProds,
	int Lmin);
// vector<length_t> selectFromCandidates(const FMatrix& Phi,
// 	const FMatrix& Phi_blur, const vector<int>& candidates, int windowLen,
// 	const FMatrix& logs_0, double p0_blur, double bestScore=-9999);

class FlockLearner {
private:
	CMatrix _T;
	FMatrix _Phi;
	FMatrix _Phi_blur;
//	ArrayXXd _pattern;
	FArray _pattern;
	vector<length_t> _startIdxs;
	vector<length_t> _endIdxs;
	length_t _Lmin;
	length_t _Lmax;
	length_t _Lfilt;
	length_t _windowLen;

	vector<double> _seedScores; //TODO remove
	vector<length_t> _seeds; //TODO remove

	// class Impl;
	// pimpl<Impl> _self;
	// std::unique_ptr<Impl> _self;
public:
	FlockLearner(const FlockLearner& other) = delete;
	FlockLearner& operator=(const FlockLearner&) = delete;

	FlockLearner() {}; // can't do = default or SWIG gets confused

	// ------------------------ pimpl version
	// forward ctor args to pimpl -- all we need if only calling from cpp
	// template<typename ...Args> FlockLearner( Args&& ...args ):
	// 	_self{ std::forward<Args>(args)... }
	// { }

	// FMatrix getFeatureMat();
	// FMatrix getBlurredFeatureMat();
	// FMatrix getPattern();
	// vector<length_t> getInstanceStartIdxs();
	// vector<length_t> getInstanceEndIdxs();

//	FlockLearner(int foo);
	// explicit ctors for SWIG

	// ------------------------ not-so pimpl version

	void learn(const double* X, int d, int n, double m_min, double m_max,
		double m_filt=-1);

	CMatrix getTimeSeries() const { return _T; }
	FMatrix getFeatureMat() const { return _Phi; }
	FMatrix getBlurredFeatureMat() const { return _Phi_blur; }
	FMatrix getPattern() const { return _pattern.matrix(); }
	// ArrayXXd& getPattern() { return _pattern; }
	vector<length_t> getInstanceStartIdxs() const;
	vector<length_t> getInstanceEndIdxs() const;



	// TODO remove
	int getWindowLen() const { return (int)_windowLen; }
	double getPatternSum();
	vector<double> getSeedScores();
	vector<length_t> getSeeds();

	vector<length_t> getCandidatesForSeed(FMatrix Phi, FMatrix Phi_blur,
		int seed);
};

// TODO remove after debug
CMatrix createRandomWalks(const double* seq, int seqLen,
	int walkLen, int nwalks=100);
vector<double> structureScores1D(const double* seq, int seqLen,
							  int subseqLen, const CMatrix& walks);

// // explicit instantiation for SWIG // TODO remove after debug
// using Eigen::Matrix;
// using Eigen::Dynamic;
// using Eigen::RowMajor;
// template Matrix<data_t, Dynamic, Dynamic, RowMajor>
// createRandWalks<double>(const double* seq, length_t seqLen, length_t walkLen,
// 	length_t nwalks);


#endif
