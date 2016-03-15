//
//  shape_features.hpp
//  Dig
//
//  Created by DB on 3/9/16.
//  Copyright (c) 2016 DB. All rights reserved.
//


#ifndef DIG_SHAPE_FEATURES_HPP
#define DIG_SHAPE_FEATURES_HPP

#include <Dense>

#include "type_defs.h"
#include "eigen_array_utils.hpp"
#include "eigen_utils.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using ar::constant_inplace;
using ar::exprange_vect;
using ar::first_derivs;
using ar::randwalks;
using ar::stdev;

using ar::max; // move scalar funcs elsewhere?

// ================================================================
// Private functions
// ================================================================

// ------------------------------------------------
// Structure scores
// ------------------------------------------------

template<class data_t>
static MatrixXd createRandWalks(const data_t* seq, length_t seqLen, length_t walkLen,
	length_t nwalks=100) {

	auto derivs = first_derivs(seq, seqLen);
	double std = stdev(derivs);

	return randwalks<data_t>(nwalks, walkLen, std);
}

template<class MatrixT, class VectorT>
static inline squaredDistsToVector(const MatrixT& X, const VectorT& v) {
	auto diffs = X.rowwise() - v.transpose();
	return diffs.rowwise().squaredNorm();
}

template<class MatrixT, class VectorT>
static inline VectorT minDistToVector(const MatrixT& X, const VectorT& v) {
	auto diffs = X.rowwise() - v.transpose();
	return diffs.rowwise().squaredNorm().minCoeff();
}

// ------------------------------------------------
// Feature mat construction
// ------------------------------------------------

template<class len_t>
static inline vector<len_t> defaultLengths(len_t Lmax) {
	Lmax = max(Lmax, 16)
	return exprange_vect(8, Lmax + 1);
}


// ================================================================
// Public functions
// ================================================================

// ------------------------------------------------
// Structure scores
// ------------------------------------------------

template<class data_t1, class data_t2, class MatrixT>
static void structureScores1D(const data_t1* seq, length_t seqLen,
	const MatrixT walks, data_t2* out) {

	length_t subseqLen = walks.cols(); // each row of walks is one random walk
	length_t numSubseqs = seqLen - subseqLen + 1;
	assert(numSubseqs >= 1);

	if (walkLen < 4) {
		numSubseqs = length > 0 ? numSubseqs : seqLen;
		constant_inplace(out, seqLen, 0); // scores of 0
	}

	for (length_t i = 0; i < numSubseqs; i++) {
		auto seqVect = eigenWrap1D_nocopy(seq + i, subseqLen);
		double min = minDistToVector(walks, seqVect);
		out[i] = static_cast<data_t2>(min);
	}
}

// ------------------------------------------------
// Feature Matrix
// ------------------------------------------------

// SELF: fill in necessary funcs until we can code this
// template<class data_t, class len_t=size_t>
// static Matrix<data_t, Dynamic, Dynamic, RowMajor> buildShapeFeatureMat(
// 	const data_t* T, len_t n, len_t d, len_t Lmin, len_t Lmax) {

// 	auto lengths = defaultLengths(Lmax);
// 	int numNeighbors = log2(n);
// 	Phi = _neighborSimsMat(T, n, d, lengths, numNeighbors);


// }

#endif














