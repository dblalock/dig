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

#include "array_utils_eigen.hpp" // for randwalks
#include "subseq.hpp"
#include "eigen_utils.hpp"

using Eigen::Map; // TODO remove
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;

using ar::constant_inplace;
using ar::length_t;
using ar::mean;
using ar::normalize_mean_inplace;
using ar::randwalks;
using ar::stdev;

using subs::first_derivs;
using subs::mapSubseqs;

typedef double data_t;

//typedef int64_t length_t;
// typedef Matrix<double, Dynamic, Dynamic, RowMajor> CMatrixXd;

//template<class data_t>
//using CMatrix = Matrix<data_t, Dynamic, Dynamic, RowMajor>;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> CMatrix;

//template<class data_t>
//using FMatrix = Matrix<data_t, Dynamic, Dynamic>
// typedef Matrix<double, Dynamic, Dynamic> FMatrix;
typedef Eigen::MatrixXd FMatrix;

// ------------------------------------------------
// Structure scores
// ------------------------------------------------

template<class data_t>
static Matrix<data_t, Dynamic, Dynamic, RowMajor> createRandWalks(
	const data_t* seq, length_t seqLen, length_t walkLen, length_t nwalks=100)
{
	auto derivs = first_derivs(seq, seqLen);
	double std = stdev(derivs.get(), seqLen - 1);

	CMatrix walks = randwalks<data_t>(nwalks, walkLen, std);

	for (int i = 0; i < nwalks; i++) {
		data_t* rowStart = walks.row(i).data();
		normalize_mean_inplace(rowStart, walkLen);
	}
	return walks;
}
// explicit instantiation for SWIG
// template Matrix<data_t, Dynamic, Dynamic, RowMajor>
// createRandWalks<double>(const data_t* seq, length_t seqLen, length_t walkLen,
// 	length_t nwalks);

template<class DenseT, class VectorT>
static inline auto distsSqToVector(const DenseT& X, const VectorT& v)
	-> Matrix<typename VectorT::Scalar, Dynamic, 1>
{
	return (X.rowwise() - v.transpose()).rowwise().squaredNorm().eval();
}

template<class DenseT, class VectorT>
static inline double minDistSqToVector(const DenseT& X, const VectorT& v) {
	return distsSqToVector(X, v).minCoeff();
}

template<class data_t1, class data_t2, class DenseT>
static void structureScores1D(const data_t1* seq, length_t seqLen,
							  length_t subseqLen, const DenseT& walks,
							  data_t2* out)
{
	//	length_t subseqLen = walks.cols(); // each row of walks is one random walk
	length_t numSubseqs = seqLen - subseqLen + 1;
	assert(numSubseqs >= 1);

	if (subseqLen < 4) {
		//		numSubseqs = subseqLen > 0 ? numSubseqs : seqLen;
		constant_inplace(out, seqLen, 0); // scores of 0
	}

	double normalized[subseqLen];
	mapSubseqs([subseqLen, &walks, &normalized](const data_t1* subseq) {
		ar::normalize_mean(subseq, subseqLen, normalized);
		// NOTE: it is *really* important that this be a const matrix or
		// Eigen will spew a wall of inscrutable errors about ambiguous
		// function overloads and return types
		Map<const Matrix<data_t1, Dynamic, 1> > seqVect(normalized, subseqLen);
		double min = minDistSqToVector(walks, seqVect);
		return min;
		// double minDist = 1e20; // huge number TODO std::huge_val or whatever
		// for (int w = 0; w < walks.rows(); w++) {
		// 	// double dist = 0;
		// 	// const double* walkRowPtr = walks.row(w).data();
		// 	// for (int i = 0; i < subseqLen; i++) {
		// 	// 	// double diff = (normalized[i] - walkRowPtr[i]);
		// 	// 	double diff = (normalized[i] - walks(w, i));
		// 	// 	dist += diff * diff;
		// 	// }
		// 	double dist = ar::dist_sq(normalized, walks.row(w).data(), subseqLen);
		// 	if (dist < minDist) {
		// 		minDist = dist;
		// 	}
		// }
		// return minDist;
	}, subseqLen, seq, seqLen, out);

	// write 0s past end of valid range
	constant_inplace(out + numSubseqs, seqLen - numSubseqs, 0);
}

// ------------------------------------------------
// Feature matrix
// ------------------------------------------------

std::pair<FMatrix, FMatrix> buildShapeFeatureMats(const double* T,
	length_t d, length_t n, length_t Lmin, length_t Lmax, length_t Lfilt);

#endif














