
#include "shape_features.hpp"

//#include "type_defs.h"
#include "array_utils_eigen.hpp"
#include "eigen_utils.hpp"
#include "subseq.hpp"
#include "slice.hpp"
#include "filter.hpp"

#include "debug_utils.h" // TODO remove

using Eigen::MatrixXd;
using Eigen::VectorXd;

using ar::constant_inplace;
using ar::copy;
using ar::dist_sq;
using ar::div_inplace;
using ar::exprange;
using ar::max_inplace;
using ar::mean_and_variance;
using ar::normalize_mean_inplace;
using ar::pad;
using ar::PAD_EDGE;
using ar::rand_idxs;
using ar::randwalks;
using ar::stdev;
using ar::sum;
using ar::variance;
using ar::max; // move scalar funcs elsewhere?

using filter::hamming_ar;
using filter::max_filter;

//using slice::vstack;
using subs::first_derivs;
using subs::mapSubseqs;
using subs::crossCorrs;

#define DEFAULT_NONZERO_THRESH .001

static const double kDistThresh = .25;

typedef int64_t length_t;
// typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> CMatrixXd;

//template<class data_t>
//using CMatrix = Eigen::Matrix<data_t, Dynamic, Dynamic, RowMajor>;
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> CMatrix;

//template<class data_t>
//using FMatrix = Eigen::Matrix<data_t, Dynamic, Dynamic>
typedef Eigen::Matrix<double, Dynamic, Dynamic> FMatrix;

// ================================================================
// Private functions
// ================================================================

// ------------------------------------------------
// Feature mat construction
// ------------------------------------------------

template<class len_t>
static inline vector<len_t> defaultLengths(len_t Lmax) {
	Lmax = max(Lmax, 16);
	return exprange(8, Lmax + 1); // +1 since range non-inclusive
}

// static inline int64_t numSubseqs(int64_t seqLen, int64_t subseqLen) {
// 	return seqLen - subseqLen + 1;
// }

// ================================================================
// Public functions
// ================================================================

// ------------------------------------------------
// Structure scores
// ------------------------------------------------

// ------------------------------------------------
// feature mat
// ------------------------------------------------

// assumes mat stored in row-major order
template<class data_t>
static data_t* rowStartPtr(length_t row, data_t* T, length_t nrows,
						   length_t ncols)
{
	return T + (row * ncols);
}

template<class data_t, class dist_t, class DenseT>
static vector<length_t> selectShapeIdxs(const data_t* seq, length_t seqLen,
	length_t subseqLen, length_t howMany, const DenseT& walks, dist_t* scores)
{
	// compute scores for each subsequence and store them in scores
	structureScores1D(seq, seqLen, subseqLen, walks, scores);
	if (sum(scores, seqLen) <= 0) {
		vector<length_t> ret;
		return ret;
	}
	// select idxs with probability proportional to their scores
	return rand_idxs(seqLen, howMany, false, scores);
}

// writes to first seqLen elements of out; writes neighborLen / 2 zeros at
// start, then numSubseqs 0s or 1s, depending on similarity, then writes
// zeros until seqLen total values have been written
template<class data_t1, class data_t2, class sim_t>
static void neighborSims1D(const data_t1* seq, length_t seqLen,
						   data_t2* neighbor, length_t neighborLen, sim_t* out)
{
	// if neighbor (shape) is completely flat, no similarity to anything
	double neighborVariance = variance(neighbor, neighborLen);
	if (neighborVariance < DEFAULT_NONZERO_THRESH) {
		// print("skipping flat neighbor");
		constant_inplace(out, seqLen, 0);
		return;
	}

	// write 0s before similarity is well-defined
	auto numSubseqs = seqLen - neighborLen + 1;
	auto prePadLen = neighborLen / 2;
	auto postPadLen = seqLen - (prePadLen + numSubseqs);
	constant_inplace(out, prePadLen, 0);

	// compute similarity between neighbor and all subseqs
	normalize_mean_inplace(neighbor, neighborLen);

	// TODO remove
	// auto avg = mean(neighbor, neighborLen);
	// assert(abs(avg) < .0001);

	mapSubseqs([neighbor, neighborLen, neighborVariance](const data_t1* subseq) {
		// double mean, variance;
		// mean_and_variance(subseq, neighborLen, mean, variance);
		double subseqMean = ar::mean(subseq, neighborLen);
		double dist = 0;
		for (length_t i = 0; i < neighborLen; i++) {
			dist += dist_sq(neighbor[i], subseq[i] - subseqMean);
		}
		dist /= (neighborVariance * neighborLen);
		// return (1.0 - dist) * (dist < kDistThresh);
//		return (1.0 - dist); // TODO uncomment above
		// return dist < 1.0 ? (1.0 - dist) : 0.0; // TODO uncomment above
		// return dist < kDistThresh ? (1.0 - dist) : 0.0;
		// return static_cast<sim_t>(dist < kDistThresh);
		return dist < kDistThresh ? 1.0 : 0.0;

	}, neighborLen, seq, seqLen, out + prePadLen);

	// write 0s after similarity is well-defined
	constant_inplace(out + prePadLen + numSubseqs, postPadLen, 0);
}

// out must be storage for a numNeighbors x seqLen array
template<class data_t1, class dist_t, class sim_t>
static size_t neighborSimsOneSignal(const data_t1* seq, length_t seqLen,
	length_t subseqLen, length_t numNeighbors, dist_t* scores_tmp, sim_t* out)
{
	MatrixXd walks = createRandWalks(seq, seqLen, subseqLen);
	auto idxs = selectShapeIdxs(seq, seqLen, subseqLen, numNeighbors, walks,
								scores_tmp);

	// PRINT_VAR(walks.rows());
	// PRINT_VAR(walks.cols());
	// PRINT_VAR(ar::to_string(idxs));

	data_t1 neighborTmp[subseqLen];
	//	auto numSubseqs = seqLen - subseqLen + 1;
	auto numShapes = idxs.size();

	// DEBUGF("using %lu neighbors", numShapes);

	// CMatrix ret(numShapes, numSubseqs);
	//	double* retPtr = ret.data();
	for (int i = 0; i < numShapes; i++) {
		auto idx = idxs[i];
		const data_t1* neighbor = seq + idx;
		copy(neighbor, subseqLen, neighborTmp);

		// assert(ar::all_eq(neighbor, neighborTmp, subseqLen)); // TODO remove

		// double* outPtr = ret.row(i).data();
		// double* outPtr = retPtr + (i * numSubseqs);
		double* outPtr = rowStartPtr(i, out, numShapes, seqLen);

		neighborSims1D(seq, seqLen, neighborTmp, subseqLen, outPtr);

		// printf("nonzeros for neighbor %lld: ", idx);
		// auto nonzeroIdxs = ar::nonzeros(outPtr, seqLen);
		// print(ar::to_string(nonzeroIdxs).c_str());

//		auto selfSimilarity = *(outPtr + idx);
//		assert(selfSimilarity > 0);
	}
	return idxs.size();
}

// NOTE: assumes each signal is a row in the matrix, not a column
template<class data_t>
//void neighborSimsMat(const data_t* T, length_t d, length_t n,
CMatrix neighborSimsMat(const data_t* T, length_t d, length_t n,
						length_t Lmin, length_t Lmax)
{
	auto lengths = defaultLengths(Lmax);
	length_t numNeighbors = std::log2(n);

	// PRINT_VAR(d);
	// PRINT_VAR(n);
	// PRINT_VAR(ar::to_string(lengths));
	// PRINT_VAR(numNeighbors);

	length_t numLengths = lengths.size();
	length_t numRows = d * numNeighbors * numLengths;
	length_t paddedLen = n + 2 * Lmax;
	// length_t numCols = n;
	length_t numCols = paddedLen;

	CMatrix Phi(numRows, numCols);
	// Phi.setZero();

	// PRINT_VAR(paddedLen);

	// create feature mat for each dimension for each length
	double signal_tmp[paddedLen];
	double scores_tmp[paddedLen];
	double sims_tmp[numNeighbors * paddedLen];

	// print("about to enter neighborSims for loop...");

	int currentRowOut = 0;
	// double* rowOutPtr = Phi.data();
	double* rowOutPtr;
	for (int dim = 0; dim < d; dim++) {
		auto rowPtr = rowStartPtr(dim, T, d, n);
		// extend signal by replicating endpoints; store result in signal_tmp
		pad(rowPtr, n, Lmax, Lmax, signal_tmp, PAD_EDGE);

		for (const auto subseqLen : lengths) {
			// DEBUGF("computing neighbor sims for dim %d...", dim);
			// sample numNeighbors shapes from the data and store the
			// similarity of the whole signal to each in a row of sims_tmp;
			// if the signal is flat, we may find fewer than numNeighbors
			// unique shapes
			auto numNeighborsFound = neighborSimsOneSignal(signal_tmp, paddedLen,
				subseqLen, numNeighbors, scores_tmp, sims_tmp);
			// DEBUGF("computed neighbor sims for dim %d", dim);

			// copy similarities for each feature into output matrix
			auto numSubseqs = n - subseqLen + 1;
			double* simsRowPtr;
			for (int i = 0; i < numNeighborsFound; i++) {
				simsRowPtr = sims_tmp + (i * paddedLen);
				rowOutPtr = Phi.row(currentRowOut).data();

				// TODO remove
				// auto nanIdxs = ar::where([](double x) { return !isfinite(x); },
				// 						 simsRowPtr, paddedLen);
				// if (nanIdxs.size()) {
				// 	printf("----> ERR!!! found nan idxs in dim %d <----", dim);
				// 	PRINT_VAR(ar::to_string(nanIdxs));
				// 	exit(1);
				// }

				// auto minVal = ar::min(simsRowPtr, paddedLen);
				// auto maxVal = ar::max(simsRowPtr, paddedLen);
				// printf("val range for neighbor %d.%d: %g-%g", dim, i,
				// 	   minVal, maxVal);

// #define SKIP_FOR_NOW
#ifndef SKIP_FOR_NOW
				auto rowSum = sum(simsRowPtr + Lmax, numSubseqs);
//				PRINT_VAR(rowSum);
				// printf( "(sum = %g)\n", rowSum);
				if (rowSum < 1) {
					// printf("skipping row with tiny sum\n");
					continue;
				} else if (rowSum > (numSubseqs / 2)) {
					// printf("skipping row that's mostly 1s\n");
					continue;
				}
#endif

//				PRINT_VAR(ar::to_string(simsRowPtr, paddedLen));
				// Phi has numSubseqs cols, but sims in each row
				// have seqLen + 2*Lmax cols due to padding; thus,
				// we copy over only the middle portion
				// auto startOffset = Lmax;
				// copy(simsRowPtr, n, rowOutPtr + startOffset);

				copy(simsRowPtr, paddedLen, rowOutPtr); // actually, remove padding after Phi_blur
				currentRowOut++; // increment output row
				// DEBUGF("copied over neighbor sims for feature %d", i);
			}
		}
	}

	// PRINT_VAR(currentRowOut);
//	for (int i = 0; i < ar::min(10, currentRowOut); i++) {
//		printf("%d) ", i);
//		auto rowString = ar::to_string(Phi.row(i).data(), 5);
//		PRINT_VAR(rowString);
//	}

	// TODO we need to propagate this up as an error if everything was
	// 0...

	int numRowsToCopy = max(currentRowOut, 1); // return at least one row
	return Phi.topRows(numRowsToCopy).eval(); // remove empty rows

//	PRINT_VAR(Phi.rows());
//	PRINT_VAR(currentRowOut);
//	print("finished calculating phi");
////	CMatrix tmp(Phi.topRows(currentRowOut)); // remove empty rows
//
//
//
//	CMatrix tmp(currentRowOut, Phi.cols());
//	tmp = Phi.topRows(currentRowOut);
//
//
//	std::cout << "tmp rows: " << tmp.rows() << "\n";
//	PRINT_VAR(tmp.rows());
//	PRINT_VAR(tmp.cols());
//	PRINT_VAR(tmp.IsRowMajor);
//	PRINT_VAR(Phi.rows());
//
//	CMatrix out(tmp);
//	out += out;
//
//	print("finished calculating out");
//
//	// return out;
}

// ------------------------------------------------
// blurred feature mat
// ------------------------------------------------

CMatrix blurFeatureMat(const CMatrix& Phi, length_t Lfilt) {
	auto d = Phi.rows();
	auto n = Phi.cols(); // note that this is with padding at this point
	CMatrix Phi_blur(d, n);
	auto filt = hamming_ar(Lfilt);

	double max_tmp[n];
	for (int i = 0; i < d; i++) {
		const double* rowInPtr = Phi.data() + (i * n);
		double* rowOutPtr = Phi_blur.data() + (i * n);
		double* filtOutPtr = rowOutPtr + Lfilt / 2; // center hamming window
		length_t filtOutLen = n - Lfilt / 2;

		// convolve with hamming filt (equiv to cross-corr since symmetric)
		crossCorrs(filt.get(), Lfilt, rowInPtr, filtOutLen, filtOutPtr);
		// divide by largest value within Lfilt / 2
		max_filter(rowOutPtr, n, Lfilt/2, max_tmp);
		max_inplace(1.0, max_tmp, n); // divide by at least 1, so no div by 0
//		max_inplace(max_tmp, n, 1.0); // works; above ordering doesn't, though...
//		max_inplace(1, max_tmp, n); // also works...
		div_inplace(rowOutPtr, max_tmp, n);

		// auto minVal = ar::min(rowInPtr, n);
		// auto maxVal = ar::max(rowInPtr, n);
		// auto minValOut = ar::min(rowOutPtr, n);
		// auto maxValOut = ar::max(rowOutPtr, n);
		// printf("val range for row %d: %g-%g\t->\t%g-%g\n", i, minVal, maxVal,
		// 	   minValOut, maxValOut);

	}
	return Phi_blur;
}

// ------------------------------------------------
// Feature Matrix
// ------------------------------------------------

std::pair<FMatrix, FMatrix> buildShapeFeatureMats(const double* T,
	length_t d, length_t n, length_t Lmin, length_t Lmax, length_t Lfilt)
{
//	neighborSimsMat(T, d, n, Lmin, Lmax);
	CMatrix Phi(neighborSimsMat(T, d, n, Lmin, Lmax));
//	CMatrix Phi(3, 4);
//	Phi.setRandom();
	// print("computed neighbor sims mat without asploding");

	// PRINT_VAR(Phi.rows());
	// PRINT_VAR(Phi.cols());

	CMatrix Phi_blur(blurFeatureMat(Phi, Lfilt));
	// print("blurred neighbor sims mat without asploding");

	// FMatrix Phi_colmajor(Phi.rows(), n);
	// FMatrix Phi_blur_colmajor(Phi_blur.rows(), n);

	// Phi_colmajor = Phi.middleCols(Lmax, n);
	// Phi_blur_colmajor = Phi_blur.middleCols(Lmax, n);

	FMatrix Phi_colmajor(Phi.middleCols(Lmax, n));
	FMatrix Phi_blur_colmajor(Phi_blur.middleCols(Lmax, n));
	// print("built col-major mats without asploding");

	// print("------------------------ buildShapeFeatureMats() ");
	// for(int i = 0; i < Phi_colmajor.rows(); i++) {
	// 	// auto rowInPtr = Phi.row(i).data();
	// 	auto minVal = Phi_colmajor.row(i).minCoeff();
	// 	auto maxVal = Phi_colmajor.row(i).maxCoeff();
	// 	printf("val range for Phi row %d:\t%g-%g;\t", i, minVal, maxVal);

	// 	minVal = Phi_blur_colmajor.row(i).minCoeff();
	// 	maxVal = Phi_blur_colmajor.row(i).maxCoeff();
	// 	printf("%g-%g\n", minVal, maxVal);
	// }


	return std::pair<FMatrix, FMatrix>(Phi_colmajor, Phi_blur_colmajor);
}

// ================================================================
// Debug // TODO remove
// ================================================================

vector<double> structureScores1D(const double* seq, int seqLen,
							  int subseqLen, const CMatrix& walks) {
	vector<double> ret(seqLen);
	structureScores1D(seq, seqLen, subseqLen, walks, ret.data());
	return ret;
}



