

#include "flock.hpp"

#include <vector>
#include <iostream> // TODO remove

#include "Dense"

// #include "type_defs.h"
#include "array_utils.hpp"
#include "eigen_utils.hpp"
// #include "pimpl_impl.hpp"
#include "subseq.hpp"
#include "shape_features.hpp"

// TODO remove these
#include <iostream> // for to_string
#include "debug_utils.h"
#include "timing_utils.hpp"
#include "plot.hpp"

// using Eigen::Array;
using Eigen::Matrix;
using Eigen::ArrayXd;
using Eigen::ArrayXXd; // two Xs for 2d array
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::vector;

using ar::add_inplace;
using ar::argmax;
using ar::argsort;
using ar::at_idxs;
using ar::exprange;
using ar::filter;
using ar::max;
using ar::min;
using ar::normalize_max_inplace;
using ar::range;
using ar::sort;
using ar::unique;

using ar::plot_array;

using subs::local_maxima;
using subs::maximum_subarray;

// ================================================================ TEMP DEBUG

CMatrix createRandomWalks(const double* seq, int seqLen,
	int walkLen, int nwalks) {
	return createRandWalks(seq, seqLen, walkLen, nwalks);
}

// ================================================================ Constants

static constexpr double kNegativeInf = -std::numeric_limits<double>::infinity();

// ================================================================ funcs

static inline length_t computeWindowLen(length_t Lmin, length_t Lmax) {
	return Lmax + (Lmax / 10); // Lmax/10 = step size of seed generation
}

vector<double> seedScores(CMatrix T, length_t Lmin, length_t Lmax) {
	length_t d = T.rows();
	length_t n = T.cols();
	vector<double> scores(n);
	assert(ar::all([](double x) { return x == 0; }, scores)); // TODO remove
	double* out = scores.data();

	auto lengths = exprange(8, max(Lmax, 16) + 1);
	// auto lengths = exprange(4, max(Lmin, 16) + 1);

	PRINT_VAR(ar::to_string(lengths));

	PRINT_VAR(d);

	double scores_tmp[n];
	for (int dim = 0; dim < d; dim++) {
		const double* signal = T.row(dim).data();

		// auto signal_plot = ar::resample(signal, n, 80);
		// plot_array(signal_plot.get(), 80, -1, .25, "seedScores signal");

		for (length_t subseqLen : lengths) {
			CMatrix walks = createRandWalks(signal, n, subseqLen);
			structureScores1D(signal, n, subseqLen, walks, scores_tmp);

			// auto isZero = ar::logical_not(scores_tmp, n);
			// auto whereFalse = ar::where(isZero.get(), n);
//			auto whereFalse = ar::where_false(scores_tmp, n);
//			PRINT_VAR(ar::to_string(whereFalse));
//			PRINT_VAR(n);
//			PRINT_VAR(subseqLen);
			// assert(ar::all(scores_tmp, n - subseqLen));
			assert(ar::all(scores_tmp, n - subseqLen - 1));

			// scale scores to max of 1 in place, then add to combined scores
			normalize_max_inplace(scores_tmp, n);
			add_inplace(out, scores_tmp, n);

			PRINT_VAR(subseqLen);

			// auto scores_plot = ar::resample(scores_tmp, n, 80);
			// plot_array(scores_plot.get(), 80, -1, .25, "seedScores scores_tmp");

//			plot_array(scores_tmp, 90, -1, .25, "scores_tmp");
			// plot_array(out, 90, -1, .25, "out");
			// PRINT_VAR(ar::min(scores_tmp, 80));
			// PRINT_VAR(ar::max(scores_tmp, 80));
			// PRINT_VAR(ar::min(out, 80));
			// PRINT_VAR(ar::max(out, 80));
			// PRINT_VAR(ar::to_string(signal, n));
//			PRINT_VAR(ar::to_string(scores_tmp, n));
//			PRINT_VAR(ar::to_string(out, n));

			assert(ar::all(scores_tmp, n - subseqLen - 1));
			assert(max(scores_tmp, n) == 1.0);
			assert(min(scores_tmp, n) >= 0);
			assert(ar::all_positive(scores_tmp, n));
			assert(ar::all_positive(scores_tmp, n));
		}
	}

	// auto out_plot = ar::resample(out, n, 80);
	// plot_array(out_plot.get(), 80, -1, .25, "seedScores out");

	return scores;
}

vector<length_t> selectSeeds(const double* scores, length_t n,
	length_t Lmin, length_t Lmax, length_t windowLen)
{
	length_t seed1 = argmax(scores, n);

	auto _scores = ar::copy(scores, n); // mutable copy

	// PRINT_VAR(ar::to_string(scores, n));

	// make scores around this seed negative so we don't select
	// any of them as the second seed
	auto zeroRadius = Lmin;
	auto start = _scores.get() + max(0, seed1 - zeroRadius);
	auto length = min(zeroRadius, n - 1 - seed1);
	constant_inplace(start, length, 0);

	length_t seed2 = argmax(_scores.get(), n);

	vector<length_t> firstTwoSeeds {seed1, seed2};
	PRINT_VAR(ar::to_string(firstTwoSeeds));

	// add a bunch of other seeds around these initial seeds
	vector<length_t> seeds {seed1, seed2};
	for (int i = 1; i <= 10; i++) {
		length_t offset = i * (Lmax / 10);
		seeds.push_back(seed1 + offset);
		seeds.push_back(seed1 - offset);
		seeds.push_back(seed2 + offset);
		seeds.push_back(seed2 - offset);
	}
	// ensure no duplicates or violations of array bounds
	auto uniqSeeds = unique(seeds);
	auto maxIdx = n - windowLen;
	return filter([maxIdx](length_t seed) {
		return 0 <= seed && seed <= maxIdx;
	}, uniqSeeds);
}

vector<double> dotProdsForSeed(const FMatrix& Phi, const FMatrix& Phi_blur,
	int seed, int windowLen)
{
	length_t numSubseqs = Phi.cols() - windowLen + 1;
	FArray seedWindow = Phi.middleCols(seed, windowLen).array();
	FArray Phi_blur_ar = Phi_blur.array();
	vector<double> dotProds(numSubseqs);
	// double sumDirect = 0;
	for (int i = 0; i < numSubseqs; i++) {
		// dotProds[i] = (seedWindow * (Phi.middleCols(i, windowLen).array())).sum();
		// dotProds[i] = (seedWindow * (Phi_blur.middleCols(i, windowLen).array())).sum();
		dotProds[i] = (seedWindow * (Phi_blur_ar.middleCols(i, windowLen))).sum();
		// sumDirect += dotProds[i];
	}
	return dotProds;
}

vector<length_t> candidatesFromDotProds(const vector<double>& dotProds, int Lmin)
{
	assert(ar::all_nonnegative(dotProds)); // TODO remove

	length_t minSpacing = Lmin;
	// length_t minSpacing = Lmin / 2;
	auto maxima = local_maxima(dotProds, minSpacing);
	auto maximaVals = at_idxs(dotProds, maxima);

	bool ascending = false;
	auto sortIdxs = argsort(maximaVals, ascending);
	return at_idxs(maxima, sortIdxs);
}

// returns all candidates in decreasing order of dot product
vector<length_t> candidatesForSeed(const FMatrix& Phi, const FMatrix& Phi_blur,
	length_t seed, length_t Lmin, length_t Lmax, length_t windowLen)
{
	// TODO static cast or have other funcs use length_t after debug
	vector<double> dotProds = dotProdsForSeed(Phi, Phi_blur,
		(int) seed, (int) windowLen);
	return candidatesFromDotProds(dotProds, (int) Lmin);
}

vector<length_t> selectFromCandidates(const FMatrix& Phi,
	const FMatrix& Phi_blur, const vector<int>& candidates, int windowLen,
	const FMatrix& logs_0, double p0_blur, double bestScore)
{
	int bestNumInstances = -1;

	auto J = logs_0.size();
	assert(J == Phi.rows() * windowLen);

	FArray pattern(Phi.rows(), windowLen);
	FArray FblurAr(Phi_blur.array());

	static const double kTinyConst = .0001;
	length_t idx = candidates[0];
	FArray counts = Phi.middleCols(idx, windowLen).array();
	FArray counts_blur = Phi_blur.middleCols(idx, windowLen).array() + kTinyConst;

	for (int i = 1; i < candidates.size(); i++) {
		auto idx = candidates[i];
		int k = i + 1;

		auto phiPtr = Phi.data() + idx;
		auto phiBlurPtr = Phi_blur.data() + idx;
		auto countsPtr = counts.data();
		auto countsBlurPtr = counts_blur.data();
		auto logs0Ptr = logs_0.data();
		auto patternPtr = pattern.data();

		bool hasNext = k < candidates.size();
		auto nextPtr = Phi_blur.data();
		if (hasNext) {
			nextPtr += candidates[k]; // next window
		}

		double score = 0;
		double enemyScore = 0;
		double randomScore = 0;
		for (int j = 0; j < J; j++) {
			countsPtr[j] += phiPtr[j];
			double count = countsPtr[j];
			countsBlurPtr[j] += phiBlurPtr[j];

			double theta_1 = countsBlurPtr[j] / k;
			double log_1 = log(theta_1);
			double log_0 = logs0Ptr[j];
			double logDiff = log_1 - log_0;
			bool mask = (log_1 > log_0) && (theta_1 > .5);
			logDiff = mask ? logDiff : 0;

			patternPtr[j] = logDiff;
			score += logDiff * count;
			if (hasNext) {
				enemyScore += logDiff * nextPtr[j];
			}
			randomScore += logDiff;
		}
		randomScore *= k * p0_blur;

		double penalty = max(randomScore, enemyScore);
		double adjustedScore = score - penalty;

		if (adjustedScore > bestScore) {
			bestScore = adjustedScore;
			bestNumInstances = k;
		}

		assert(score >= 0);

		assert(!Phi.IsRowMajor);
		assert(!Phi_blur.IsRowMajor);
		assert(!logs_0.IsRowMajor);
		assert(ar::all_nonnegative(pattern.data(), J));
		assert(ar::all_nonnegative(counts.data(), J));
		assert(ar::all_nonnegative(counts_blur.data(), J));
		assert(counts.sum() > 0);
		assert(counts_blur.sum() > 0);
		assert(pattern.sum() > 0);
	}

	// return first k instances
	vector<length_t> ret(bestNumInstances);
	for (int i = 0; i < bestNumInstances; i++) {
		ret[i] = candidates[i];
	}
	return ret;
}

// void instancesForSeed(const FMatrix& Phi, const FMatrix& Phi_blur,
// 	length_t seed, length_t Lmin, length_t Lmax, length_t windowLen,
// 	const FMatrix& logs_0, double p0_blur,
// 	double& bestScore, FArray& bestPattern, vector<length_t>& bestInstances)
// 	// const ArrayXXd& logs_0, double p0_blur,
// 	// double& bestScore, ArrayXXd& bestPattern, vector<length_t>& bestInstances)
// {

// 	// printf("================================ %lld\n", seed);
// 	// PRINT_VAR(bestScore);

// 	// auto windowLen = computeWindowLen(Lmin, Lmax);
// 	auto candidates = candidatesForSeed(Phi, Phi_blur, seed,
// 		Lmin, Lmax, windowLen);

// 	printf("================================ %lld: ", seed);
// 	auto candidatesStr = ar::to_string(candidates);
// 	printf("%s\n", candidatesStr.c_str());

// 	// initialize set of candidates and counts with best candidate; we add
// 	// a tiny constant to the blurred counts to that we don't get -inf logs
// 	// from zeros
// 	static const double kTinyConst = .0001;
// 	length_t idx = candidates[0];
// 	vector<length_t> usedCandidates;
// 	usedCandidates.push_back(idx);
// 	ArrayXXd counts = Phi.middleCols(idx, windowLen).array();
// 	ArrayXXd counts_blur = Phi_blur.middleCols(idx, windowLen).array() + kTinyConst;

// 	static const double LOG_0p5 = -.6931471806; // ln(.5)

// 	ArrayXXd pattern(Phi.rows(), windowLen);

// 	// ArrayXXd zeros(Phi.rows(), windowLen); // TODO remove
// 	// zeros.setZero();

// 	auto J = logs_0.size();
// 	assert(J == Phi.rows() * windowLen);
// //	PRINT_VAR(J);

// 	assert(!Phi.IsRowMajor);
// 	assert(!Phi_blur.IsRowMajor);
// 	assert(!bestPattern.IsRowMajor);
// 	assert(!logs_0.IsRowMajor);

// 	FArray FblurAr(Phi_blur.array());

// 	// int bestNumInstances = 1;
// 	for (int i = 1; i < candidates.size(); i++) {

// 		// print("-----");

// 		auto idx = candidates[i];
// 		int k = i + 1;

// 		// counts_blur += Phi_blur.middleCols(idx, windowLen).array();
// 		// here are the steps we notionally compute; we combine them into
// 		// two expressions below to ensure that we make use of Eigen
// 		// expression templates
// 		// ArrayXXd theta_1 = (counts_blur / k);
// 		// ArrayXXd logs_1 = theta_1.log();
// 		// ArrayXXd logs_1 = (counts_blur / k).log();
// 		// ArrayXXd logs_1 = (theta_1 > .5).select(logs_1_tmp, 0);
// 		// ArrayXXd logDiffs = (logs_1 - logs_0);
// 		// ArrayXXd mask = (theta_1 > .5) && (counts > 0);
// 		// .select(counts * logDiffs, 0);
// 		// pattern = ((theta_1 > .5) && (logDiffs > 0)).select(logDiffs, 0);

// 		// counts += Phi.middleCols(idx, windowLen).array();
// 		// double score = (counts * pattern).sum();

// 		// counts += Phi.middleCols(idx, windowLen).array();
// 		// counts_blur += Phi_blur.middleCols(idx, windowLen).array();



// 		auto phiPtr = Phi.data() + idx;
// 		auto phiBlurPtr = Phi_blur.data() + idx;
// 		auto countsPtr = counts.data();
// 		auto countsBlurPtr = counts_blur.data();
// 		auto logs0Ptr = logs_0.data();
// 		auto patternPtr = pattern.data();
// 		double score = 0;
// 		for (int j = 0; j < J; j++) {
// 			countsPtr[j] += phiPtr[j];
// 			double count = countsPtr[j];
// 			countsBlurPtr[j] += phiBlurPtr[j];

// 			double theta_1 = countsBlurPtr[j] / k;
// 			double log_1 = log(theta_1);
// 			double log_0 = logs0Ptr[j];
// 			double logDiff = log_1 - log_0;
// 			bool mask = (log_1 > log_0) && (theta_1 > .5);
// 			logDiff = mask ? logDiff : 0;

// 			patternPtr[j] = logDiff;

// 			// try just using theta_1 instead of logDiffs; not
// 			// mathematically interesting though...
// 			// patternPtr[j] = mask * theta_1;
// 			// patternPtr[j] = theta_1;

// 			 score += logDiff * count;
// 		}

// 		assert(score >= 0);

// 		// pattern is all 0s except for maybe like 1 place;
// 		// ArrayXXd logs_1 = (counts_blur / k).log();
// 		// ArrayXXd logDiffs = (logs_1 - logs_0);
// 		// pattern = ((logs_1 > LOG_0p5) && (logDiffs > 0)).select(logDiffs, 0);

// 		// ArrayXXd logDiffs = ((counts_blur / k).log() - logs_0);
// 		// pattern = (logDiffs * (logDiffs > logs_0.max(LOG_0p5)).cast<double>());

// 		// TODO try going back to this version
// 		// ArrayXXd logs_1 = (counts_blur / k).log();
// 		// pattern = (logs_1 > logs_0.max(LOG_0p5) ).select(logs_1 - logs_0, zeros);


// 		// print("------------------------");



// //		assert(ar::all_nonnegative(logDiffs.data(), logDiffs.size()));
// 		assert(ar::all_nonnegative(pattern.data(), J));

// 		assert(ar::all_nonnegative(counts.data(), J));
// 		assert(ar::all_nonnegative(counts_blur.data(), J));
// //		assert(ar::all_nonnegative( (-logs_1).eval().data(), logs_1.size() ));
// //		bool allNonPositive = ar::all([](double x) { return x <= .0001; },
// //								  logs_1.data(), J);
// //		if (!allNonPositive) {
// //			PRINT_VAR(ar::to_string(logs_1.data(), J));
// //			ArrayXXd means = counts_blur / k;
// //			PRINT_VAR(ar::to_string(means.data(), J));
// //			assert(ar::all_nonnegative(means.data(), J));
// //
// //			auto positiveIdxs = ar::where([](double x) { return x > .0001; },
// //										  logs_1.data(), J);
// //			PRINT_VAR(ar::to_string(positiveIdxs));
// //			double firstPositiveVal = logs_1(positiveIdxs[0]);
// //			PRINT_VAR(firstPositiveVal);
// //			PRINT_VAR(J);
// //			PRINT_VAR(logs_1.size());
// //
// //			assert(allNonPositive);
// //		}

// 		assert(counts.sum() > 0);
// 		assert(counts_blur.sum() > 0);
// 		assert(pattern.sum() > 0);

// 		// ArrayXXd means = counts_blur / k;
// 		// auto meansAbovePt5 = ar::where([](double x) { return x > .5; },
// 		// 	means.data(), J);

// 		// // PRINT_VAR(ar::to_string(meansAbovePt5));
// 		// // auto meansStr = ar::to_string(means.data(), means.size());
// 		// // PRINT_VAR(meansStr);

// 		// auto logDiffs = (logs_1 - logs_0).eval();
// 		// auto logDiffsAbovePt5 = ar::where([](double x) { return x > LOG_0p5; },
// 		// 	logDiffs.data(), J);

// 		// // PRINT_VAR(ar::to_string(logDiffsAbovePt5));
// 		// // auto logDiffsStr = ar::to_string(logDiffs.data(), J);
// 		// // PRINT_VAR(logDiffsStr);

// 		// auto patternNonzeroIdxs = ar::where([](double x) { return x > 0; },
// 		// 									pattern.data(), J);
// 		// // PRINT_VAR(ar::to_string(patternNonzeroIdxs));

// 		// auto logDiffsValsAbovePt5 = at_idxs(logDiffs.data(), logDiffsAbovePt5);
// 		// // PRINT_VAR(ar::to_string(logDiffsValsAbovePt5));

// 		// auto patternValsAbove0 = at_idxs(pattern.data(), patternNonzeroIdxs);
// 		// // PRINT_VAR(ar::to_string(patternValsAbove0));


// 		// PRINT_VAR(score);

// 		// compute random odds
// 		double randomOdds = pattern.sum() * p0_blur * k;

// 		// compute nearest enemy odds
// 		double nextWindowOdds = kNegativeInf;
// 		if (k < candidates.size()) {
// 			auto nextIdx = candidates[k];
// 			FArray nextWindow = FblurAr.middleCols(nextIdx, windowLen);
// 			nextWindowOdds = (nextWindow * pattern).sum();
// 			// nextWindowOdds = (Phi.middleCols(nextIdx, windowLen).array()
// 				// * pattern).sum();
// 		}

// 		double penalty = max(randomOdds, nextWindowOdds);
// 		double adjustedScore = score - penalty;



// 		if (adjustedScore > bestScore) {
// 			printf("================================ %lld\n", seed);
// 			PRINT_VAR(bestScore);
// 			PRINT_VAR(ar::to_string(candidates));

// 			bestScore = adjustedScore;

// 			PRINT_VAR(bestScore);

// 			// TODO does direct copy help? nope, still garbage
// 			// bestPattern = pattern;
// 			ar::copy(pattern.data(), pattern.size(), bestPattern.data());

// 			// bestNumInstances = k;
// 			// bestInstances = at_idxs(candidates, range(k));
// 			bestInstances = candidates;
// 			bestInstances.resize(k);
// 			PRINT_VAR(bestScore);
// 			PRINT_VAR(ar::to_string(bestInstances));
// 			auto candidatesStr = ar::to_string(candidates);
// 			PRINT_VAR(candidatesStr);

// 			PRINT_VAR(bestPattern.sum());
// 		}
// 	}
// }

void instancesForSeed(const FMatrix& Phi, const FMatrix& Phi_blur,
	length_t seed, length_t Lmin, length_t Lmax, length_t windowLen,
	const FMatrix& logs_0, double p0_blur,
	double& bestScore, FArray& bestPattern, vector<length_t>& bestInstances)
{
	printf("================================ %lld\n", seed);

	auto candidates = candidatesForSeed(Phi, Phi_blur, seed,
		Lmin, Lmax, windowLen);

//	int bestNumInstances = -1;

	auto J = logs_0.size();
	assert(J == Phi.rows() * windowLen);

	FArray pattern(Phi.rows(), windowLen);
	FArray FblurAr(Phi_blur.array());

	static const double kTinyConst = .0001;
	length_t idx = candidates[0];
	FArray counts = Phi.middleCols(idx, windowLen).array();
	FArray counts_blur = Phi_blur.middleCols(idx, windowLen).array() + kTinyConst;

	for (int i = 1; i < candidates.size(); i++) {
		auto idx = candidates[i];
		int k = i + 1;

		auto phiPtr = Phi.col(idx).data();
		auto phiBlurPtr = Phi_blur.col(idx).data();
		auto countsPtr = counts.data();
		auto countsBlurPtr = counts_blur.data();
		auto logs0Ptr = logs_0.data();
		auto patternPtr = pattern.data();

		bool hasNext = k < candidates.size();
		auto nextPtr = Phi_blur.data();
		if (hasNext) {
			nextPtr += candidates[k]; // next window
		}

		double score = 0;
		double enemyScore = 0;
		double randomScore = 0;
		for (int j = 0; j < J; j++) {
			countsPtr[j] += phiPtr[j];
			double count = countsPtr[j];
			countsBlurPtr[j] += phiBlurPtr[j];

			double theta_1 = countsBlurPtr[j] / k;
			double log_1 = log(theta_1);
			double log_0 = logs0Ptr[j];
			double logDiff = log_1 - log_0;
			bool mask = (log_1 > log_0) && (theta_1 > .5);
			logDiff = mask ? logDiff : 0;

			patternPtr[j] = logDiff;
			score += logDiff * count;
			if (hasNext) {
				enemyScore += logDiff * nextPtr[j];
			}
			randomScore += logDiff;
		}
		randomScore *= k * p0_blur;

		double penalty = max(randomScore, enemyScore);
		double adjustedScore = score - penalty;

		auto candStr = ar::to_string(candidates.data(), k);
		candStr += " -> ";
		auto scoresStr = string_with_format("%.2f (%.2f, %.2f, %.2f)",
									   adjustedScore, score, randomScore, enemyScore);
		candStr += scoresStr;
		PRINT_VAR(candStr);

		if (adjustedScore > bestScore) {
			bestScore = adjustedScore;
			bestPattern = pattern;
			// store best instances
			bestInstances.resize(k);
			for (int i = 0; i < k; i++) {
				bestInstances[i] = candidates[i];
			}

			bool rowmajor = false;
			ar::imshow(bestPattern.data(), bestPattern.rows(),
				bestPattern.cols(), "bestPattern", rowmajor);
			ar::imshow(Phi.col(candidates[0]).data(), bestPattern.rows(),
				bestPattern.cols(), "firstCandidate", rowmajor);
		}

		assert(score >= 0);
		assert(!Phi.IsRowMajor);
		assert(!Phi_blur.IsRowMajor);
		assert(!logs_0.IsRowMajor);
		assert(ar::all_nonnegative(pattern.data(), J));
		assert(ar::all_nonnegative(counts.data(), J));
		assert(ar::all_nonnegative(counts_blur.data(), J));
		assert(counts.sum() > 0);
		assert(counts_blur.sum() > 0);
//		assert(pattern.sum() > 0); // not true if theta_1 < .5
	}

//	//
//	bestInstances.resize(k);
//	for (int i = 0; i < bestNumInstances; i++) {
//		bestInstances[i] = candidates[i];
//	}
}

vector<length_t> findPatternInstances(const FMatrix Phi, const FMatrix Phi_blur,
	vector<length_t> seeds, length_t Lmin, length_t Lmax, length_t windowLen,
									  	FArray& bestPattern)
//	ArrayXXd& bestPattern)
{
	// precompute stats about Phi and Phi_blur
	// auto windowLen = computeWindowLen(Lmin, Lmax);
	// ArrayXXd logs_0(Phi.rows(), windowLen);
	// ArrayXd rowMeans = Phi_blur.rowwise().mean().array();
	FArray logs_0(Phi.rows(), windowLen);
	ArrayXd rowMeans = Phi_blur.rowwise().mean();
	logs_0.colwise() = rowMeans.array().log().eval();
	double p0_blur = Phi_blur.mean();

	assert( logs_0(0, 0) == log(rowMeans(0)) );
	assert(!logs_0.IsRowMajor);

	print("--------- findPatternInstances() ");

	PRINT_VAR(Phi.sum());
	PRINT_VAR(Phi_blur.sum());
	// PRINT_VAR(ar::to_string(rowMeans.data(), rowMeans.size()));
	// std::cout << "logs_0" << logs_0 << "\n";
	// PRINT_VAR(p0_blur);

	// for(int i = 0; i < Phi.rows(); i++) {
	// 	// auto rowInPtr = Phi.row(i).data();
	// 	auto minVal = Phi.row(i).minCoeff();
	// 	auto maxVal = Phi.row(i).maxCoeff();
	// 	printf("val range for Phi row %d:\t%g-%g;\t", i, minVal, maxVal);

	// 	minVal = Phi_blur.row(i).minCoeff();
	// 	maxVal = Phi_blur.row(i).maxCoeff();
	// 	printf("%g-%g\n", minVal, maxVal);
	// }

	// PRINT_VAR(seeds.size());
	// PRINT_VAR(ar::to_string(seeds));
	// PRINT_VAR(bestPattern.sum());

	double bestScore = kNegativeInf;
	vector<length_t> bestInstances;
	for (auto seed : seeds) {
		instancesForSeed(Phi, Phi_blur, seed, Lmin, Lmax, windowLen,
			logs_0, p0_blur, 					// feature matrix stats
			bestScore, bestPattern, bestInstances); // output args
	}
	sort(bestInstances);

	return bestInstances;
}

std::pair<vector<length_t>, vector<length_t> > extractTrueInstances(
	const FMatrix& Phi, const FMatrix& Phi_blur,
	const vector<length_t>& bestStartIdxs, const FArray& pattern,
	length_t Lmin, length_t Lmax, length_t windowLen)
{
	// init return values
	auto sortedIdxs = sort(bestStartIdxs);
	vector<length_t> startIdxs(sortedIdxs);
	vector<length_t> endIdxs(sortedIdxs);

	double finalPatternSum = pattern.sum();
	PRINT_VAR(finalPatternSum);

	// compute column sums (adjusted by expected sum)
	ArrayXd sums = pattern.colwise().sum();
	assert(sums.size() == windowLen);
	auto k = bestStartIdxs.size();
	double p0 = Phi_blur.mean();
	double expectedOnesFrac = pow(p0, k-1);
	double expectedOnesPerCol = expectedOnesFrac * Phi.rows();
	sums -= expectedOnesPerCol;

	PRINT_VAR(p0);
	PRINT_VAR(expectedOnesFrac);
	PRINT_VAR(expectedOnesPerCol);
	PRINT_VAR(ar::to_string(sums.data(), sums.size()));

	// find best start and end offsets of pattern within returned windows
	length_t start = -1, end = -1; // will be set by max_subarray
	length_t minSpacing = Lmin;
	maximum_subarray(sums.data(), sums.size(), start, end, minSpacing);

	PRINT_VAR(ar::to_string(sortedIdxs));
	PRINT_VAR(ar::to_string(startIdxs));
	PRINT_VAR(ar::to_string(endIdxs));
	PRINT_VAR(start);
	PRINT_VAR(end);

	// ensure pattern length is >= Lmin; greedily expand the range if not
	while ((end - start) < Lmin) {
		print("expanding pattern because shorter than Lmin");
		auto nextStartVal = start > 0 ? sums(start - 1) : kNegativeInf;
		auto nextEndVal = end < windowLen ? sums(end) : kNegativeInf;
		if (nextStartVal > nextEndVal) {
			start--;
		} else {
			end++;
		}
	}
	// ensure pattern length is <= Lmax; greedily narrow the range if not
	while ((end - start) > Lmax) {
		print("truncating pattern because longer than Lmax");
		if (sums(start) > sums(end-1)) {
			end--;
		} else {
			start++;
		}
	}

	PRINT_VAR(start);
	PRINT_VAR(end);

	finalPatternSum = pattern.sum();
	PRINT_VAR(finalPatternSum);

	add_inplace(startIdxs, start);
	add_inplace(endIdxs, end);
	PRINT_VAR(ar::to_string(startIdxs));
	PRINT_VAR(ar::to_string(endIdxs));
	return std::pair<vector<length_t>, vector<length_t> >(startIdxs, endIdxs);
}

std::pair<vector<length_t>, vector<length_t> > findPattern(const CMatrix& T,
	const FMatrix& Phi, const FMatrix& Phi_blur, ArrayXXd& pattern,
	const vector<length_t>& seeds,
	length_t Lmin, length_t Lmax, length_t windowLen)
{
	// auto scores = seedScores(T, Lmin, Lmax);
	// auto seeds = selectSeeds(scores.data(), scores.size(),
		// Lmin, Lmax, windowLen);

	double initialPatternSum = pattern.sum();
	PRINT_VAR(initialPatternSum);

	// length_t windowLen = computeWindowLen(Lmin, Lmax);
	// ArrayXXd pattern(Phi.rows(), windowLen); // set by func below
	auto starts = findPatternInstances(Phi, Phi_blur, seeds, Lmin, Lmax,
		windowLen, pattern);

	double finalPatternSum = pattern.sum();
	PRINT_VAR(finalPatternSum);

	PRINT_VAR(ar::to_string(starts));

	return extractTrueInstances(Phi, Phi_blur, starts, pattern,
		Lmin, Lmax, windowLen);
}

void FlockLearner::learn(const double* X, const int d, const int n,
	double m_min, double m_max, double m_filt)
{
	PRINT_VAR(d);
	PRINT_VAR(n);

	_T = eigenWrap2D_nocopy_const(X, d, n);

	for (int dim = 0; dim < d; dim++) {
		auto rowPtr = _T.row(dim).data();
		auto downsampled = ar::resample(rowPtr, n, 80);
		// auto downsampled = ar::resample(rowPtr, n, n);
		plot_array(downsampled.get(), 80, -1, .5, "initial signal");
		PRINT_VAR(ar::to_string(downsampled.get(), 80));
	}

	if (m_min < 1) {
		m_min = m_min * n;
	}
	if (m_max < 1) {
		m_max = m_max * n;
	}
	if (m_filt <= 0) {
		m_filt = m_min;
	} else if (m_filt < 1) {
		m_filt = m_filt * n;
	}
	_Lmin = m_min;
	_Lmax = m_max;
	_Lfilt = m_filt;

	_windowLen = computeWindowLen(_Lmin, _Lmax);

	cputime_t t0 = timeNow();

	print("------------------------ building feature mats");
	auto mats = buildShapeFeatureMats(X, d, n, _Lmin, _Lmax, _Lfilt);
	_Phi = mats.first;
	_Phi_blur = mats.second;
	PRINT_VAR(_Lmin);
	PRINT_VAR(_Lmax);
	PRINT_VAR(_Lfilt);
	PRINT_VAR(_Phi.rows());
	PRINT_VAR(_windowLen);
	PRINT_VAR(_Phi.cols());
	_pattern.resize(_Phi.rows(), _windowLen);

	bool rowmajor = false;
	// bool twoEnded = false;
	ar::imshow(_Phi.data(), _Phi.rows(), _Phi.cols(), "Phi", rowmajor);
	ar::imshow(_Phi_blur.data(), _Phi_blur.rows(), _Phi_blur.cols(), "Phi_blur", rowmajor);
//	return;

	// TODO remove unneeded vars
	print("------------------------ computing seed scores for debug");
	_seedScores = seedScores(_T, _Lmin, _Lmax);
	_seeds = selectSeeds(_seedScores.data(), _seedScores.size(),
		_Lmin, _Lmax, _windowLen);

	// TODO remove -- are we even writing to this obj?
	// apparently; gets set to basically all 0s for some reason...
	_pattern.setRandom();

	cputime_t t1 = timeNow();

	print("------------------------ learning patttern");

	// auto scores_plot1 = ar::resample(_seedScores, 80);
	// plot_array(scores_plot1.data(), 80, -1, .25, "prefindPattern seedScores");

	// auto startsAndEnds = findPattern(_T, _Phi, _Phi_blur, patternWrapper,
	auto startsAndEnds = findPattern(_T, _Phi, _Phi_blur, _pattern,
		_seeds, _Lmin, _Lmax, _windowLen);
	// _pattern = patternWrapper.matrix();
	_startIdxs = startsAndEnds.first;
	_endIdxs = startsAndEnds.second;

	PRINT_VAR(ar::to_string(_startIdxs));
	PRINT_VAR(ar::to_string(_endIdxs));

	double finalPatternSum = _pattern.sum();
	PRINT_VAR(finalPatternSum);

	cputime_t t2 = timeNow();

	auto featureDuration = durationMs(t1, t0);
	auto searchDuration = durationMs(t2, t1);

	// TODO remove after debug
	// okay, so we're successfully sending python the pattern; the pattern
	// just happens to be garbage
	// _pattern.setRandom();
	// for (auto i : ar::exprange(2, 16)) {
	// 	_pattern.col(i).setConstant(i);
	// }

	// auto scores_plot = ar::resample(_seedScores, 80);
	// plot_array(scores_plot.data(), 80, -1, .25, "findPattern seedScores");

	std::cout << "feature duration, search duration (ms): " << featureDuration
		<< ", " << searchDuration << "\n";
}

// ================================================================ public class

// vector<length_t> FlockLearner::getInstanceStartIdxs() { return _startIdxs; }
vector<length_t> FlockLearner::getInstanceStartIdxs() const {
	PRINT_VAR(ar::to_string(_startIdxs));
	return _startIdxs;
}
// vector<length_t> FlockLearner::getInstanceEndIdxs() { return _endIdxs; }
vector<length_t> FlockLearner::getInstanceEndIdxs() const {
	PRINT_VAR(ar::to_string(_endIdxs));
	return _endIdxs;
}

// TODO remove
double FlockLearner::getPatternSum() { return _pattern.sum(); }
vector<double> FlockLearner::getSeedScores() { return _seedScores; }
vector<length_t> FlockLearner::getSeeds() { return _seeds; }

vector<length_t> FlockLearner::getCandidatesForSeed(FMatrix Phi, FMatrix Phi_blur,
								   int seed) {
	return candidatesForSeed(Phi, Phi_blur, seed, _Lmin, _Lmax, _windowLen);
}
