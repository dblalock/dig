

#include "flock.hpp"

#include <vector>

#include "Dense"

// #include "type_defs.h"
#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "pimpl_impl.hpp"
#include "subseq.hpp"

#include "shape_features.hpp"

// using Eigen::Array;
using Eigen::Matrix;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::vector;

using ar::add_inplace;
using ar::argsort;
using ar::at_idxs;
using ar::exprange;
using ar::filter;
using ar::max;
using ar::min;
using ar::normalize_max_inplace;
using ar::rand_idxs;
using ar::range;
using ar::sort;
using ar::unique;

using subs::local_maxima;
using subs::maximum_subarray;

// ================================================================ Constants

static constexpr double kNegativeInf = -std::numeric_limits<double>::infinity();

// ================================================================ funcs

vector<double> seedScores(CMatrix T, length_t Lmin, length_t Lmax) {
	length_t d = T.rows();
	length_t n = T.cols();
	vector<double> scores(n);
	double* out = scores.data();

	auto lengths = exprange(8, max(Lmax, 16) + 1);

	double scores_tmp[n];
	for (int dim = 0; dim < d; dim++) {
		for (auto subseqLen : lengths) {
			double* signal = T.row(dim).data();
			auto walks = createRandWalks(signal, n, subseqLen);
			structureScores1D(signal, n, subseqLen, walks, scores_tmp);

			// scale scores to max of 1 in place, then add to combined scores
			normalize_max_inplace(scores_tmp, n);
			add_inplace(out, scores_tmp, n);
		}
	}
	return scores;
}

static inline length_t computeWindowLen(length_t Lmin, length_t Lmax) {
	return Lmax + (Lmax / 10); // Lmax/10 = step size of seed generation
}

vector<length_t> selectSeeds(double* scores, length_t n,
	length_t Lmin, length_t Lmax)
{
	length_t howMany = 1;
	bool replace = false;
	length_t seed1 = rand_idxs(n, howMany, replace, scores)[0];

	// make scores around this seed negative so we don't select
	// any of them as the second seed
	auto start = scores + max(0, seed1 - Lmin);
	auto length = min(Lmin, n - 1 - seed1);
	constant_inplace(start, length, -99);

	length_t seed2 = rand_idxs(n, howMany, replace, scores)[0];

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
	auto windowLen = computeWindowLen(Lmin, Lmax);
	auto maxIdx = n - windowLen;
	return filter([maxIdx](length_t seed) {
		return 0 <= seed && seed <= maxIdx;
	}, uniqSeeds);
}

// returns all candidates in decreasing order of dot product
vector<length_t> candidatesForSeed(FMatrix Phi, FMatrix Phi_blur,
	length_t seed, length_t Lmin, length_t Lmax)
{
	auto windowLen = computeWindowLen(Lmin, Lmax);
	length_t numSubseqs = Phi.cols() - windowLen + 1;

	ArrayXd seedWindow = Phi.middleCols(seed, windowLen).array();
	vector<double> dotProds(numSubseqs);
	for (int i = 0; i < numSubseqs; i++) {
		dotProds[i] = (seedWindow * Phi.middleCols(i, windowLen).array()).sum();
	}
	length_t minSpacing = Lmin;
	auto maxima = local_maxima(dotProds, minSpacing);
	auto maximaVals = at_idxs(dotProds, maxima);

	bool ascending = false;
	auto sortIdxs = argsort(maximaVals, ascending);
	return at_idxs(maxima, sortIdxs);
}

void instancesForSeed(FMatrix Phi, FMatrix Phi_blur,
	length_t seed, length_t Lmin, length_t Lmax,
	const ArrayXd& logs_0, double p0_blur,
	double& bestScore, ArrayXd& bestPattern, vector<length_t>& bestInstances)
{
	auto windowLen = computeWindowLen(Lmin, Lmax);
	auto candidates = candidatesForSeed(Phi, Phi_blur, seed, Lmin, Lmax);

	// initialize set of candidates and counts with best candidate; we add
	// a tiny constant to the counts to that we don't get -inf logs from zeros
	static const double kTinyConst = .00001;
	length_t idx = candidates[0];
	vector<length_t> bestCandidates;
	bestCandidates.push_back(idx);
	ArrayXd counts = Phi.middleCols(idx, windowLen).array() + kTinyConst;
	ArrayXd counts_blur = Phi_blur.middleCols(idx, windowLen).array() + kTinyConst;

	static const double LOG_0p5 = -.6941471806; // ln(.5)

	ArrayXd pattern(Phi.rows(), windowLen);
	// int bestNumInstances = 1;
	for (int i = 1; i < candidates.size(); i++) {
		auto idx = candidates[i];
		int k = i + 1;
		counts_blur += Phi_blur.middleCols(idx, windowLen).array();
		// here are the steps we notionally compute; we combine them into
		// two expressions below to ensure that we make use of Eigen
		// expression templates
		// ArrayXd theta_1 = (counts_blur / k);
		// ArrayXd logs_1 = theta_1.log();
		// ArrayXd logDiffs = (logs_1 - logs_0);
		// ArrayXd pattern = logDiffs * (logDiffs > log_diffs_threshs);

		ArrayXd logDiffs = ((counts_blur / k).log() - logs_0);
		pattern = (logDiffs * (logDiffs > logs_0.max(LOG_0p5)).cast<double>()).eval();

		counts += Phi.middleCols(idx, windowLen).array();
		double score = (counts * pattern).sum();

		// compute random odds
		double randomOdds = pattern.sum() * p0_blur * k;

		// compute nearest enemy odds
		double nextWindowOdds = kNegativeInf;
		if (k < candidates.size()) {
			auto nextIdx = candidates[k];
			nextWindowOdds = (Phi.middleCols(nextIdx, windowLen).array() * pattern).sum();
		}

		double penalty = max(randomOdds, nextWindowOdds);
		score -= penalty;

		if (score > bestScore) {
			bestScore = score;
			bestPattern = pattern;
			// bestNumInstances = k;
			bestInstances = at_idxs(candidates, range(k));
		}
	}
}

vector<length_t> findPatternInstances(FMatrix Phi, FMatrix Phi_blur,
	vector<length_t> seeds, length_t Lmin, length_t Lmax, ArrayXd& bestPattern)
{
	// precompute stats about Phi and Phi_blur
	auto windowLen = computeWindowLen(Lmin, Lmax);
	ArrayXd logs_0(Phi.rows(), windowLen);
	VectorXd rowMeans = Phi_blur.rowwise().mean();
	logs_0.colwise() = rowMeans.array();
	double p0_blur = Phi_blur.mean();

	double bestScore = kNegativeInf;
	vector<length_t> bestInstances;
	for (auto seed : seeds) {
		instancesForSeed(Phi, Phi_blur, seed, Lmin, Lmax,
			logs_0, p0_blur, 					// feature matrix stats
			bestScore, bestPattern, bestInstances); // output args
	}
	return bestInstances;
}

std::pair<vector<length_t>, vector<length_t> > extractTrueInstances(
	const FMatrix& Phi, const FMatrix& Phi_blur,
	const vector<length_t>& bestStartIdxs, const ArrayXd& pattern,
	length_t Lmin, length_t Lmax, length_t windowLen)
{
	// init return values
	auto sortedIdxs = sort(bestStartIdxs);
	vector<length_t> startIdxs(sortedIdxs);
	vector<length_t> endIdxs(sortedIdxs);

	// compute column sums (adjusted by expected sum)
	ArrayXd sums = pattern.colwise().sum();
	assert(sums.size() == windowLen);
	auto kBest = bestStartIdxs.size();
	double p0 = Phi_blur.mean();
	double expectedOnesFrac = pow(p0, kBest-1);
	double expectedOnesPerCol = expectedOnesFrac * Phi.rows();
	sums = sums - expectedOnesPerCol;

	// find best start and end offsets of pattern within returned windows
	length_t start, end;
	length_t minSpacing = Lmin;
	maximum_subarray(sums.data(), sums.size(), start, end, minSpacing);

	// ensure pattern length is >= Lmin; greedily expand the range if not
	while ((end - start) < Lmin) {
		auto nextStartVal = start > 0 ? sums[start - 1] : kNegativeInf;
		auto nextEndVal = end < windowLen ? sums[end] : kNegativeInf;
		if (nextStartVal > nextEndVal) {
			start--;
		} else {
			end++;
		}
	}
	// ensure pattern length is <= Lmax; greedily narrow the range if not
	while ((end - start) > Lmax) {
		if (sums[start] > sums[end-1]) {
			end--;
		} else {
			start++;
		}
	}

	add_inplace(startIdxs, start);
	add_inplace(endIdxs, end);
	return std::pair<vector<length_t>, vector<length_t> >(startIdxs, endIdxs);
}

std::pair<vector<length_t>, vector<length_t> > findPattern(CMatrix T,
	FMatrix Phi, FMatrix Phi_blur, length_t Lmin, length_t Lmax)
{
	auto scores = seedScores(T, Lmin, Lmax);
	auto seeds = selectSeeds(scores.data(), scores.size(), Lmin, Lmax);

	length_t windowLen = computeWindowLen(Lmin, Lmax);
	ArrayXd pattern(Phi.rows(), windowLen); // set by func below
	auto starts = findPatternInstances(Phi, Phi_blur, seeds, Lmin, Lmax,
		pattern);

	return extractTrueInstances(Phi, Phi_blur, starts, pattern,
		Lmin, Lmax, windowLen);

}


// ================================================================ pimpl

// typedef double data_t;

class FlockLearner::Impl {
//private:
public:
	CMatrix _T;
	FMatrix _Phi;
	FMatrix _Phi_blur;
	ArrayXd _pattern;
	vector<length_t> _startIdxs;
	vector<length_t> _endIdxs;
	length_t _Lmin;
	length_t _Lmax;
	length_t _Lfilt;

//public:
	// ------------------------ ctors
	Impl(const double* X, const int d, const int n, double m_min, double m_max,
		 double m_filt):
		_T(eigenWrap2D_nocopy_const(X, d, n))
	{
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

		auto windowLen = computeWindowLen(_Lmin, _Lmax);
		auto mats = buildShapeFeatureMats(X, d, n, _Lmin, _Lmax, _Lfilt);
		_Phi = mats.first;
		_Phi_blur = mats.second;
		_pattern = ArrayXd(_Phi.rows(), windowLen);

		auto startsAndEnds = findPattern(_T, _Phi, _Phi_blur, _Lmin, _Lmax);
		_startIdxs = startsAndEnds.first;
		_endIdxs = startsAndEnds.second;
	}

//	Impl(const double* X, int d, int n, double m_min, double m_max):
//		Impl(X, d, n, static_cast<int>(m_min * n), static_cast<int>(m_max * n))
//	{}

};

// ================================================================ public class


//template<class T>
//FlockLearner<T>::~FlockLearner() = default;

//FlockLearner::FlockLearner(const double* X, int d, int n, int m_min, int m_max,
//	int m_filt):
//	_self(X, d, n, m_min, m_max, m_filt)
//{}

//FlockLearner::FlockLearner(int foo) {
//	
//}

FlockLearner::FlockLearner(const double* X, int d, int n,
						   double m_min, double m_max, double m_filt):
	_self(X, d, n, m_min, m_max, m_filt)
{}

// TODO remove
//FlockLearner::FlockLearner(const double* X, int d, int n):
//	_self(X, d, n, 0.0,0.0,0.0)
//{}

FMatrix FlockLearner::getFeatureMat() { return _self->_Phi; }
FMatrix FlockLearner::getBlurredFeatureMat() { return _self->_Phi_blur; }
FMatrix FlockLearner::getPattern() { return _self->_pattern; }
vector<length_t> FlockLearner::getInstanceStartIdxs() { return _self->_startIdxs; }
vector<length_t> FlockLearner::getInstanceEndIdxs() { return _self->_endIdxs; }

// template<class T>
// FlockLearner<T>::findPattern()

