//
//  nearest_neighbor.cpp
//  Dig
//
//  Created by DB on 10/22/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include "nearest_neighbor.hpp"
#include "dig.hpp"

#include <math.h>
#include <vector>
#include "dtw.hpp"

// ================================================================
// Classifier
// ================================================================

// class TSClassifier {
// private:
// 	class impl;
// 	std::unique_ptr<impl> _pimpl;
// public:
// 	TSClassifier();
// 	~TSClassifier();

// 	void setAlgorithm(ClassificationAlgorithm algo);
// 	bool addExample(const double* X, int m, int n, int label);
// 	int classify(const double* X, int m, int n);
// };

template <class data_t, class len_t, class dist_t=data_t >
dist_t distance(const data_t* X, const data_t* Y, len_t m, len_t n, ClassificationAlgorithm algo, dist_t thresh=INFINITY) {
	dist_t sum = 0;
	// for (int dim = 0; dim < n; dim++) {	//TODO totals across each dimension
		switch (algo) {
			case NN_L1:
				sum += dist_L1(X,Y, m);
				break;
			case NN_L2:
				sum += dist_L2(X,Y, m);
				break;
			case NN_DTW:
				sum += dist_dtw(X,Y, m, 1);	//TODO don't hardcode warping constraint
				break;
			default:
				return -1;
		}
	// }
	return sum;
}


//TODO overload within DtwDatabase that doesn't take in a vector of examples
template <class data_t, class len_t, class dist_t=double, class label_t=int>
auto findNearestNeighbor(const DTWExample<data_t, len_t, label_t>& query,
	const std::vector<DTWExample<data_t, len_t, label_t>>& examples,
	const DTWTempStorage<data_t, len_t, dist_t>* storage=nullptr)
	-> decltype(query) {

	assertf(examples.size() > 0, "findNearestNeighbor: no examples given");

	if (storage == nullptr) {
		storage = new DTWTempStorage<data_t, len_t, dist_t>(query.getLength());
	}

	double bsf = INFINITY;
	size_t bestIdx = 0;
	for (size_t i = 0; i < examples.size(); i++) {
		auto example = examples[i];
		auto dist = dtw_abandon(query,example,bsf,storage);
		if (dist < bsf) {
			bsf = dist;
			bestIdx = i;
		}
	}
	return examples[bestIdx];
}

// Create a DTWExample from the given query using the warping constraint
// of the 1st example provided. The warping constraint cannot be specified
// since the comparisons will necessarily use the warping constraints of
// each DTWExample in order for their lower bounds to be correct. If you
// want to vary the warping constraint, create new DTWExample instances.
template <class data_t, class len_t, class dist_t=double, class label_t=int>
DTWExample<data_t, len_t, label_t> findNearestNeighbor(const data_t* query, len_t len,
	const std::vector<DTWExample<data_t, len_t, label_t>>& examples,
	const DTWTempStorage<data_t, len_t, dist_t>* storage=NULL) {

	assertf(examples.size() > 0, "findNearestNeighbor: no examples given");

	len_t warp = examples[0].getWarp();
	DTWExample<double, int> q(query, len, warp);
	return findNearestNeighbor(q, examples, storage);
}

class TSClassifier::impl {
	friend class TSClassifier;
private:
	ClassificationAlgorithm _algo;
	// std::vector<std::array<double>> _examples;
	std::vector<std::vector<double>> _examples;		//TODO won't be contiguous...problem?
	std::vector<DTWExample<double, int>> _dtwExamples;
	std::vector<int> _labels;

	int dtw_classify(const double* X, int m, int n) {
		auto nn = findNearestNeighbor(X,m,_dtwExamples);
		return nn.getLabel();

		// TODO this should not ignore the 2nd dimension and intelligently
		// combine the distances from different dimensions

		// TODO if the query is longer than a given example (of length m),
		// have the example only compare to the last m points of the query.
			// this way we can classify things of different lengths on an
			// ongoing basis in a subsequence search
	}
};

TSClassifier::TSClassifier(ClassificationAlgorithm algo): _pimpl{ new impl() } {
	_pimpl->_algo = algo;
}
TSClassifier::~TSClassifier() = default;

void TSClassifier::setAlgorithm(ClassificationAlgorithm algo) {
	printf("set algo to %d\n", algo);
	_pimpl->_algo = algo;
}

void TSClassifier::addExample(const double* X, int m, int n, int label) {
	_pimpl->_examples.push_back(std::vector<double>(X, X + m*n) );
	_pimpl->_labels.push_back(label);

	int warp = 1;	//TODO don't hardcode warping constraint
	_pimpl->_dtwExamples.emplace_back(X,m,warp,label);

	printf("pimpl added example\n");
}

int TSClassifier::classify(const double* X, int m, int n) {
	if (_pimpl->_algo == NN_DTW) {
		return _pimpl->dtw_classify(X,m,n);
	}

	double bsf = INFINITY;
	int label = -1;
	for (int i = 0; i < _pimpl->_examples.size(); i++) {
		auto example = _pimpl->_examples[i];
		auto dist = distance(X, &example[0], m, n, _pimpl->_algo, bsf);
		if (dist < bsf) {
			bsf = dist;
			label = _pimpl->_labels[i];
		}
	}
	return label;
}

