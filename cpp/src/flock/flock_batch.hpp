//
//  Flock.hpp
//  Dig
//
//  Created by DB on 3/9/16.
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef DIG_FLOCK_BATCH_HPP
#define DIG_FLOCK_BATCH_HPP

#include "flock.hpp"
#include "circular.hpp"
#include "array_utils.hpp"

using ar::copy;
using ar::min;

template <class T=double, int D=3, int N=1024>
class BatchFlockLearner {
public:
	enum { size = N*D };

	BatchFlockLearner(const BatchFlockLearner& other) = delete;
	BatchFlockLearner& operator=(const BatchFlockLearner&) = delete;

	BatchFlockLearner() {}; // SWIG can't do "= default"

	void push_back(int dim, const T& value) {
		_circ_arrays[dim].push_back(value);
	}

	void learn(int useHistoryLen, double Lmin, double Lmax) {
		useHistoryLen = min(N, useHistoryLen);

		copyToContiguous(useHistoryLen);
		_learner.learn(_ar.data(), D, useHistoryLen, Lmin, Lmax);
	}

	void clear() {
		for (int d = 0; d < D; d++) {
			_circ_arrays[d].clear();
		}
	}

	vector<int64_t> getStartIdxs() { return _learner.getInstanceStartIdxs(); }
	vector<int64_t> getEndIdxs() { return _learner.getInstanceEndIdxs(); }

	const FlockLearner& getLearner() { return _learner; }

	// TODO remove after debug
	void dummyLearn() {
		_learner.dummyLearn();
	}

private:
	void copyToContiguous(int useHistoryLen) {
		assert(useHistoryLen <= N);
		for (int d = 0; d < D; d++) {
			auto buff = _circ_arrays[d];
			auto readFrom = buff.end() - useHistoryLen;
			T* writeTo = _ar.data() + (d * useHistoryLen);
			ar::copy(readFrom, useHistoryLen, writeTo);
		}
	}

	std::array<circular_array<T, N>, D> _circ_arrays;
	std::array<T, size> _ar;
	FlockLearner _learner;

};


#endif // DIG_FLOCK_BATCH_HPP
