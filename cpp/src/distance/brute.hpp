//
//  neighbors.hpp
//  Dig
//
//  Created by DB on 6/29/14.
//  Copyright (c) 2016 DB. All rights reserved.
//

#include <vector>

#include "Dense"
#include "redsvd.hpp"

#include "array_utils.hpp"
#include "neighbors.hpp"




// TODO just code brute force NeighborDB_L2 here using Eigen and give no
// craps about integrating with array_utils or anything

using std::vector;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::MatrixXi;
using Eigen::ArrayXi;

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;

using ar::argsort;
using ar::map;
using ar::mapi;

typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;

typedef int64_t idx_t;

static const int16_t kInvalidIdx = -1;

// wrapper subclasses to allow only specifying type
// template<class T, int Dim1=Dynamic, int Dim2=Dynamic>
// class MatrixX: public Matrix<T, Dynamic, Dynamic> {};

// template<class T, int Dim1=Dynamic, int Dim2=Dynamic>
// class RowMatrixX: public Matrix<T, Dim1, Dim2, RowMajor> {};

// ------------------------------------------------ brute force distances

template<class MatrixT, class VectorT>
VectorT squaredDistsToVector(const MatrixT& X, const VectorT& v) {
	// auto diffs = X.rowwise() - v.transpose();
	// return diffs.rowwise().squaredNorm();
	return (X.rowwise() - v.transpose()).rowwise().squaredNorm();
}
template<class MatrixT, class VectorT>
VectorT distsToVector(const MatrixT& X, const VectorT& v) {
	return squaredDistsToVector(X, v).array().sqrt().matrix();
}

template<int RetStorageOrder=ColMajor, class MatrixT=MatrixXd, class VectorT=VectorXd>
/** compute distances between rows of X and rows of V */
auto squaredDistsToVectors(const MatrixT& X, const MatrixT& V,
	VectorT rowSquaredNorms) -> Matrix<decltype(
		std::declval<typename MatrixT::Scalar>() * std::declval<typename VectorT::Scalar>()),
		Dynamic, Dynamic, RetStorageOrder>
{
	// create a matrix of appropriate type; this way dists for each vector
	// can be contiguous (if ColumnMajor storage)
	Matrix<decltype(
		std::declval<typename MatrixT::Scalar>() * std::declval<typename VectorT::Scalar>()),
		Dynamic, Dynamic, RetStorageOrder>
		dists = -2. * (X * V.transpose());
	// MatrixT dists = -2. * (X * V.transpose());
	auto colSquaredNorms = V.rowwise().squaredNorm().eval();
	auto colSquaredNormsAsRow = colSquaredNorms.transpose().eval();
	dists.colwise() += rowSquaredNorms;
	dists.rowwise() += colSquaredNormsAsRow;
	// TODO profile and see if for loop that touches each element once is faster

	return dists;
}
template<class MatrixT>
/** compute distances between rows of X and rows of V */
MatrixT squaredDistsToVectors(const MatrixT& X, const MatrixT& V) {
	Matrix<typename MatrixT::Scalar, Dynamic, 1> rowSquaredNorms =
		X.rowwise().squaredNorm().eval();
	return squaredDistsToVectors(X, V, rowSquaredNorms);
}

template<class MatrixT>
MatrixT distsToVectors(const MatrixT& X, const MatrixT& V) {
	return squaredDistsToVectors(X, V).array().sqrt().matrix();
}


// ------------------------------------------------ neighbor munging

template<template<class...> class Container>
void sort_neighbors_ascending_distance(const Container<Neighbor>& neighbors) {
	std::sort(std::begin(neighbors), std::end(neighbors),
	    [](const Neighbor& a, const Neighbor& b) -> bool
		{
			return a.dist < b.dist;
		}
	);
}

/** given a sorted collection of the best neighbors found so far, (potentially)
 * inserts a new neighbor into the list such that the sorting is preserved */
template<template<class...> class Container>
inline void insert_neighbor(Container<Neighbor> neighbors_bsf, Neighbor newNeighbor) {
	size_t i = neighbors_bsf.size() - 1;
	auto dist = newNeighbor.dist;

	if (dist < neighbors_bsf[i].dist) {
		neighbors_bsf[i] = newNeighbor;
	}

	while (i > 0 && neighbors_bsf[i-1].dist > dist) {
		// swap new and previous neighbor
		Neighbor tmp = neighbors_bsf[i-1];
		neighbors_bsf[i-1] = neighbors_bsf[i];
		neighbors_bsf[i] = tmp;
		i--;
	}
}
template<template<class...> class Container>
inline void insert_neighbor(Container<Neighbor> neighbors_bsf, double dist,
	int idx)
{
	return insert_neighbor(neighbors_bsf, {.dist = dist, .idx = idx});
}


template<class T>
vector<Neighbor> knn_from_dists(const T* dists, size_t len, size_t k) {
	k = ar::min(k, len);
	vector<Neighbor> ret(k); // warning: populates it with k 0s
	for (idx_t i = 0; i < k; i++) {
		ret[i] = Neighbor{.dist = dists[i], .idx = i};
	}
	sort_neighbors_ascending_distance(ret);
	for (idx_t i = k; i < len; i++) {
		insert_neighbor(ret, dists[i], i);
	}
	return ret;
}

template<class T, class R>
vector<Neighbor> neighbors_in_radius(const T* dists, size_t len, R radius_sq) {
	vector<Neighbor> neighbors;
	for (idx_t i = 0; i < len; i++) {
		auto dist = dists(i);
		if (dists(i) <= radius_sq) {
			neighbors.emplace_back(dist, i);
		}
	}
	return neighbors;
}

vector<vector<idx_t> > idxs_from_nested_neighbors(
	const vector<vector<Neighbor> >& neighbors)
{
	return map([](const vector<Neighbor>& inner_neighbors) -> std::vector<idx_t> {
		return map([](const Neighbor& n) -> idx_t {
			return n.idx;
		}, inner_neighbors);
	}, neighbors);
}

// ------------------------------------------------ brute force search

namespace brute {

	// ------------------------ single query

	template<class RowMatrixT, class VectorT>
	inline vector<Neighbor> radius(const RowMatrixT& X, const VectorT& query,
										 float radius_sq)
	{
		VectorT dists = squaredDistsToVector(X, query);
		return neighbors_in_radius(dists.data(), dists.size(), radius_sq);
	}

	template<class RowMatrixT, class VectorT>
	inline Neighbor onenn(const RowMatrixT& X, const VectorT& query) {
		typename RowMatrixT::Index idx;
		double dist = (X.rowwise() - query.transpose()).rowwise().squaredNorm().minCoeff(&idx);
		return {.dist = dist, .idx = static_cast<int32_t>(idx)}; // TODO fix length_t
	}

	template<class RowMatrixT, class VectorT>
	vector<Neighbor> knn(const RowMatrixT& X, const VectorT& query,
							   size_t k)
	{
		VectorT dists = squaredDistsToVector(X, query);
		return knn_from_dists(dists.data(), dists.size(), k);
	}

	// ------------------------ batch of queries

	template<class RowMatrixT, class ColVectorT>
	inline vector<vector<Neighbor> > radius_batch(const RowMatrixT& X,
		const RowMatrixT& queries, float radius_sq, const ColVectorT& rowNorms)
	{
		auto dists = squaredDistsToVectors(X, queries, rowNorms);
		assert(queries.rows() == dists.cols());
		auto num_queries = queries.rows();

		vector<vector<Neighbor> > ret;
		for (idx_t j = 0; j < num_queries; j++) {
			ret.emplace_back(neighbors_in_radius(dists.data(),
												 dists.size(), radius_sq));
		}
		return ret;
	}

	template<class RowMatrixT, class ColVectorT>
	inline vector<vector<Neighbor> > knn_batch(const RowMatrixT& X,
		const RowMatrixT& queries, size_t k, const ColVectorT& rowNorms)
	{
		auto dists = squaredDistsToVectors(X, queries, rowNorms);
		assert(queries.rows() == dists.cols());
		auto num_queries = queries.rows();
		vector<vector<Neighbor> > ret;
		for (idx_t j = 0; j < num_queries; j++) {
			ret.emplace_back(knn_from_dists(dists.col(j).data(),
											dists.rows(), k));
		}
		return ret;
	}

	template<class RowMatrixT, class ColVectorT>
	inline vector<Neighbor> onenn_batch(const RowMatrixT& X,
		const RowMatrixT& queries, const ColVectorT& rowNorms)
	{
		auto wrapped_neighbors = knn_batch(X, queries, 1, rowNorms);
		return ar::map([](const vector<Neighbor> el) {
			return el[0];
		}, wrapped_neighbors);
	}

} // namespace brute

// superclass using Curiously Recurring Template Pattern to get compile-time
// polymorphism; i.e., we can call arbitrary child class functions without
// having to declare them anywhere in this class. This is the same pattern
// used by Eigen.
template<class Derived, class RowMatrixT>
class IndexBase {
public:
	// ------------------------------------------------ return only idxs

	template<class VectorT>
	vector<idx_t> radius_idxs(const VectorT& query, float radius_sq) {
		auto neighbors = Derived::radius(query, radius_sq);
		return ar::map([](const Neighbor& n) {
			return n.idx;
		}, neighbors);
	}

	template<class VectorT>
	idx_t onenn_idxs(const VectorT& query) {
		auto neighbor = Derived::onenn(query);
		return query.idx;
	}

	template<class VectorT>
	vector<idx_t> knn_idxs(const VectorT& query, size_t k) {
		auto neighbors = Derived::knn(query, k);
		// TODO have neighbor idx be idx_t; we might overflow 32 bits so
		// needs to be 64
		return map([](const Neighbor& n) { return static_cast<idx_t>(n.idx); },
				   neighbors);
	}

	// ------------------------------------------------ batch return only idxs

	vector<vector<idx_t> > radius_batch_idxs(const RowMatrixT& queries,
											 float radius_sq)
	{
		auto neighbors = Derived::radius_batch(queries, radius_sq);
		return idxs_from_nested_neighbors(neighbors);
	}

	vector<idx_t> onenn_batch_idxs(const RowMatrixT& queries) {
		auto neighbors = Derived::onenn_batch(queries);
		return map([](const Neighbor& n) {
			return n.idx;
		}, neighbors);
	}

	vector<vector<idx_t> > knn_batch_idxs(const RowMatrixT& queries, size_t k) {
		auto neighbors = Derived::knn_batch(queries, k);
		return _idxs_from_nested_neighbors(neighbors);
	}
};

// ------------------------------------------------ L2IndexBrute

template<class RowMatrixT>
class L2IndexBrute: public IndexBase<L2IndexBrute<RowMatrixT>, RowMatrixT> {
public:

	typedef Matrix<typename RowMatrixT::Scalar, Dynamic, 1> ColVectorT;

	explicit L2IndexBrute(const RowMatrixT& data):
		_data(data)
	{
		_rowNorms = data.rowwise().squaredNorm();
	}

	// ------------------------------------------------ single query

	template<class VectorT>
	vector<Neighbor> radius(const VectorT& query, float radius_sq) {
		return brute::radius(_data, query, radius_sq);
	}

	template<class VectorT>
	Neighbor onenn(const VectorT& query) {
		return brute::onenn(_data, query);
	}

	template<class VectorT>
	vector<Neighbor> knn(const VectorT& query, size_t k) {
		return brute::knn(_data, query, k);
	}

	// ------------------------------------------------ batch of queries

	vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
		float radius_sq)
	{
		return radius_batch(_data, queries, radius_sq, _rowNorms);
	}

	vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, size_t k) {
		return knn_batch(_data, queries, k, _rowNorms);
	}

	vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
		return onenn_batch(_data, queries, _rowNorms);
	}

private:
	RowMatrixT _data;
	ColVectorT _rowNorms;
};

// ------------------------------------------------ early abandoning search
// TODO DRY out these funcs

namespace abandon {

// ================================ distance functions

template<int BlockSize=8, class VectorT1, class VectorT2, class dist_t>
dist_t dist_sq(const VectorT1& x, const VectorT2& y, dist_t thresh) {
	// typedef decltype(x.data()[0] * y.data()[0]) dist_t;
	auto n = x.size();
	dist_t dist = 0;
	int32_t i = 0;
	for (; i < (n - BlockSize) ; i += BlockSize) {
		dist += (x.segment<BlockSize>(i) - y.segment<BlockSize>(i)).squaredNorm();
		if (dist > thresh) {
			return dist;
		}
	}
	for (; i < n; i++) {
		auto diff = x(i) - y(i);
		dist += diff * diff;
	}
}

// TODO if we end up passing in sorted query, rename x to sorted and
// modify for loop accordingly
template<int BlockSize=8, class VectorT1, class VectorT2, class IdxVectorT,
	class dist_t>
dist_t dist_sq_order(const VectorT1& x, const VectorT2& y,
	const IdxVectorT& order, dist_t thresh)
{
	auto n = x.size();
	int32_t numBlocks = order.size();
	assert(numBlocks == n / BlockSize);

	dist_t dist = 0;
	for (int32_t i = 0; i < numBlocks; i++) {
		auto idx = order(i);
		dist += (x.segment<BlockSize>(idx) - y.segment<BlockSize>(idx)).squaredNorm();
		if (dist > thresh) {
			return dist;
		}
	}
	int32_t finalBlockEnd = n - (n % BlockSize);
	for (int32_t i = finalBlockEnd; i < n; i++) {
		auto diff = x(i) - y(i);
		dist += diff * diff;
	}
}

template<int BlockSize=8, class VectorT1, class VectorT2, class IdxVectorT,
	class dist_t>
dist_t dist_sq_order_presorted(const VectorT1& x_sorted, const VectorT2& y,
	const IdxVectorT& order, dist_t thresh)
{
	auto n = x_sorted.size();
	int32_t numBlocks = order.size();
	assert(numBlocks == n / BlockSize);

	dist_t dist = 0;
	for (int32_t i = 0; i < numBlocks; i++) {
		auto idx = order(i);
		dist += (x_sorted.segment<BlockSize>(i * BlockSize) - y.segment<BlockSize>(idx)).squaredNorm();
		if (dist > thresh) {
			return dist;
		}
	}
	int32_t finalBlockEnd = n - (n % BlockSize);
	for (int32_t i = finalBlockEnd; i < n; i++) {
		auto diff = x_sorted(i) - y(i);
		dist += diff * diff;
	}
}

// ================================ query preprocessing

template<int BlockSize=8, class VectorT, class IdxT>
void create_ordered_query(const VectorT& query, const VectorT& means,
	typename VectorT::Scalar * query_out, IdxT* order_out)
{
	typedef typename VectorT::Scalar data_t;
	typedef std::pair<IdxT, data_t> pair;

	// XXX query must be padded to be a multiple of BlockSize
	IdxT len = query.size();
	IdxT num_blocks = len / BlockSize;

	// compute squared distances to mean for each block, reusing query storage
	//
	// TODO sum these dists and return it so we can get LB based on centroid
	//
	idx_t idx = 0;
	for (idx_t i = 0; i < len; i++) {
		query_out(i) = (query.segment<BlockSize>(idx) - means.segment<BlockSize>(idx)).squaredNorm();
		idx += BlockSize;
	}

	// pair idx with value // TODO pass in storage for this
	// assert((BlockSize * sizeof(pair)) <= len * sizeof(IdxT)); // will it fit?
	auto pairs = std::unique_ptr<pair>(new pair[num_blocks]);
	for (idx_t block_num = 0; block_num < num_blocks; block_num++) {
		pairs[block_num] = pair(block_num, query_out(block_num)); // sum stored in query
	}
	// auto pairs = mapi([](length_t i, data_t x) {
	// 	return pair(i, x);
	// }, data, len);

	// sort pairs by summed distance
	std::sort(pairs.get(), pairs.get() + len,
	[](const pair& lhs, const pair& rhs) {
		return lhs.second > rhs.second;
	});

	// unpack pairs into sorted query and order; order is just an order
	// for the blocks, so sorted query has groups of BlockSize consecutive
	// idxs reordered
	idx = 0;
	for (IdxT i = 0; i < num_blocks; i++) {
		order_out[i] = pairs[i].first;
		for (IdxT j = 0; j < BlockSize; j++) {
			query_out[idx] = pairs[i].second + j;
			idx++;
		}
	}
}

// ================================ early-abandoning search

// ------------------------ radius

template<class RowMatrixT, class VectorT, class DistT>
vector<Neighbor> radius_sequential(const RowMatrixT& X,
	const VectorT& query, idx_t num_rows, DistT radius_sq)
{
	vector<Neighbor> ret;
	for (idx_t i = 0; i < num_rows; i++) {
		auto dist = dist_sq(X.row(i).eval(), query, radius_sq);
		if (dist <= radius_sq) {
			ret.emplace_back(dist, i);
		}
	}
	return ret;
}

template<class RowMatrixT, class VectorT, class IdxVectorT, class DistT>
vector<Neighbor> radius_order(const RowMatrixT& X,
	const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
	DistT radius_sq)
{
	vector<Neighbor> ret;
	for (idx_t i = 0; i < num_rows; i++) {
		auto dist = dist_sq_order_presorted(query_sorted, X.row(i).eval(),
			order, radius_sq);
		if (dist <= radius_sq) {
			ret.emplace_back(dist, i);
		}
	}
	return ret;
}

template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
	class DistT>
vector<Neighbor> radius(const RowMatrixT& X, const VectorT1& query,
	const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
	idx_t num_rows, idx_t num_rows_thresh, DistT radius_sq)
{
	if (num_rows >= num_rows_thresh) {
		create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
		return radius_order(X, query_tmp, order_tmp, num_rows, radius_sq);
	} else {
		return radius_sequential(X, query, num_rows, radius_sq);
	}
}

// ------------------------ 1nn

template<class RowMatrixT, class VectorT, class DistT>
Neighbor onenn_sequential(const RowMatrixT& X,
	const VectorT& query, idx_t num_rows, DistT d_bsf=kMaxDist)
{
	Neighbor ret{.dist = d_bsf, .idx = kInvalidIdx };
	for (idx_t i = 0; i < num_rows; i++) {
		auto dist = dist_sq(X.row(i).eval(), query, d_bsf);
		if (dist < d_bsf) {
			ret = {.dist = dist, .idx = i};
		}
	}
	return ret;
}

template<class RowMatrixT, class VectorT, class IdxVectorT, class DistT>
Neighbor onenn_order(const RowMatrixT& X,
	const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
	DistT d_bsf=kMaxDist)
{
	Neighbor ret{.dist = d_bsf, .idx = kInvalidIdx};
	for (idx_t i = 0; i < num_rows; i++) {
		auto dist = dist_sq_order_presorted(query_sorted, X.row(i).eval(),
			order, d_bsf);
		if (dist < ret.dist) {
			ret = {.dist = dist, .idx = i};
		}
	}
	return ret;
}

template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
	class DistT>
Neighbor onenn(const RowMatrixT& X, const VectorT1& query,
	const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
	idx_t num_rows, idx_t num_rows_thresh, DistT d_bsf=kMaxDist)
{
	if (num_rows >= num_rows_thresh) {
		create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
		return onenn_order(X, query_tmp, order_tmp, num_rows, d_bsf);
	} else {
		return onenn_sequential(X, query, num_rows, d_bsf);
	}
}

// ------------------------ knn

template<class RowMatrixT, class VectorT, class DistT>
vector<Neighbor> knn_sequential(const RowMatrixT& X,
	const VectorT& query, idx_t num_rows, int k, DistT d_bsf=kMaxDist)
{
	vector<Neighbor> ret(k, {.dist = d_bsf, .idx = kInvalidIdx});
	for (idx_t i = 0; i < num_rows; i++) {
		auto dist = dist_sq(X.row(i).eval(), query, d_bsf);
		insert_neighbor(ret, dist, i); // figures out whether dist is lower
	}
	return ret;
}

template<class RowMatrixT, class VectorT, class IdxVectorT, class DistT>
vector<Neighbor> knn_order(const RowMatrixT& X,
	const VectorT& query_sorted, const IdxVectorT& order, idx_t num_rows,
	int k, DistT d_bsf=kMaxDist)
{
	vector<Neighbor> ret(k, {.dist = d_bsf, .idx = kInvalidIdx});
	for (idx_t i = 0; i < num_rows; i++) {
		auto dist = dist_sq_order_presorted(query_sorted, X.row(i).eval(),
			order, d_bsf);
		insert_neighbor(ret, dist, i); // figures out whether dist is lower
	}
	return ret;
}

template<class RowMatrixT, class VectorT1, class VectorT2, class IdxVectorT,
	class DistT>
vector<Neighbor> knn(const RowMatrixT& X, const VectorT1& query,
	const VectorT2& means, VectorT1& query_tmp, IdxVectorT& order_tmp,
	idx_t num_rows, idx_t num_rows_thresh, int k, DistT d_bsf=kMaxDist)
{
	if (num_rows >= num_rows_thresh) {
		create_ordered_query(query, means, query_tmp.data(), order_tmp.data());
		return knn_order(X, query_tmp, order_tmp, num_rows, d_bsf);
	} else {
		return knn_sequential(X, query, num_rows, d_bsf);
	}
}

} // namespace abandon


template<class RowMatrixT>
class L2IndexAbandon: public IndexBase<L2IndexAbandon<RowMatrixT>, RowMatrixT> {
public:

	typedef Matrix<typename RowMatrixT::Scalar, 1, Dynamic, RowMajor> RowVectT;
	typedef Matrix<int32_t, 1, Dynamic, RowMajor> RowVectIdxsT;

	explicit L2IndexAbandon(const RowMatrixT& data):
		_data(data)
	{
		_colMeans = data.colwise().mean();
		_orderIdxs = RowVectIdxsT(data.cols());
	}



	template<class VectorT>
	Neighbor onenn(const VectorT& query) {
//		typename RowMatrixT::Index idx;
//		double dist = (_data.rowwise() - query.transpose()).rowwise().squaredNorm().minCoeff(&idx);
//		return {.dist = dist, .idx = static_cast<int32_t>(idx)}; // TODO fix length_t
	}

	template<class VectorT>
	vector<Neighbor> knn(const VectorT& query, size_t k) {
//		VectorT dists = squaredDistsToVector(_data, query);
//		return knn_from_dists(dists.data(), dists.size(), k);
	}

	// ------------------------------------------------ batch of queries

	vector<vector<Neighbor> > radius_batch(const RowMatrixT& queries,
										   double radius_sq)
	{
		vector<vector<Neighbor> > ret;
		for (idx_t j = 0; j < queries.rows(); j++) {
			ret.emplace_back(radius(queries.row(j).eval(), radius_sq));
		}
		return ret;
	}

	vector<vector<Neighbor> > knn_batch(const RowMatrixT& queries, size_t k) {
		vector<vector<Neighbor> > ret;
		for (idx_t j = 0; j < queries.rows(); j++) {
			ret.emplace_back(knn(queries.row(j).eval(), k));
		}
		return ret;
	}

	vector<Neighbor> onenn_batch(const RowMatrixT& queries) {
		vector<Neighbor> ret;
		for (idx_t j = 0; j < queries.rows(); j++) {
			ret.emplace_back(onenn(queries.row(j).eval()));
		}
		return ret;
	}

private:
	RowMatrixT _data;
	RowVectT _colMeans;
	RowVectIdxsT _orderIdxs;
};


	// X TODO similar class called L2Index_abandon that does UCR ED
	// (without znorming)
	//	-prolly also give it batch funcs, but these just wrap the
	// 	one-at-a-time func
	//	-also give it ability to store approx top eigenvects for early
	// 	abandoning

	// TODO all of the above need insert and delete methods
	//	-alloc a bigger mat than we need and store how many rows are used
	//	-wrap this up in its own class that's subclass of RowMatrixT

	// X TODO abstract superclass for all the KNN-ish classes

	// TODO wrapper for eigen mat prolly needs to also pad ends of rows so
	// that each row is 128-bit aligned for our SIMD instrs
	// 	-will prolly need to wrap rows in a Map to explicitly tell it the
	// 	alignment though, cuz won't know it's safe at compile-time
	//	-just punt this for now cuz it's really ugly

	// TODO brute force (not early abandoning) stuff needs num_rows to support
	// inserts and deletes

	// TODO tests for all of the above

	// TODO E2LSH impl
	// TODO Selective Hashing Impl on top of it

