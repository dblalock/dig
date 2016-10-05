//
//  cluster.hpp
//  Dig
//
//  Created by DB on 2016-10-2
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __CLUSTER_HPP
#define __CLUSTER_HPP

#include <sys/types.h>
#include <vector>

#include "Dense"
#include "array_utils.hpp"
#include "eigen_utils.hpp" // for mat_traits

namespace cluster {

template<class assignment_t=int32_t, class MatrixT=char>
auto kmeans(const MatrixT& X, assignment_t k)
    -> std::pair<typename mat_traits<MatrixT>::RowMatrixT,
        std::vector<assignment_t> >
{
    using Scalar = typename MatrixT::Scalar;
    using RowMatrixT = typename mat_traits<MatrixT>::RowMatrixT;
	using Index = typename MatrixT::Index;
    assert(k <= X.rows());

    vector<assignment_t> assignments(X.rows());
    RowMatrixT centroids(k, X.cols());

    if (k == 1) { // one centroid -> global mean, all assigned to 0
        centroids = X.colwise().mean();
        return std::make_pair(centroids, assignments);
    }
    // initialize centroids to random rows in X
    auto initial_idxs =  ar::rand_idxs(X.rows(), k);
    for (assignment_t i = 0; i < k; i++) {
        centroids.row(i) = X.row(initial_idxs[i]);
    }

    // precompute row norms + allocate centroid stats
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> row_norms =
        X.rowwise().squaredNorm();
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> centroid_norms =
        centroids.rowwise().squaredNorm().transpose();
    Eigen::Matrix<Index, Eigen::Dynamic, 1> counts(k);

    // lloyd's algorithm to optimize the centroids + assignments
    RowMatrixT dists(X.rows(), k);
    Scalar prev_dist = -1;
    while (true) {
        // compute dists via ||x-y||^2 = ||x||^2 + ||y||^2 - 2x.y
        dists = static_cast<Scalar>(-2.) * (X * centroids.transpose());
        dists.colwise() += row_norms;
        dists.rowwise() += centroid_norms;

        // update assignments and centroids
        Scalar current_dist = 0; // if this is unchanged, we've converged
        centroids.setZero();
        counts.setZero();
        Index idx = -1;
        for (int i = 0; i < dists.rows(); i++) {
            current_dist += dists.row(i).minCoeff(&idx);
            counts(idx) += 1;
            centroids.row(idx) += X.row(i);
            assignments[i] = static_cast<assignment_t>(idx);
        }
        for (assignment_t i = 0; i < k; i++) { // divide sums by counts to get means
            centroids.row(i) /= counts(i);
        }
        centroid_norms = centroids.rowwise().squaredNorm().transpose();

        // break if assignments haven't changed (which yields same dist)
        if (current_dist == prev_dist) { break; }
        prev_dist = current_dist;
        PRINT_VAR(prev_dist);
    }

    return std::make_pair(centroids, assignments);
}

} // namespace cluster
#endif // __CLUSTER_HPP
