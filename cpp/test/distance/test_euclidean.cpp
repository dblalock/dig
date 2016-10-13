
#include "catch.hpp"
#include "euclidean.hpp"

#include "Dense"

#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "testing_utils.hpp"

using RowMatrixXd = RowMatrix<double>;
using RowVectorXd = RowVector<double>;

template<class MatrixT, class VectorT>
void _test_squared_dists_to_vector(const MatrixT& X, const VectorT& q) {
    VectorT dists = dist::squared_dists_to_vector(X, q);
    for (int32_t i = 0; i < X.rows(); i++) {
        double simple_dist = ar::dist_sq(X.row(i).data(), q.data(), q.size());
        REQUIRE(approxEq(dists(i), simple_dist));
    }
}

template<class MatrixT>
void _test_squared_dists_to_vectors(const MatrixT& X, const MatrixT& V) {
    MatrixT dists = dist::squared_dists_to_vectors(X, V);
    for (int32_t i = 0; i < X.rows(); i++) {
        for (int32_t j = 0; j < V.rows(); j++) {
            double simple_dist = ar::dist_sq(X.row(i).data(), V.row(j).data(),
                                         V.row(j).size());
            REQUIRE(approxEq(dists(i, j), simple_dist));
        }
    }
}

// TODO put this in a different test file
TEST_CASE("squared_dists_to_vector(s)", "distance") {
    for (int i = 0; i < 5; i++) {
        for (int n = 1; n <= 125; n *= 5) {
            for (int d = 1; d <= 100; d *= 10) {
                RowMatrixXd X(n, d);
                X.setRandom();

                RowVectorXd q(d);
                q.setRandom();
                _test_squared_dists_to_vector(X, q);

                int num_queries = std::max(n / 10, 1);
                RowMatrixXd V(num_queries, d);
                V.setRandom();
                _test_squared_dists_to_vectors(X, V);
            }
        }
    }
}
