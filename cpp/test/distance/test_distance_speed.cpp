
#include <iostream>

#include "euclidean.hpp"
#include "catch.hpp"


TEST_CASE("tmp", "debug") {
    using Scalar = float;
    int N = 100;
    int D = 40;
    RowMatrix<Scalar> X(N, D);
    X.setRandom();
    RowVector<Scalar> q(D);
    q.setRandom();

    auto d = dist::abandon::dist_sq(X.row(0), q);
	std::cout << "found distance: " << d << "\n";
}
