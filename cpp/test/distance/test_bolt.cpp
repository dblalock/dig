

#include "catch.hpp"
#include "bolt.hpp"

#include "Dense"

#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "testing_utils.hpp"
#include "debug_utils.hpp"
// #include "bit_ops.hpp"
#include "memory.hpp"


static constexpr int M = 2;
static constexpr int subvect_len = 3;
static constexpr int ncodebooks = 2 * M;
static constexpr int total_len = ncodebooks * subvect_len;
static constexpr int total_sz = 16 * total_len;
static constexpr int ncentroids = 16; // always 16 for 4 bits
static constexpr int ncentroids_total = ncentroids * ncodebooks;
static constexpr int codebook_sz = ncentroids * subvect_len;

template<class T>
RowMatrix<T> create_rowmajor_centroids(int start_coeff=10) {
    RowMatrix<T> C(ncentroids_total, subvect_len);
    for (int i = 0; i < ncentroids_total; i++) {
        for (int j = 0; j < subvect_len; j++) {
            C(i, j) = start_coeff * i + j;
        }
    }
    return C;
}


TEST_CASE("bolt_smoketest", "[mcq][bolt]") {
    // TODO instantiate bolt encoder object here
    // printf("done");
}


TEST_CASE("bolt_encode_centroids", "[mcq][bolt]") {
    auto C = create_rowmajor_centroids<int>();
    ColMatrix<int> C_out(ncentroids, total_len);
    bolt_encode_centroids<M>(C.data(), total_len, C_out.data());
//    std::cout << C_out << "\n"; // yes, looks exactly right

    for (int m = 0; m < ncodebooks; m++) {
        auto cin_start_ptr = C.data() + m * codebook_sz;
        auto cout_start_ptr = C_out.data() + m * codebook_sz;
        for (int i = 0; i < ncentroids; i++) { // for each centroid
            for (int j = 0; j < subvect_len; j++) { // for each dim
                CAPTURE(m);
                CAPTURE(i);
                CAPTURE(j);
                auto cin_ptr = cin_start_ptr + (subvect_len * i) + j;
                auto cout_ptr = cout_start_ptr + (ncentroids * j) + i;
                REQUIRE(*cin_ptr == *cout_ptr);
            }
        }
    }
}

TEST_CASE("bolt_lut_l2", "[mcq][bolt]") {
    // create centroids with predictable patterns; note that we have to
    // be careful not to saturate the range of the uint8_t distances, which
    // means all of these entries have to within 15 of the corresponding
    // element of the query
    auto centroids_rowmajor = create_rowmajor_centroids<float>(1);
    
    ColMatrix<float> centroids(ncentroids, total_len);
    bolt_encode_centroids<M>(centroids_rowmajor.data(), total_len, centroids.data());
    
    // for 4 codebooks, subvect_len = 3, q =
    // [8, 9, 10, 24, 25, 26, 40, 41, 42, 56, 57, 58]
    RowVector<float> q(total_len);
    for (int m = 0; m < ncodebooks; m++) {
        for (int j = 0; j < subvect_len; j++) {
            auto idx = m * subvect_len + j;
            q(idx) = ncentroids * m + j + (ncentroids / 2);
        }
    }
    
    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    //    ColMatrix<float> lut_out(ncentroids, ncodebooks);
    //    ColMatrix<int32_t> lut_out(ncentroids, ncodebooks);
    //    ColMatrix<uint16_t> lut_out(ncentroids, ncodebooks);
    //    lut_out.fill(42); // there should be none of these when we print it
    bolt_lut_l2<M>(q.data(), total_len, centroids.data(), lut_out.data());
    
    std::cout << centroids_rowmajor << "\n\n";
    std::cout << centroids << "\n\n";
    std::cout << q << "\n";
    std::cout << lut_out.cast<int>() << "\n";
    
    for (int m = 0; m < ncodebooks; m++) {
        for (int i = 0; i < ncentroids; i++) {
            float dist_sq = 0;
            for (int j = 0; j < subvect_len; j++) {
                auto col = m * subvect_len + j;
                auto diff = centroids_rowmajor(i + m * ncentroids, j) - q(m * subvect_len + j);
                dist_sq += diff * diff;
            }
            CAPTURE(m);
            CAPTURE(i);
            REQUIRE(dist_sq == lut_out(i, m));
        }
    }
}






