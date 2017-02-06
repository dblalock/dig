

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

ColMatrix<float> create_bolt_centroids(int start_coeff=1) {
    auto centroids_rowmajor = create_rowmajor_centroids<float>(start_coeff);
    ColMatrix<float> centroids(ncentroids, total_len);
    bolt_encode_centroids<M>(centroids_rowmajor.data(), total_len, centroids.data());
    return centroids;
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
    
//    std::cout << centroids_rowmajor << "\n\n";
//    std::cout << centroids << "\n\n";
//    std::cout << q << "\n";
//    std::cout << lut_out.cast<int>() << "\n";
    
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

TEST_CASE("bolt_encode", "[mcq][bolt]") {
    auto centroids = create_bolt_centroids(1);
    
    SECTION("encode one vector") {
        // for 4 codebooks, subvect_len = 3, q =
        // [0, 1, 2, 18, 19, 20, 36, 37, 38, 54, 55, 56]
        RowVector<float> q(total_len);
        for (int m = 0; m < ncodebooks; m++) {
            for (int j = 0; j < subvect_len; j++) {
                auto idx = m * subvect_len + j;
                // add on a 2m at the end so which centroid it is changes by
                // 2 for each codebook
                q(idx) = ncentroids * m + j + (2 * m);
            }
        }
        
        RowVector<uint8_t> encoding_out(M);
        bolt_encode<M>(q.data(), 1, total_len, centroids.data(), encoding_out.data());
        
//        std::cout << "q: " << q << "\n";
//        std::cout << "centroids:\n" << centroids << "\n";
//        std::cout << "raw encoding bytes: " << encoding_out.cast<int>() << "\n";
//        std::cout << "encoding:\n";
        
        for(int m = 0; m < 2 * M; m++) {
            int byte = encoding_out(m / 2);
            int idx = m % 2 ? byte >> 4 : byte &0x0F;
            REQUIRE(idx == 2 * m);
        }
//        std::cout << "\n";
    }
    
    SECTION("encode rows of matrix") {
        static constexpr int nrows = 10;
        RowMatrix<float> X(nrows, total_len);
        for (int i = 0; i < nrows; i++) {
            for (int m = 0; m < ncodebooks; m++) {
                for (int j = 0; j < subvect_len; j++) {
                    auto idx = m * subvect_len + j;
                    // add on m at the end so which centroid it is changes by
                    // 1 for each codebook; also add on i so that each row
                    // will pick centroids 1 higher the previous ones
                    X(i, idx) = ncentroids * m + j + m + (i % 5);
                }
            }
        }

        RowMatrix<uint8_t> encoding_out(nrows, M);
        bolt_encode<M>(X.data(), nrows, total_len, centroids.data(), encoding_out.data());
        
        for (int i = 0; i < nrows; i++) {
            for(int m = 0; m < 2 * M; m++) {
                int byte = encoding_out(i, m / 2);
                int idx = m % 2 ? byte >> 4 : byte &0x0F;
                REQUIRE(idx == m + (i % 5));
            }
        }
    }
}

TEST_CASE("bolt_scan", "[mcq][bolt]") {
    
    
}


