

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
RowMatrix<T> create_rowmajor_centroids(T centroid_step=10,
    T codebook_step=16)
{
    RowMatrix<T> C(ncentroids_total, subvect_len);
    for (int i = 0; i < ncentroids_total; i++) {
        int centroid_start_val = centroid_step * (i % ncentroids) +
            (i / ncentroids) * codebook_step;
        for (int j = 0; j < subvect_len; j++) {
            C(i, j) = centroid_start_val + j;
        }
    }
    return C;
}

ColMatrix<float> create_bolt_centroids(float centroid_step=1,
    float codebook_step=16)
{
    auto centroids_rowmajor = create_rowmajor_centroids<float>(
        centroid_step, codebook_step);
    ColMatrix<float> centroids(ncentroids, total_len);
    bolt_encode_centroids<M>(centroids_rowmajor.data(), total_len, centroids.data());
    return centroids;
}

RowVector<float> create_bolt_query() {
    // for 4 codebooks, subvect_len = 3, q =
    // [0, 1, 2, 18, 19, 20, 36, 37, 38, 54, 55, 56]
    RowVector<float> q(total_len);
    for (int m = 0; m < ncodebooks; m++) {
        for (int j = 0; j < subvect_len; j++) {
            auto idx = m * subvect_len + j;
            q(idx) = ncentroids * m + j + (ncentroids / 2);
        }
    }
    return q;
}

RowMatrix<float> create_X_matrix(int64_t nrows) {
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
    return X;
}

RowMatrix<uint8_t> create_bolt_codes(int64_t nrows, ColMatrix<float> centroids)
{
    auto X = create_X_matrix(nrows);
    RowMatrix<uint8_t> X_enc(nrows, M);
    bolt_encode<M>(X.data(), nrows, total_len, centroids.data(), X_enc.data());
    return X_enc;
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
    
    RowVector<float> q = create_bolt_query();
    
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
            int idx = m % 2 ? byte >> 4 : byte & 0x0F;
            REQUIRE(idx == 2 * m);
        }
//        std::cout << "\n";
    }
    
    SECTION("encode rows of matrix") {
        static constexpr int nrows = 10;
        auto encoding_out = create_bolt_codes(nrows, centroids);
        
        for (int i = 0; i < nrows; i++) {
            for(int m = 0; m < 2 * M; m++) {
                // indices are packed into upper and lower 4 bits
                int byte = encoding_out(i, m / 2);
                int idx = m % 2 ? byte >> 4 : byte & 0x0F;
                REQUIRE(idx == m + (i % 5)); // i % 5 from how we designed mat
            }
        }
    }
}

TEST_CASE("bolt_scan", "[mcq][bolt]") {
    static constexpr int nblocks = 1;
    static constexpr int nrows = 32 * nblocks;
    
    // create random codes from in [0, 15]
    RowMatrix<uint8_t> X(nrows, total_len);
    X.setRandom();
    RowMatrix<uint8_t> codes = X / 16;
    
    // create centroids
    ColMatrix<float> centroids = create_bolt_centroids(1);

    // create query and look-up tables
    RowVector<float> q = create_bolt_query();
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    bolt_lut_l2<M>(q.data(), total_len, centroids.data(), luts.data());
    
//    PRINTLN_VAR(codes.cast<int>());
//    PRINTLN_VAR(centroids);
//    PRINTLN_VAR(luts.cast<int>());
//    PRINTLN_VAR(q);
    
    // do the scan to compute the distances
    auto dists_out = aligned_alloc<uint8_t>(nrows);
    bolt_scan<M>(codes.data(), luts.data(), dists_out, nblocks);
    
    for (int b = 0; b < nblocks; b++) {
        auto dist_ptr = dists_out + b * 32;
        for (int i = 0; i < 32; i++) {
            int dist = dist_ptr[i];
            
            // compute dist this should have returned based on the LUT
            int dist_true = 0;
            for (int m = 0; m < M; m++) {
                uint8_t byte = codes(i, m);
                uint8_t low_bits = byte & 0x0F;
                uint8_t high_bits = (byte >> 4) & 0x0F;
                
                dist_true += luts(low_bits, 2 * m);
                dist_true += luts(high_bits, 2 * m + 1);
            }
            dist_true = dist_true > 255 ? 255 : dist_true;
            CAPTURE(b);
            CAPTURE(i);
            REQUIRE(dist_true == dist);
        }
    }
    
    aligned_free(dists_out);
}
