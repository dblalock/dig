

#include "test_bolt.hpp"
//#include "catch.hpp"

#include "array_utils.hpp"
#include "testing_utils.hpp"
#include "debug_utils.hpp"
#include "memory.hpp"


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

        for (int m = 0; m < 2 * M; m++) {
            int byte = encoding_out(m / 2);
            int idx = m % 2 ? byte >> 4 : byte & 0x0F;
            REQUIRE(idx == 2 * m);
        }
//        std::cout << "\n";
    }

    SECTION("encode rows of matrix") {
        static constexpr int nrows = 10;
        auto X = create_X_matrix(nrows);
        RowMatrix<uint8_t> encoding_out(nrows, M);
        bolt_encode<M>(X.data(), nrows, total_len, centroids.data(),
                       encoding_out.data());

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
    static constexpr int nblocks = 937; // arbitrary weird number
    static constexpr int nrows = 32 * nblocks;

    // create random codes from [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create centroids
    ColMatrix<float> centroids = create_bolt_centroids(1);

    // create query and look-up tables
    RowVector<float> q = create_bolt_query();
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    bolt_lut_l2<M>(q.data(), total_len, centroids.data(), luts.data());

//    PRINTLN_VAR(codes.cast<int>());
//    PRINTLN_VAR(codes.topRows(2).cast<int>());
//    PRINTLN_VAR(centroids);
//    PRINTLN_VAR(luts.cast<int>());
//    PRINTLN_VAR(q);

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);
    bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks);
    bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks);
    bolt_scan<M, true>(codes.data(), luts.data(), dists_u16_safe.data(), nblocks);

//    PRINTLN_VAR(dists_u8.cast<int>());
//    PRINTLN_VAR(dists_u16.cast<int>());
//    PRINTLN_VAR(dists_u16_safe.cast<int>());

//    printf("dists true:\n");
    check_bolt_scan(dists_u8.data(), dists_u16.data(), dists_u16_safe.data(),
                    luts, codes, M, nblocks);

//    for (int b = 0; b < nblocks; b++) {
//        auto dist_ptr_u8 = dists_u8.data() + b * 32;
//        auto dist_ptr_u16 = dists_u16.data() + b * 32;
//        auto dist_ptr_u16_safe = dists_u16_safe.data() + b * 32;
//        auto codes_ptr = codes.data() + b * M * 32;
//        for (int i = 0; i < 32; i++) {
//            int dist_u8 = dist_ptr_u8[i];
//            int dist_u16 = dist_ptr_u16[i];
//            int dist_u16_safe = dist_ptr_u16_safe[i];
//
//            // compute dist the scan should have returned based on the LUT
//            int dist_true_u8 = 0;
//            int dist_true_u16 = 0;
//            int dist_true_u16_safe = 0;
//            for (int m = 0; m < M; m++) {
////                uint8_t byte = codes(i, m);
//                uint8_t code = codes_ptr[i + 32 * m];
//                uint8_t low_bits = code & 0x0F;
//                uint8_t high_bits = (code >> 4) & 0x0F;
//
//                auto d0 = luts(low_bits, 2 * m);
//                auto d1 = luts(high_bits, 2 * m + 1);
//
//                // uint8 distances
//                dist_true_u8 += d0 + d1;
//
//                // uint16 distances
//                auto pair_dist = d0 + d1;
//                dist_true_u16 += pair_dist > 255 ? 255 : pair_dist;
//
//                // uint16 safe distance
//                dist_true_u16_safe += d0 + d1;
//            }
//            dist_true_u8 = dist_true_u8 > 255 ? 255 : dist_true_u8;
//            CAPTURE(b);
//            CAPTURE(i);
//            REQUIRE(dist_true_u8 == dist_u8);
//            REQUIRE(dist_true_u16 == dist_u16);
//            REQUIRE(dist_true_u16_safe == dist_u16_safe);
//        }
//    }
//    printf("\n");
}
