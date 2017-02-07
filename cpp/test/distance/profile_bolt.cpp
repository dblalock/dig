
#include "test_bolt.hpp"

#include "catch.hpp"
#include "bolt.hpp"

#include "eigen_utils.hpp"
#include "timing_utils.hpp"
#include "testing_utils.hpp"
#include "memory.hpp"

#include "debug_utils.hpp"


TEST_CASE("bolt encoding speed", "[bolt][mcq][profile]") {
    static constexpr int nrows = 10000;
    // static constexpr int M = 32;
    static constexpr int M = 16;
    static constexpr int ncodebooks = 2 * M;
    static constexpr int ncentroids = 16;
    static constexpr int ncentroids_total = ncentroids * ncodebooks;
    static constexpr int subvect_len = 4;
    static constexpr int ncols = ncodebooks * subvect_len;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    RowMatrix<uint8_t> encoding_out(nrows, M);
    double t = 0;
    {
        EasyTimer _(t);
        bolt_encode<M>(X.data(), nrows, ncols, centroids.data(),
                       encoding_out.data());
    }
    prevent_optimizing_away_dists(encoding_out.data(), nrows);
    print_dist_stats("bolt encode", encoding_out.data(), nrows, t);
}


TEST_CASE("bolt lut encoding speed", "[bolt][mcq][profile]") {
    static constexpr int nrows = 10000;
    // static constexpr int M = 32;
    static constexpr int M = 16;
    static constexpr int ncodebooks = 2 * M;
    static constexpr int ncentroids = 16;
    static constexpr int ncentroids_total = ncentroids * ncodebooks;
    static constexpr int subvect_len = 4;
    static constexpr int ncols = ncodebooks * subvect_len;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> Q(nrows, ncols);
    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);
    double t;
    {
        EasyTimer _(t);
        for (int i = 0; i < nrows; i++) {
            bolt_lut_l2<M>(Q.row(i).data(), ncols, centroids.data(), lut_out.data());
        }
    }
    print_dist_stats("bolt encode lut", nrows, t);
}


TEST_CASE("bolt scan speed", "[bolt][mcq][profile]") {
    // static constexpr int nblocks = 37;
    // static constexpr int nblocks = 500 * 1000;
    static constexpr int nblocks = 100 * 1000;
    // static constexpr int nblocks = 10 * 1000;
    // static constexpr int nblocks = 1 * 1000;
    static constexpr int nrows = 32 * nblocks;
    // static constexpr int M = 8; // number of bytes per compressed vect
    static constexpr int M = 16; // number of bytes per compressed vect
    // static constexpr int M = 32; // number of bytes per compressed vect
    static constexpr int ncodebooks = 2 * M;
    static constexpr int ncentroids = 16;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create random luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    luts.setRandom();
    luts = luts.array() / (2 * M); // make max lut value small

    double t = 0;

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);

    {
        EasyTimer _(t);
        bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks);
    }
    print_dist_stats("bolt scan uint8", dists_u8.data(), nrows, t);
    {
        EasyTimer _(t);
        bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks);
    }
    print_dist_stats("bolt scan uint16", dists_u16.data(), nrows, t);
    {
        EasyTimer _(t);
        bolt_scan<M, true>(codes.data(), luts.data(), dists_u16_safe.data(), nblocks);
    }
    print_dist_stats("bolt scan uint16_safe", dists_u16_safe.data(), nrows, t);

    if (nrows < 100 * 1000) {
        check_bolt_scan(dists_u8.data(), dists_u16.data(), dists_u16_safe.data(),
            luts, codes, M, nblocks);
    }

    // auto dists_u8 = aligned_random_ints<uint8_t>(nrows);
    // RowVector<uint8_t> _dists_u8(nrows);
    // RowVector<uint16_t> _dists_u16(nrows);
    // RowVector<uint16_t> _dists_u16_safe(nrows);
    // auto dists_u8 = _dists_u8.data();
    // auto dists_u16 = _dists_u16.data();
    // auto dists_u16_safe = _dists_u16_safe.data();

    // PRINTLN_VAR(luts.cast<int>());

    // SECTION("uint8") {
    //     {
    //         EasyTimer _(t);
    //         bolt_scan<M>(codes.data(), luts.data(), dists_u8, nblocks);
    //     }
    //     print_dist_stats(dists_u8, "bolt scan uint8", nrows, t);
    //     // aligned_free(dists_u8);
    // }
    // // auto dists_u16 = aligned_random_ints<uint16_t>(nrows);
    // SECTION("uint16") {
    //     {
    //         EasyTimer _(t);
    //         bolt_scan<M>(codes.data(), luts.data(), dists_u16, nblocks);
    //     }
    //     print_dist_stats(dists_u16, "bolt scan uint16", nrows, t);
    //     // aligned_free(dists_u16);
    // }
    // // auto dists_u16_safe = aligned_random_ints<uint16_t>(nrows);
    // SECTION("uint16 safe") {
    //     {
    //         EasyTimer _(t);
    //         bolt_scan<M, true>(codes.data(), luts.data(), dists_u16_safe, nblocks);
    //     }
    //     print_dist_stats(dists_u16_safe, "bolt scan uint16 safe", nrows, t);
    //     // aligned_free(dists_u16_safe);
    // }

    // bool fail = false;
    // for (int i = 0; i < nrows; i++) {
    //     fail = fail || dists_u8[i] > dists_u16[i];
    //     // fail = fail || dists_u16[i] > dists_u16_safe[i];
    // }
    // std::cout << "fail? " << fail << "\n";

    // if (nrows < 10 * 10000) {
    //     check_bolt_scan(dists_u8, dists_u16, dists_u16_safe, luts, codes,
    //         M, nblocks);
    // }

    // aligned_free(dists_u8);
    // aligned_free(dists_u16);
    // aligned_free(dists_u16_safe);
}
