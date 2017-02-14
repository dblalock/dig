
// #include "test_bolt.hpp"

#include "catch.hpp"
#include "bolt.hpp"

#include "eigen_utils.hpp"
#include "timing_utils.hpp"
#include "testing_utils.hpp"
#include "memory.hpp"

#include "debug_utils.hpp"

static constexpr int kNreps = 10;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 5;

static constexpr int M = 32; // number of bytes per compressed vect
static constexpr int64_t nrows_enc = 10*1000; // number of rows to encode
static constexpr int64_t nrows_lut = 10*1000; // number of luts to create
static constexpr int64_t nblocks_scan = 1000*1000 / 32;
static constexpr int64_t nblocks_query = 100*1000 / 32;
static constexpr int subvect_len = 4; // M * subvect_len = number of features
static constexpr int nqueries = 100;

static constexpr int bits_per_codebook = 4;

static constexpr int ncodebooks = M * (8 / bits_per_codebook);
static constexpr int ncentroids = (1 << bits_per_codebook);
static constexpr int ncentroids_total = ncentroids * ncodebooks;
static constexpr int ncols = ncodebooks * subvect_len;
static constexpr int lut_data_sz = ncentroids * ncodebooks;


TEST_CASE("print bolt params", "[bolt][mcq][profile]") {
    printf("------------------------ bolt\n");
    printf("---- bolt profiling parameters\n");
    printf("bolt M: %d\n", M);
    printf("bolt nrows_enc: %g\n", (double)nrows_enc);
    printf("bolt nrows_lut: %g\n", (double)nrows_lut);
    // printf("nblocks_scan: %g\n", (double)nblocks_scan);
    printf("bolt nrows_scan: %g\n", (double)nblocks_scan * 32);
    // printf("nblocks_query: %g\n", (double)nblocks_query);
    printf("bolt nrows_query: %g\n", (double)nblocks_query * 32);
    printf("bolt subvect_len: %d\n", subvect_len);
    printf("bolt nqueries: %d\n", nqueries);
    printf("---- bolt timings\n");
}

TEST_CASE("bolt encoding speed", "[bolt][mcq][profile]") {
    static constexpr int nrows = nrows_enc;


    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    RowMatrix<uint8_t> encoding_out(nrows, M);

    // PROFILE_DIST_COMPUTATION("bolt encode", 5, encoding_out.data(), nrows,
    //     bolt_encode<M>(X.data(), nrows, ncols, centroids.data(),
    //                    encoding_out.data()));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt encode", kNtrials,
        encoding_out.data(), nrows,
        bolt_encode<M>(X.data(), nrows, ncols, centroids.data(),
                       encoding_out.data()));
}


TEST_CASE("bolt lut encoding speed", "[bolt][mcq][profile]") {
    static constexpr int nrows = nrows_lut;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> Q(nrows, ncols);
    Q.setRandom();
    ColMatrix<uint8_t> lut_out(ncentroids, ncodebooks);

    // PROFILE_DIST_COMPUTATION_LOOP("bolt encode lut", 5, lut_out.data(),
    //     lut_data_sz, nrows,
    //     bolt_lut<M>(Q.row(i).data(), ncols, centroids.data(), lut_out.data()));
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt encode lut", kNtrials,
        lut_out.data(), lut_data_sz, nrows,
        bolt_lut<M>(Q.row(i).data(), ncols, centroids.data(), lut_out.data()));
}

//static constexpr uint8_t kNumProfileIters = 5;

TEST_CASE("bolt scan speed", "[bolt][mcq][profile]") {
    static constexpr int nblocks = nblocks_scan;
    static constexpr int nrows = nblocks_scan * 32;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create random luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
    luts.setRandom();
    luts = luts.array() / (2 * M); // make max lut value small

    // do the scan to compute the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<uint16_t> dists_u16_safe(nrows);

    // PROFILE_DIST_COMPUTATION("bolt scan uint8", 5, dists_u8.data(), nrows,
    //     bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks));
    // PROFILE_DIST_COMPUTATION("bolt scan uint16", 5, dists_u16.data(), nrows,
    //     bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks));
    // PROFILE_DIST_COMPUTATION("bolt scan uint16 safe", 5, dists_u16_safe.data(), nrows,
    //     bolt_scan<M>(codes.data(), luts.data(), dists_u16_safe.data(), nblocks));

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint8", kNtrials,
        dists_u8.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u8.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16", kNtrials,
        dists_u16.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u16.data(), nblocks));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "bolt scan uint16 safe", kNtrials,
        dists_u16_safe.data(), nrows,
        bolt_scan<M>(codes.data(), luts.data(), dists_u16_safe.data(), nblocks));
}

template<int M, bool Safe=false, class dist_t=void>
void _run_query(const uint8_t* codes, int nblocks,
    const float* q, int ncols,
    const float* centroids,
    uint8_t* lut_out, dist_t* dists_out)
{
    bolt_lut<M>(q, ncols, centroids, lut_out);
    bolt_scan<M, Safe>(codes, lut_out, dists_out, nblocks);
}

TEST_CASE("bolt query (lut + scan) speed", "[bolt][mcq][profile]") {
    static constexpr int nblocks = nblocks_query;
    static constexpr int nrows = nblocks * 32;

    // create random codes from in [0, 15]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();
    codes = codes.array() / 16;

    // create random queries
    RowMatrix<float> Q(nqueries, ncols);
    Q.setRandom();

    // create random centroids
    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();

    // storage for luts
    ColMatrix<uint8_t> luts(ncentroids, ncodebooks);

    SECTION("uint8_t") {
        RowVector<uint8_t> dists(nrows);
        // PROFILE_DIST_COMPUTATION_LOOP("bolt query u8", 5,
        //     dists.data(), nrows, nqueries,
        //     _run_query<M>(codes.data(), nblocks, Q.row(i).data(), ncols,
        //         centroids.data(), luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt query u8", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query<M>(codes.data(), nblocks, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) );
    }
    SECTION("uint16_t") {
        RowVector<uint16_t> dists(nrows);
        // PROFILE_DIST_COMPUTATION_LOOP("bolt query u16", 5,
        //     dists.data(), nrows, nqueries,
        //     _run_query<M>(codes.data(), nblocks, Q.row(i).data(), ncols,
        //         centroids.data(), luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt query u16", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query<M>(codes.data(), nblocks, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) );
    }
    SECTION("uint16_t safe") {
        RowVector<uint16_t> dists(nrows);
        // PROFILE_DIST_COMPUTATION_LOOP("bolt query u16 safe", 5,
        //     dists.data(), nrows, nqueries,
        //     (_run_query<M, true>(codes.data(), nblocks, Q.row(i).data(), ncols,
        //         centroids.data(), luts.data(), dists.data()) ));
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "bolt query u16 safe", kNtrials,
            dists.data(), nrows, nqueries,
            (_run_query<M, true>(codes.data(), nblocks, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) ));
    }
}
