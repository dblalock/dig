#include "catch.hpp"
#include "multi_codebook.hpp"

//#include "Dense"

#include "timing_utils.hpp"
#include "bit_ops.hpp"
#include "memory.hpp"

// TODO split this into smaller functions and also call from other test file
TEST_CASE("popcnt_timing", "[mcq][profile]") {
//    int nblocks = 5 * 1000 * 1000;
    int nblocks = 1000 * 1000;
    int64_t N = 32 * nblocks;
    int M = 8; // popcnt only implemented for 1,2,4,8B
    auto N_millions = static_cast<double>(N) / 1e6;
    printf("searching through %.3f million vecs (%.3fMB)...\n", N_millions, N_millions * M);
    
    static const uint8_t mask_low4b = 0x0F;
    static const uint8_t block_sz_rows = 32;

    // random database of codes
//    RowVector<uint8_t> codes_rowmajor_(N * M);
//    codes_rowmajor_.setRandom();
//    uint8_t* codes = codes_rowmajor_.data();
    uint8_t* codes = aligned_alloc<uint8_t>(N * M);

    // random query
//    RowVector<uint8_t> q_(M);
//    q_.setRandom();
//    uint8_t* q = q_.data();
    uint8_t* q = aligned_alloc<uint8_t>(M);

    // create 32B LUTs (two copies, pre-unpacked) and 16B LUTs
    REQUIRE(block_sz_rows == 32);
    uint8_t* popcount_luts16 = aligned_alloc<uint8_t>(M * block_sz_rows);
    uint8_t* popcount_luts32 = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);
    for (uint8_t j = 0; j < M; j++) {
        uint8_t byte = q[j];
        uint8_t low_bits = byte & mask_low4b;
        uint8_t high_bits = byte >> 4;

        // non-vectorized; just a sequence of 16B LUTs
        auto lut_ptr_scalar = popcount_luts16 + 16 * 2 * j;
        for (uint8_t i = 0; i < 16; i++) {
            lut_ptr_scalar[i +  0] = popcount(i ^ low_bits);
            lut_ptr_scalar[i + 16] = popcount(i ^ high_bits);
        }

        // vectorized; two consecutive copies of each LUT, to fill 32B
        auto lut_ptr = popcount_luts32 + block_sz_rows * 2 * j;
        for (uint8_t i = 0; i < 16; i++) {
            lut_ptr[i +  0] = popcount(i ^ low_bits);
            lut_ptr[i + 16] = popcount(i ^ low_bits);
            lut_ptr[i + 32] = popcount(i ^ high_bits);
            lut_ptr[i + 48] = popcount(i ^ high_bits);
        }
    }

    // create block vertical layout version of codes
    uint8_t* block_codes = aligned_alloc<uint8_t>(N * M);
    for (int nn = 0; nn < nblocks; nn++) { // for each block
        auto block_start_idx = nn * block_sz_rows * M;
        auto in_block_ptr = codes + block_start_idx;
        auto out_block_ptr = block_codes + block_start_idx;
        for (int i = 0; i < block_sz_rows; i++) {  // for each row
            for (int j = 0; j < M; j++) {           // for each col
                auto in_ptr = in_block_ptr + (i * M) + j;
                auto out_ptr = out_block_ptr + (j * block_sz_rows) + i;
                *out_ptr = *in_ptr;
            }
        }
    }

    // store distances from each method; first two don't actually need align,
    // but as well give all of them identical treatment
    uint8_t* dists_popcnt = aligned_alloc<uint8_t>(N);
    uint8_t* dists_scalar = aligned_alloc<uint8_t>(N);
    uint8_t* dists_vector = aligned_alloc<uint8_t>(N);
    uint8_t* dists_unpack = aligned_alloc<uint8_t>(N);
    uint8_t* dists_incorrect = aligned_alloc<uint8_t>(N);

    // ------------------------------------------------ timing

//    double t_popcnt = 0, t_scalar = 0, t_vector = 0, t_vector_broken = 0;
    double t = 0;

    std::cout << "starting searches...\n";
    
    { // compute distances using popcnt instruction
        uint64_t q_uint = *(uint64_t*)q;
        EasyTimer _(t);
        dist::popcount_8B(codes, q_uint, dists_popcnt, N);
    }
//    std::cout << "t popcnt: " << t_popcnt << "\n";
    printf("t_popcnt: %.2f (%.1fM/s)\n", t, N_millions / (t/1e3));
    
    
    // scalar version omitted for now because it takes 10x longer
    { // compute distances using one lookup at a time, vertical mem layout
//        std::cout << "scalar search:\n";
        EasyTimer _(t);
        dist::naive_block_lut_dists_32x8B_4b(block_codes, popcount_luts32, dists_scalar, nblocks);
    }
    printf("t_scalar: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    
    { // compute vectorized distances
        EasyTimer _(t);
        dist::block_lut_dists_32x8B_4b(block_codes, popcount_luts32, dists_vector, nblocks);
//        dist::block_lut_dists_32x8B_4b(block_codes, popcount_luts_scalar, dists_vector, nblocks);
    }
    printf("t_vector: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
//    std::cout << "t vector: " << t_vector << "(" << N_millions / (t_vector/1e3) << "M vects/s) \n";
    
    { // compute vectorized distances with experimental (incorrect) impl
        EasyTimer _(t);
        dist::block_lut_dists_32x8B_4b_unpack(block_codes, popcount_luts16, dists_unpack, nblocks);
    }
    printf("t_unpack: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    { // compute vectorized distances with experimental (incorrect) impl
        EasyTimer _(t);
        dist::incorrect_block_lut_dists_32x8B_4b(block_codes, popcount_luts16, dists_incorrect, nblocks);
    }
    printf("t_vector_broken: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    bool same = true;
    bool same2 = true;
    for (int n = 0; n < N; n++) {
//        same &= dists_popcnt[n] == dists_scalar[n];
        same &= dists_popcnt[n] == dists_vector[n];
        same2 &= dists_popcnt[n] == dists_unpack[n];
//        same2 &= dists_popcnt[n] == dists_incorrect[n];
//        REQUIRE(dists_popcnt[n] == dists_scalar[n]);
//        REQUIRE(dists_popcnt[n] == dists_vector[n]);
    }
    REQUIRE(same); // rarely, but occasionally fails?
    REQUIRE(same2); // rarely, but occasionally fails?
    std::cout << "same? " << same << std::endl;
    std::cout << "same2? " << same2 << std::endl; // so stores don't get optimized away
    std::cout << "done" << std::endl;

    
    aligned_free<uint8_t>(q);
    aligned_free<uint8_t>(codes);
    aligned_free<uint8_t>(block_codes);
    aligned_free<uint8_t>(popcount_luts16);
    aligned_free<uint8_t>(popcount_luts32);
    aligned_free<uint8_t>(dists_popcnt);
    aligned_free<uint8_t>(dists_scalar);
    aligned_free<uint8_t>(dists_vector);
}
