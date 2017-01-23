
#include "catch.hpp"
#include "multi_codebook.hpp"

#include "Dense"

#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "testing_utils.hpp"
#include "debug_utils.hpp"
#include "bit_ops.hpp"
#include "memory.hpp"
//#include "slice.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;

// static const int AlignBytes = 32;

TEST_CASE("popcnt", "[mcq]") {
    int N = 32; // must be multiple of 32 for vectorized lookups
    int M = 8;  // must be 8 for tests that cast to uint64_t

    // TODO aligned alloc if we test vectorized version
    // uint8_t X_[N * M];
    uint8_t* codes = aligned_alloc<uint8_t>(N * M);

    // set lower 4 bits in each byte to 2,4,...,2M, upper to i
    for (int i = 0; i < N; i++) {
        for (uint8_t j = 0; j < M; j++) {
            uint8_t upper_bits = (i % 16) << 4;
            codes[M * i + j] = 2 * j + upper_bits;
        }
    }
    // uint8_t* codes = &X_[0];

    // uint8_t q_[M];
    uint8_t* q = aligned_alloc<uint8_t>(M);
    for (uint8_t i = 0; i < M; i++) { // successive 4 bits are 0,1,2,...
        // q_[i] = (2 * i) + (((2 * i) + 1) << 4);
        uint8_t upper_bits = (((2 * i) + 1) % 16) << 4;
        q[i] = (2 * i) + upper_bits;
    }
    // uint8_t* q = &q_[0];

    std::cout << "q:\n";
    uint64_t q_uint = *(uint64_t*)q;
    dumpEndianBits(q_uint);

    // compute distances using our function
    uint8_t dists[N];
    dist::popcount_8B(codes, q_uint, &dists[0], N);

    // compute distances by casting to int64 arrays
    // std::cout << "bit diffs:\n";
    for (int i = 0; i < N; i++) {
        uint64_t x = *(uint64_t*)(codes + M * i);
        auto diffs = x ^ q_uint;
        // dumpEndianBits(diffs);
        int count = popcount(diffs);
        REQUIRE(count == dists[i]);
        // std::cout << "global dist: " << count << "\n";
    }

//     std::cout << "---- computed dists using whole popcount; now using subvects\n";

//     // compute distances using blocks of 4 bits
//     uint8_t dists2[N];
//     for (int i = 0; i < N; i++) {
//         auto row_ptr = codes + M * i;
//         uint64_t x = *(uint64_t*)row_ptr;
//         std::cout << "x:\n";
//         dumpEndianBits(x);
//         uint8_t dist = 0;

//         for (int j = 0; j < M; j++) {
//             uint8_t code = row_ptr[j];
//             dumpEndianBits(code, false);
//         }
//         std::cout << "\n";
//         for (int j = 0; j < M; j++) {
//             uint8_t code = row_ptr[j];
//             printf("%d ", code);
//         }
//         std::cout << "\n";

//         for (int j = 0; j < 2*M; j++) {
//             // XXX; this will break on a big-endian machine because the
//             // first bytes in the array won't be the low bits of q and x
//             uint8_t q_bits = static_cast<uint8_t>((q_uint >> (4 * j)) & 0x0F);
//             uint8_t x_bits = static_cast<uint8_t>((x      >> (4 * j)) & 0x0F);
//             uint8_t count = popcount(x_bits ^ q_bits);
//             dist += count;
// //            printf("x_bits, q_bits, count: %d, %d, %d\n", x_bits, q_bits, count);
// //            dumpBits(x_bits);
// //            dumpBits(q_bits);
//         }
// //        std::cout << "dist: " << count << "\n";
//         REQUIRE(dist == (int)dists[i]);
//     }


    SECTION("8B codes, non-vectorized 4b lookup table") {
        // 1) define popcnt LUTs
        // 2) ensure that lookups with it are same as popcnt
        // 3) profile both
        // 4) variant of lut function that just uses one LUT, cuz that's all it needs
        
        static const uint8_t mask_low4b = 0x0F;

        // tile this so that we instead have a collection of luts
        uint8_t* popcount_luts = aligned_alloc<uint8_t>(M * 2 * 16);
        for (uint8_t j = 0; j < 2*M; j++) {
            REQUIRE(M <= 8);
            uint8_t q_bits = static_cast<uint8_t>((q_uint >> (4 * j)) & mask_low4b);
            auto lut_ptr = popcount_luts + 16 * j;
//            printf("j, q bits: %d, %d\n", j, q_bits); // yep, j == q_bits
            for (uint8_t i = 0; i < 16; i++) {
                lut_ptr[i] = popcount(i ^ q_bits);
            }
        }

        RowVector<uint8_t> _dists_lut(N);
        auto dists_lut = _dists_lut.data();
        dist::lut_dists_8B_4b(codes, popcount_luts, dists_lut, N);

        for (int i = 0; i < N; i++) {
            int d_lut = dists_lut[i];
            int d = dists[i];
            printf("d, d_lut = %d, %d\n", d, d_lut);
            REQUIRE(d == d_lut);
        }
        
        aligned_free(popcount_luts);
    }
    
    SECTION("8B codes, vectorized lookup table") {
        REQUIRE(true);
        
        static const uint8_t mask_low4b = 0x0F;
        static const uint8_t block_sz_rows = 32;
        
        int nblocks = N / block_sz_rows;
        assert(N % block_sz_rows == 0);
        
        uint8_t* popcount_luts = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);
        uint8_t* block_codes = aligned_alloc<uint8_t>(N * M);
        
        // copy row-major codes to col-major in blocks of 32
        for (int nn = 0; nn < nblocks; nn++) { // for each block
            auto block_start_idx = nn * block_sz_rows * M;
            auto in_block_ptr = codes + block_start_idx;
            auto out_block_ptr = block_codes + block_start_idx;
            for (int i = 0; i < block_sz_rows; i++) {  // for each row
                for (int j = 0; j < M; j++) {           // for each col
                    auto in_ptr = in_block_ptr + (i * M) + j;
                    auto out_ptr = out_block_ptr + (j * N) + i;
                    *out_ptr = *in_ptr;
                }
            }
        }
        
        // create 32B luts for vectorized lookups
        REQUIRE(block_sz_rows == 32); // following loop assumes this is true
        for (uint8_t j = 0; j < M; j++) {
            uint8_t byte = q[j];
            uint8_t low_bits = byte & mask_low4b;
            uint8_t high_bits = byte >> 4;
            auto lut_ptr = popcount_luts + block_sz_rows * 2 * j;
            for (uint8_t i = 0; i < 16; i++) {
                lut_ptr[i +  0] = popcount(i ^ low_bits);
                lut_ptr[i + 16] = popcount(i ^ low_bits);
                lut_ptr[i + 32] = popcount(i ^ high_bits);
                lut_ptr[i + 48] = popcount(i ^ high_bits);
            }
        }
        
        // compute vectorized dists
        uint8_t* dists_vect = aligned_alloc<uint8_t>(N);
//        dist::block_lut_dists_32x8B_4b(block_codes, popcount_luts, dists_vect, nblocks);
        dist::naive_block_lut_dists_32x8B_4b(block_codes, popcount_luts, dists_vect, nblocks);
        
        // check whether we got the dists right
        for (int i = 0; i < N; i++) {
            int d_vect = dists_vect[i];
            int d = dists[i];
            printf("d, d_vect = %d, %d\n", d, d_vect);
//            REQUIRE(d == d_vect);
        }
        REQUIRE(true); // TODO rm
        
        aligned_free<uint8_t>(popcount_luts);
        aligned_free<uint8_t>(block_codes);
    }
    
    
    aligned_free<uint8_t>(codes);
    aligned_free<uint8_t>(q);
}




