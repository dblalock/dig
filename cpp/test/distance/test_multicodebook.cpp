
#include "catch.hpp"
#include "multi_codebook.hpp"

#include "Dense"

#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "testing_utils.hpp"
#include "debug_utils.hpp"
#include "bit_ops.hpp"

using Eigen::Matrix;
using Eigen::Dynamic;

TEST_CASE("popcnt", "[mcq]") {
    int N = 16; // must be <= 16
    int M = 8;
//    auto X = RowMatrix<uint8_t>(N, M);
//    auto q = RowVector<uint8_t>(M);
//    auto X = RowVector<uint64_t>(N);
//    auto Q = RowVector<uint64_t>(1);
//    X.setRandom();
//    Q.setRandom();
//    uint64_t q = Q(0);
//    uint64_t q = 0x0001020304050607ULL;
//    uint64_t q = 0x0123456789abcdefULL;
    
    // TODO aligned alloc
    uint8_t X_[N * M];
    // set lower 4 bits in each byte to 2,4,...,2M, upper to i
    for (int i = 0; i < N; i++) {
        for (uint8_t j = 0; j < M; j++) {
//            X_[M * i + j] = 2 * j + (i << 4);
            X_[M * i + j] = 2 * j + ((i+1) << 4); // TODO rm
        }
    }
    uint8_t* codes = &X_[0];
    
    uint8_t q_[M];
    for (uint8_t i = 0; i < M; i++) { // successive 4 bits are 0,1,2,...
        q_[i] = (2 * i) + (((2 * i) + 1) << 4);
    }
    uint8_t* q = &q_[0];
    
    std::cout << "q:\n";
    uint64_t q_uint = *(uint64_t*)q;
    dumpEndianBits(q_uint);
    
    // compute distances using our function
    uint8_t dists[N];
//    uint8_t* codes = reinterpret_cast<uint8_t*>(X.data());
    dist::popcount_8B(codes, q_uint, &dists[0], N);
    
    // compute distances by casting to int64 arrays
    std::cout << "bit diffs:\n";
    for (int i = 0; i < N; i++) {
        uint64_t x = *(uint64_t*)(codes + M * i);
        auto diffs = x ^ q_uint;
        dumpEndianBits(diffs);
        int count = popcount(diffs);
//        int count = popcount(x ^ q_uint);
        REQUIRE(count == dists[i]);
//        PRINT_VAR(count);
        std::cout << "global dist: " << count << "\n";
    }
    
    std::cout << "---- computed dists using whole popcount; now using subvects\n";
    
    // compute distances using
    uint8_t dists2[N];
    for (int i = 0; i < N; i++) {
        auto row_ptr = codes + M * i;
        uint64_t x = *(uint64_t*)row_ptr;
        std::cout << "x:\n";
        dumpEndianBits(x);
        uint8_t dist = 0;
        
        for (int j = 0; j < M; j++) {
            uint8_t code = row_ptr[j];
            dumpEndianBits(code, false);
        }
        std::cout << "\n";
        for (int j = 0; j < M; j++) {
            uint8_t code = row_ptr[j];
            printf("%d ", code);
        }
        std::cout << "\n";
        
        for (int j = 0; j < 2*M; j++) {
            
            uint8_t q_bits = static_cast<uint8_t>((q_uint >> (4 * j)) & 0x0F);
            uint8_t x_bits = static_cast<uint8_t>((x      >> (4 * j)) & 0x0F);
            uint8_t count = popcount(x_bits ^ q_bits);
            dist += count;
//            printf("x_bits, q_bits, count: %d, %d, %d\n", x_bits, q_bits, count);
//            dumpBits(x_bits);
//            dumpBits(q_bits);
        }
//        std::cout << "dist: " << count << "\n";
        REQUIRE(dist == (int)dists[i]);
    }
    

//    SECTION("8B lookup table") {
        // 1) define popcnt LUTs
        // 2) ensure that lookups with it are same as popcnt
        // 3) profile both
        // 4) variant of lut function that just uses one LUT, cuz that's all it needs
    
        static const uint8_t mask_low4b = 0x0F;
    
        // tile this so that we instead have a collection of luts
        RowVector<uint8_t> _luts(M * 2 * 16); // get aligned storage
        uint8_t* popcount_luts = _luts.data();
        for (uint8_t j = 0; j < 2*M; j++) {
            REQUIRE(M <= 8);
            uint8_t q_bits = static_cast<uint8_t>((q_uint >> (4 * j)) & mask_low4b);
            auto lut_ptr = popcount_luts + 16 * j;
//            printf("j, q bits: %d, %d\n", j, q_bits); // yep, j == q_bits
            for (uint8_t i = 0; i < 16; i++) {
                lut_ptr[i] = popcount(i ^ q_bits);
//                lut_ptr[15 - i] = popcount(i ^ q_bits);
//                printf("m, i, q, count: %d, %d, %d, %d\n", j, i, q_bits, lut_ptr[i]);
//                REQUIRE(lut_ptr[i] <= 4);
            }
        }
    
//        std::cout << "lut:\n";
//        for (int i = 0; i < 2*M; i+=2) {
//            dumpEndianBits(*(uint64_t*)(popcount_luts + 16 * i));
//            dumpEndianBits(*(uint64_t*)(popcount_luts + 16 * i + 8));
//            std::cout << "\n";
//        }
    
//        uint8_t dists_lut[N];
        RowVector<uint8_t> _dists_lut(N);
        auto dists_lut = _dists_lut.data();
        dist::lut_dists_8B_4b(codes, popcount_luts, dists_lut, N);
    
//        ar::print(ar::mul(dists, 1, N).get(), N);
//        ar::print(ar::mul(dists_lut, 1, N).get(), N);
//        
        for (int i = 0; i < N; i++) {
//            PRINT_VAR(i);
            int d_lut = dists_lut[i];
            int d = dists[i];
            printf("d, d_lut = %d, %d\n", d, d_lut);
//            PRINT_VAR(d_lut);
//            PRINT_VAR(d);
            REQUIRE(d == d_lut);
//            PRINT_VAR((uint)dists_lut[i]);
//            PRINT_VAR((uint)dists[i]);
//            REQUIRE((uint)dists_lut[i] == (uint)dists[i]);
        }
    
//    }
}




