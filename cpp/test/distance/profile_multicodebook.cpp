#include "catch.hpp"
#include "multi_codebook.hpp"

//#include "Dense"

#include "timing_utils.hpp"
#include "bit_ops.hpp"
#include "memory.hpp"

template<class DistT>
void prevent_optimizing_away_dists(DistT* dists, int64_t N) {
    volatile bool foo = true;
    for (int64_t n = 0; n < N; n++) { foo &= dists[n] > 42; }
    if (foo) { std::cout << " "; }
}

// TODO split this into smaller functions and also call from other test file
TEST_CASE("popcnt_timing", "[mcq][profile]") {
//    int nblocks = 5 * 1000 * 1000;
    // static constexpr int nblocks = 1000 * 1000;
    static constexpr int nblocks = 100 * 1000;
    int64_t N = 32 * nblocks;
    static constexpr int M = 16;
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

    // create 32B LUTs (two copies, pre-unpacked) and 16B LUTs; we also create
    // a 256-element LUT for 8bit codes and non-vectorized search
    REQUIRE(block_sz_rows == 32);
    uint8_t* popcount_luts16 = aligned_alloc<uint8_t>(M * block_sz_rows);
    uint8_t* popcount_luts32 = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);
    uint8_t* popcount_luts256b = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);

    uint16_t* popcount_luts256s = aligned_alloc<uint16_t>(M * 256);
    uint32_t* popcount_luts256i = aligned_alloc<uint32_t>(M * 256);
    float* popcount_luts256f = aligned_alloc<float>(M * 256);

    uint8_t low_counts[16];
    uint8_t high_counts[16];
    for (uint8_t j = 0; j < M; j++) {
        uint8_t byte = q[j];
        uint8_t low_bits = byte & mask_low4b;
        uint8_t high_bits = byte >> 4;

//        uint8_t low_count = popcount(i ^ low_bits);
//        uint8_t high_count = popcount(i ^ low_bits);
//        for (uint8_t i = 0; i < 16; i++) {
//            low_counts[i]
//        }

        // just a sequence of 16B LUTs
        auto lut_ptr16 = popcount_luts16 + 16 * 2 * j;
        for (uint8_t i = 0; i < 16; i++) {
            lut_ptr16[i +  0] = popcount(i ^ low_bits);
            lut_ptr16[i + 16] = popcount(i ^ high_bits);
        }

        // two consecutive copies of each LUT, to fill 32B
        auto lut_ptr32 = popcount_luts32 + block_sz_rows * 2 * j;
        for (uint8_t i = 0; i < 16; i++) {
            lut_ptr32[i +  0] = popcount(i ^ low_bits);
            lut_ptr32[i + 16] = popcount(i ^ low_bits);
            lut_ptr32[i + 32] = popcount(i ^ high_bits);
            lut_ptr32[i + 48] = popcount(i ^ high_bits);
        }

        // 256bit LUTs (of various types) for 8bit codes
        auto lut_ptr_b = popcount_luts256b + 16 * j;
        auto lut_ptr_s = popcount_luts256s + 16 * j;
        auto lut_ptr_i = popcount_luts256i + 16 * j;
        auto lut_ptr_f = popcount_luts256f + 16 * j;
        for (uint16_t i = 0; i < 256; i++) {
            uint8_t count = popcount(static_cast<uint8_t>(i) ^ byte);
            lut_ptr_b[i] = count;
            lut_ptr_s[i] = count;
            lut_ptr_i[i] = count;
            lut_ptr_f[i] = count;
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


    // ================================================================ timing

//    double t_popcnt = 0, t_scalar = 0, t_vector = 0, t_vector_broken = 0;
    double t = 0;

    std::cout << "starting searches...\n";

    // ------------------------------------------------ 4bit codes

    std::cout << "-------- dists with 4bit codes\n";
    
    { // compute distances using popcnt instruction
        uint64_t q_uint = *(uint64_t*)q;
        EasyTimer _(t);
        dist::popcount_8B(codes, q_uint, dists_popcnt, N);
    }
    printf("t_popcnt: %.2f (%.1fM/s)\n", t, N_millions / (t/1e3));


    // scalar version omitted for now because it takes 10x longer
    { // compute distances using one lookup at a time, vertical mem layout
        EasyTimer _(t);
        dist::debug_lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_scalar, nblocks);
    }
    printf("t_scalar: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));


    { // compute vectorized distances
        EasyTimer _(t);
        dist::lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_vector, nblocks);
    }
    printf("t_vector: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    { // compute vectorized distances with impl that unpacks 16B luts into 32B
        EasyTimer _(t);
        dist::lut_dists_block32_4b_unpack<M>(block_codes, popcount_luts16, dists_unpack, nblocks);
    }
    printf("t_unpack: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    { // compute vectorized distances with experimental (incorrect) impl
        EasyTimer _(t);
        dist::incorrect_lut_dists_block32_4b<M>(block_codes, popcount_luts16, dists_incorrect, nblocks);
    }
    printf("t_vector_broken: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));


    // ------------------------------------------------ 8bit codes

    std::cout << "-------- dists with 8bit codes\n";
    
    uint8_t*  dists_8b_b = aligned_alloc<uint8_t>(N);
    uint16_t* dists_8b_s = aligned_alloc<uint16_t>(N);
    uint32_t* dists_8b_i = aligned_alloc<uint32_t>(N);
    float*    dists_8b_f = aligned_alloc<float>(N);
    
    { // 8bit lookups, 8bit distances
        EasyTimer _(t);
        dist::lut_dists_8b<M>(codes, popcount_luts256b, dists_8b_b, N);
    }
    prevent_optimizing_away_dists(dists_8b_b, N);
    printf("t 8b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    { // 8bit lookups, 16bit distances
        EasyTimer _(t);
        dist::lut_dists_8b<M>(codes, popcount_luts256s, dists_8b_s, N);
    }
    prevent_optimizing_away_dists(dists_8b_s, N);
    printf("t 8b, 16b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    { // 8bit lookups, 32bit distances
        EasyTimer _(t);
        dist::lut_dists_8b<M>(codes, popcount_luts256i, dists_8b_i, N);
    }
    prevent_optimizing_away_dists(dists_8b_i, N);
    printf("t 8b, 32b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    { // 8bit lookups, 32bit float distances
        EasyTimer _(t);
        dist::lut_dists_8b<M>(codes, popcount_luts256f, dists_8b_f, N);
    }
    prevent_optimizing_away_dists(dists_8b_f, N);
    printf("t 8b, float dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    // ------------------------------------------------ <NOTE>
    // below this point, no effort made at achieving correctness;
    // scans are operating on unitialized memory
    // ------------------------------------------------ </NOTE>
    
    // ------------------------------------------------ 12bit codes
    
    std::cout << "-------- dists with 12bit codes\n";
    
    int8_t*  dists_12b_b = aligned_alloc<int8_t>(N);
    int16_t*  dists_12b_s = aligned_alloc<int16_t>(N);
    int32_t*  dists_12b_i = aligned_alloc<int32_t>(N);
    
    int lut_sz_12b = (1 << 12);
    int8_t* popcount_luts12b_b = aligned_alloc<int8_t>(M * lut_sz_12b);
    int16_t* popcount_luts12b_s = aligned_alloc<int16_t>(M * lut_sz_12b);
    int32_t* popcount_luts12b_i = aligned_alloc<int32_t>(M * lut_sz_12b);
    
    { // 12bit lookups, 8bit distances
        EasyTimer _(t);
        dist::lut_dists_12b<M>(codes, popcount_luts12b_b, dists_12b_b, N);
    }
    prevent_optimizing_away_dists(dists_12b_b, N);
    printf("t 12b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    { // 12bit lookups, 16bit distances
        EasyTimer _(t);
        dist::lut_dists_12b<M>(codes, popcount_luts12b_s, dists_12b_s, N);
    }
    prevent_optimizing_away_dists(dists_12b_b, N);
    printf("t 12b, 16b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    { // 12bit lookups, 32bit distances
        EasyTimer _(t);
        dist::lut_dists_12b<M>(codes, popcount_luts12b_i, dists_12b_i, N);
    }
    prevent_optimizing_away_dists(dists_12b_b, N);
    printf("t 12b, 32b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    // ------------------------------------------------ 16bit codes
    // NOTE: no effort made at achieving correctness; unitialized memory
    
    std::cout << "-------- dists with 16bit codes\n";
    
    int8_t*  dists_16b_b = aligned_alloc<int8_t>(N);
    int16_t*  dists_16b_s = aligned_alloc<int16_t>(N);
    int32_t*  dists_16b_i = aligned_alloc<int32_t>(N);
    
    int lut_sz_16b = (1 << 16);
    int8_t* popcount_luts16b_b = aligned_alloc<int8_t>(M * lut_sz_16b);
    int16_t* popcount_luts16b_s = aligned_alloc<int16_t>(M * lut_sz_16b);
    int32_t* popcount_luts16b_i = aligned_alloc<int32_t>(M * lut_sz_16b);
    
    uint16_t* codes16 = (uint16_t*)codes;
    
    { // 12bit lookups, 8bit distances
        EasyTimer _(t);
        dist::lut_dists_16b<M>(codes16, popcount_luts16b_b, dists_16b_b, N);
    }
    prevent_optimizing_away_dists(dists_16b_b, N);
    printf("t 16b, 8b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    { // 12bit lookups, 16bit distances
        EasyTimer _(t);
        dist::lut_dists_16b<M>(codes16, popcount_luts16b_s, dists_16b_s, N);
    }
    prevent_optimizing_away_dists(dists_16b_s, N);
    printf("t 16b, 16b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    { // 12bit lookups, 16bit distances
        EasyTimer _(t);
        dist::lut_dists_16b<M>(codes16, popcount_luts16b_i, dists_16b_i, N);
    }
    prevent_optimizing_away_dists(dists_16b_i, N);
    printf("t 16b, 32b dist: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));
    
    // ------------------------------------------------ floats
    // NOTE: no effort made at achieving correctness; unitialized memory
    
    static constexpr uint64_t nblocksf = 10000;
    static constexpr uint64_t Nf = 256 * nblocksf;
    static constexpr double Nf_millions = static_cast<double>(Nf) / 1e6;
    static constexpr int Df = 64;
    float* Xf = aligned_alloc<float>(Nf * Df);
    float* qf = aligned_alloc<float>(Df);
    float* distsf = aligned_alloc<float>(Nf);
    
    std::cout << "-------- float distances where D = " << Df << "\n";
    
    { // operate on 8 floats per col (1 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<8, Df>(Xf, qf, distsf, nblocksf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 8 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));
    
    { // operate on 16 floats per col (2 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<16, Df>(Xf, qf, distsf, nblocksf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 16 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    { // operate on 32 floats per col (4 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<32, Df>(Xf, qf, distsf, nblocksf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 32 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));
    
    { // operate on 128 floats per col (4 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<128, Df>(Xf, qf, distsf, nblocksf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 128 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));
    
    
    aligned_free(Xf);
    aligned_free(qf);
    aligned_free(distsf);
    
    // ------------------------------------------------ int8s
    static constexpr uint64_t nblocksb = 100000;
    static constexpr uint64_t Nb = 256 * nblocksb;
    static constexpr double Nb_millions = static_cast<double>(Nb) / 1e6;
    static constexpr int Db = 64;
    int8_t* Xb = aligned_alloc<int8_t>(Nb * Db);
    int8_t* qb = aligned_alloc<int8_t>(Db);
    uint16_t* distsb = aligned_alloc<uint16_t>(Nb);
    
    std::cout << "-------- int8 distances where D = " << Db << "\n";
    
    { // operate on 32 int8s per col (1 stripe)
        EasyTimer _(t);
        dist::byte_dists_vertical<32, Db>(Xb, qb, distsb, nblocksb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 32 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
    
    { // operate on 64 int8s per col (2 stripes)
        EasyTimer _(t);
        dist::byte_dists_vertical<64, Db>(Xb, qb, distsb, nblocksb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 64 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
    
    { // operate on 128 int8s per col (4 stripes)
        EasyTimer _(t);
        dist::byte_dists_vertical<128, Db>(Xb, qb, distsb, nblocksb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 128 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
    
    // ------------------------------------------------ correctness

    
    bool same = true;
    bool same2 = true;
    for (int64_t n = 0; n < N; n++) {
//        same &= dists_popcnt[n] == dists_scalar[n];
        same &= dists_popcnt[n] == dists_vector[n];
        same2 &= dists_popcnt[n] == dists_unpack[n];
//        same2 &= dists_popcnt[n] == dists_incorrect[n];
//        REQUIRE(dists_popcnt[n] == dists_scalar[n]);
//        REQUIRE(dists_popcnt[n] == dists_vector[n]);
    }
    std::cout << "same? " << same << std::endl;
    std::cout << "same2? " << same2 << std::endl; // so stores don't get optimized away
    REQUIRE(same); // rarely, but occasionally fails?
    REQUIRE(same2); // rarely, but occasionally fails?
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
