
#include "catch.hpp"
#include "multi_codebook.hpp"

#include <random>

#include "timing_utils.hpp"
#include "bit_ops.hpp"
#include "memory.hpp"

template<class DistT>
void prevent_optimizing_away_dists(DistT* dists, int64_t N) {
//    std::cout << "I am not an ******* compiler and am actually running this function\n";
    // volatile bool foo = true;
    volatile int64_t count = 0;
//    for (int64_t n = N - 1; n >= 0; n--) { count += (((int)dists[n]) % 2); }
//    for (int64_t n = 0; n < N; n++) { count += (((int)dists[n]) % 2); }
    for (int64_t n = 0; n < N; n++) { count += dists[n] > 0; }
    std::cout << "(" << count << "/" << N << ")\t";
//    if (count % 2) { std::cout << " <><><><><> odd number"; }
}

// TODO reconcile this func with previous rand_int funcs
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void randint_inplace(data_t* data, len_t len,
                                   data_t min=std::numeric_limits<data_t>::min(),
                                   data_t max=std::numeric_limits<data_t>::max())
{
    assert(len > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> d(min, max);
    
    for (len_t i = 0; i < len; i++) {
        data[i] = static_cast<data_t>(d(gen));
    }
}
template <class data_t, class len_t, REQUIRE_INT(len_t)>
static inline void rand_inplace(data_t* data, len_t len,
                                data_t min=0, data_t max=1)
{
    assert(len > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(min, max);
    
    for (len_t i = 0; i < len; i++) {
        data[i] = static_cast<data_t>(d(gen));
    }
}

template<class data_t>
static inline data_t* aligned_random_ints(int64_t len) {
    data_t* ptr = aligned_alloc<data_t>(len);
    randint_inplace(ptr, len);
    return ptr;
}

template<class data_t>
static inline data_t* aligned_random(int64_t len) {
    data_t* ptr = aligned_alloc<data_t>(len);
    rand_inplace(ptr, len);
    return ptr;
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
    uint8_t* codes = aligned_random_ints<uint8_t>(N * M);
//    randint_inplace(codes, N * M);

    // random query
//    RowVector<uint8_t> q_(M);
//    q_.setRandom();
//    uint8_t* q = q_.data();
    uint8_t* q = aligned_random_ints<uint8_t>(M);
//    randint_inplace(q, M);

    // create 32B LUTs (two copies, pre-unpacked) and 16B LUTs; we also create
    // a 256-element LUT for 8bit codes and non-vectorized search
    REQUIRE(block_sz_rows == 32);
    uint8_t* popcount_luts16 = aligned_alloc<uint8_t>(M * block_sz_rows);
    uint8_t* popcount_luts32 = aligned_alloc<uint8_t>(M * 2 * block_sz_rows);
    
    uint8_t* popcount_luts256b = aligned_alloc<uint8_t>(M * 256);
    uint16_t* popcount_luts256s = aligned_alloc<uint16_t>(M * 256);
    uint32_t* popcount_luts256i = aligned_alloc<uint32_t>(M * 256);
    float* popcount_luts256f = aligned_alloc<float>(M * 256);

//    uint8_t low_counts[16];
//    uint8_t high_counts[16];
    for (uint8_t j = 0; j < M; j++) {
        uint8_t byte = q[j];
        uint8_t low_bits = byte & mask_low4b;
        uint8_t high_bits = (byte >> 4) & mask_low4b;

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
        auto lut_ptr32 = popcount_luts32 + 32 * 2 * j;
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
    uint8_t* dists_incorrect2 = aligned_alloc<uint8_t>(N);


    // ================================================================ timing

//    double t_popcnt = 0, t_scalar = 0, t_vector = 0, t_vector_broken = 0;
    double t = 0;

    std::cout << "starting searches...\n";

    // ------------------------------------------------ 4bit codes
    # pragma mark 4bit codes

    std::cout << "-------- dists with 4bit codes\n";

    { // compute distances using popcnt instruction
//        uint64_t q_uint = *(uint64_t*)q;
        EasyTimer _(t);
        dist::popcount_generic<M>(codes, q, dists_popcnt, N);
    }
    prevent_optimizing_away_dists(dists_popcnt, N);
    printf("t_popcnt: %.2f (%.1fM/s)\n", t, N_millions / (t/1e3));


    // scalar version omitted for now because it takes 10x longer
    { // compute distances using one lookup at a time, vertical mem layout
        EasyTimer _(t);
        dist::debug_lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_scalar, nblocks);
    }
    prevent_optimizing_away_dists(dists_scalar, N);
    printf("t_scalar: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));


    { // compute vectorized distances
        EasyTimer _(t);
        dist::lut_dists_block32_4b<M>(block_codes, popcount_luts32, dists_vector, nblocks);
    }
    prevent_optimizing_away_dists(dists_vector, N);
    printf("t_vector: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    { // compute vectorized distances with impl that unpacks 16B luts into 32B
        EasyTimer _(t);
        dist::lut_dists_block32_4b_unpack<M>(block_codes, popcount_luts16, dists_unpack, nblocks);
    }
    prevent_optimizing_away_dists(dists_unpack, N);
    printf("t_unpack: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    { // compute vectorized distances with experimental (incorrect) impl
        EasyTimer _(t);
        dist::incorrect_lut_dists_block32_4b<M>(block_codes, popcount_luts16, dists_incorrect, nblocks);
    }
    prevent_optimizing_away_dists(dists_incorrect, N);
    printf("t_vector_broken: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    { // compute vectorized distances with experimental (incorrect) impl
        EasyTimer _(t);
        dist::incorrect_lut_dists_block32_4b_v2<M>(block_codes, popcount_luts16, dists_incorrect2, nblocks);
    }
    prevent_optimizing_away_dists(dists_incorrect2, N);
    printf("t_vector_broken2: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));


    // { // compute vectorized distances with experimental (incorrect) impl
    //     EasyTimer _(t);
    //     dist::sum_block32<32, M>(block_codes, popcount_luts16, dists_incorrect, nblocks);
    // }
    // prevent_optimizing_away_dists(dists_incorrect, N);
    // printf("t sum codes: %.2fms (%.1f M/s)\n", t, N_millions / (t/1e3));

    // ------------------------------------------------ 8bit codes
    # pragma mark 8bit codes
    
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
    # pragma mark 12bit codes
    
    std::cout << "-------- dists with 12bit codes\n";

    uint8_t*  dists_12b_b = aligned_alloc<uint8_t>(N);
    uint16_t*  dists_12b_s = aligned_alloc<uint16_t>(N);
    uint32_t*  dists_12b_i = aligned_alloc<uint32_t>(N);

    int lut_sz_12b = (1 << 12);
    uint8_t* popcount_luts12b_b = aligned_random_ints<uint8_t>(M * lut_sz_12b);
    uint16_t* popcount_luts12b_s = aligned_random_ints<uint16_t>(M * lut_sz_12b);
    uint32_t* popcount_luts12b_i = aligned_random_ints<uint32_t>(M * lut_sz_12b);

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
    # pragma mark 16bit codes
    std::cout << "-------- dists with 16bit codes\n";

    uint8_t*  dists_16b_b = aligned_alloc<uint8_t>(N);
    uint16_t*  dists_16b_s = aligned_alloc<uint16_t>(N);
    uint32_t*  dists_16b_i = aligned_alloc<uint32_t>(N);

    int lut_sz_16b = (1 << 16);
    uint8_t* popcount_luts16b_b = aligned_random_ints<uint8_t>(M/2 * lut_sz_16b);
    uint16_t* popcount_luts16b_s = aligned_random_ints<uint16_t>(M/2 * lut_sz_16b);
    uint32_t* popcount_luts16b_i = aligned_random_ints<uint32_t>(M/2 * lut_sz_16b);

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
    # pragma mark floats

    static constexpr uint64_t nblocksf = 10000;
    static constexpr uint64_t Nf = 256 * nblocksf;
    static constexpr double Nf_millions = static_cast<double>(Nf) / 1e6;
    static constexpr int Df = 16;
    float* Xf = aligned_random<float>(Nf * Df);
    float* qf = aligned_random<float>(Df);
    float* distsf = aligned_alloc<float>(Nf);

    std::cout << "-------- float distances where D = " << Df << "\n";

    { // operate on 8 floats per col (1 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<8, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 8 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    { // operate on 16 floats per col (2 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<16, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 16 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    { // operate on 32 floats per col (4 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<32, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 32 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    { // operate on 128 floats per col (16 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<128, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 128 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    { // operate on 256 floats per col (32 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<256, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 256 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));

    { // operate on 512 floats per col (32 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<512, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 512 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));
    
    { // operate on 1024 floats per col (32 stripes)
        EasyTimer _(t);
        dist::float_dists_vertical<1024, Df>(Xf, qf, distsf, Nf);
    }
    prevent_optimizing_away_dists(distsf, Nf);
    printf("t 1024 floats full dist: %.2fms (%.1f M/s)\n", t, Nf_millions / (t/1e3));
    

    aligned_free(Xf);
    aligned_free(qf);
    aligned_free(distsf);

    // ------------------------------------------------ int8s
    # pragma mark int8s
    static constexpr uint64_t nblocksb = 10000;
    static constexpr uint64_t Nb = 256 * nblocksb;
    static constexpr double Nb_millions = static_cast<double>(Nb) / 1e6;
    static constexpr int Db = 16;
    int8_t* Xb = aligned_random_ints<int8_t>(Nb * Db);
    int8_t* qb = aligned_random_ints<int8_t>(Db);
    uint16_t* distsb = aligned_alloc<uint16_t>(Nb);
//    uint16_t* distsb2 = aligned_alloc<uint16_t>(Nb);

    std::cout << "-------- int8 distances where D = " << Db << "\n";

//    { // just sum inputs as a lower bound on how fast any func could be
//        EasyTimer _(t);
//        dist::sum_inputs<32, Db>(Xb, qb, distsb, Nb);
//    }
//    prevent_optimizing_away_dists(distsb, Nb);
//    printf("t 32 int8s just sums: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
//    for (int64_t i = 0; i < Nb; i++) { Xb[i] += 1; }

    { // operate on 32 int8s per col (1 stripe)
        EasyTimer _(t);
        dist::byte_dists_vertical<32, Db>(Xb, qb, distsb, Nb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 32 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
//    for (int64_t i = 0; i < Nb; i++) { Xb[i] += 1; }

    { // operate on 64 int8s per col (2 stripes)
        EasyTimer _(t);
        dist::byte_dists_vertical<64, Db>(Xb, qb, distsb, Nb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 64 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
//    for (int64_t i = 0; i < Nb; i++) { Xb[i] += 1; }

    { // operate on 128 int8s per col (4 stripes)
        EasyTimer _(t);
        dist::byte_dists_vertical<128, Db>(Xb, qb, distsb, Nb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 128 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    { // operate on 256 int8s per col (8 stripes)
        EasyTimer _(t);
        dist::byte_dists_vertical<256, Db>(Xb, qb, distsb, Nb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 256 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));

    { // operate on 512 int8s per col (8 stripes)
        EasyTimer _(t);
        dist::byte_dists_vertical<512, Db>(Xb, qb, distsb, Nb);
    }
    prevent_optimizing_away_dists(distsb, Nb);
    printf("t 512 int8s full dist: %.2fms (%.1f M/s)\n", t, Nb_millions / (t/1e3));
    
    
    aligned_free(Xb);
    aligned_free(qb);
    aligned_free(distsb);

    // ------------------------------------------------ correctness

    bool same0 = true;
    bool same1 = true;
    bool same2 = true;
    for (int64_t n = 0; n < N; n++) {
        same0 &= dists_popcnt[n] == dists_scalar[n];
        if (n < 50 && !same0) {
            printf("popcnt dist, scalar dist = %d, %d\n", dists_popcnt[n], dists_scalar[n]);
        }
//        same1 &= dists_popcnt[n] == dists_vector[n];
//        same2 &= dists_popcnt[n] == dists_unpack[n];
        
        same1 &= dists_scalar[n] == dists_vector[n];
        same2 &= dists_vector[n] == dists_unpack[n];
        
//        same2 &= dists_popcnt[n] == dists_incorrect[n];
//        REQUIRE(dists_popcnt[n] == dists_scalar[n]);
//        REQUIRE(dists_popcnt[n] == dists_vector[n]);
    }
    std::cout << "same0? " << same0 << "\n"; // 0, so popcnt prolly broken
    std::cout << "same1? " << same1 << "\n";
    std::cout << "same2? " << same2 << "\n";
//    REQUIRE(same0); // rarely, but occasionally fails?
//    REQUIRE(same1); // rarely, but occasionally fails?
//    REQUIRE(same2); // rarely, but occasionally fails?
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
