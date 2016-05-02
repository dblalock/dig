

#include "tree2.hpp"

// ================================================================
// Typedefs and usings
// ================================================================

typedef Eigen::Matrix<hash_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> HashMat;
typedef Eigen::Matrix<hash_t, Eigen::Dynamic, 1> HashVect;

// using std::unordered_map;
using std::map;
using RedSVD::RedSVD;

typedef uint8_t bin_t;
typedef uint8_t depth_t;

// ================================================================
// Constants
// ================================================================

const hash_t MAX_HASH_VALUE = 7; // int8_t
const hash_t MIN_HASH_VALUE = 0;
const hash_t HASH_VALUE_OFFSET = 4;

// const hash_t MAX_HASH_VALUE = 32767; // int16_t
// const hash_t MIN_HASH_VALUE = -32768;
// const double TARGET_HASH_SPREAD_STDS = 3.0; // +/- 3 std deviations
const double TARGET_HASH_SPREAD_STDS = 2.0; // +/- this many std deviations
// const double TARGET_HASH_SPREAD_STDS = 1.0; // +/- this many std deviations

const uint16_t MAX_POINTS_PER_LEAF = 512;

const

// ================================================================
// Constants
// ================================================================
