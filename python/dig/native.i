// %module digpp //doesn't create any file with this name; just fails to find stuff
%module dig
%{
#define SWIG_FILE_WITH_INIT
#include "../../cpp/src/include/dig.hpp"
#include "../../cpp/src/include/dist.hpp"
#include "../../cpp/src/include/neighbors.hpp"
%}

// include numpy swig stuff
%include "numpy.i"
%init %{
import_array();
%}

// ================================================================
// apply numpy typemaps to my own funcs based on their parameters
// ================================================================

// ================================
// in-place modification of arrays
// ================================

// ------------------------------- 1D arrays
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* inVec, int len)};

// ================================
// read-only input arrays
// ================================

// ------------------------------- 1D arrays
%apply (double* IN_ARRAY1, int DIM1) {(const double* ar, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* buff, int n)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* buff, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* buff, int buffLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* x, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* x, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* x, int xLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* q, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* q, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* q, int qLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* query, int qLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v, int len)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v, int inLen)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v1, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v1, int len1)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v2, int m)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v2, int n)};
%apply (double* IN_ARRAY1, int DIM1) {(const double* v2, int len2)};

// ------------------------------- 2D arrays
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* A, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* X, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* A, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int m, int n)};

// ================================
// returned arrays
// ================================

// ------------------------------- 1D arrays
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* outVec, int len)};
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* outVec, int outLen)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* outVec, int len)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* outVec, int outLen)};


// ================================================================
// actually have swig parse + wrap the files
// ================================================================
%include "../../cpp/src/include/dig.hpp"
%include "../../cpp/src/include/dist.hpp"
%include "../../cpp/src/include/neighbors.hpp"
