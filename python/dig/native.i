// %module digpp //doesn't create any file with this name; just fails to find stuff
%module dig
%{
#define SWIG_FILE_WITH_INIT
#include <vector>
#include <sys/types.h>
#include "../../cpp/src/include/dig.hpp"
#include "../../cpp/src/include/dist.hpp"
#include "../../cpp/src/include/neighbors.hpp"
// #include "../../cpp/src/neighbors/tree.hpp"
%}

// include numpy swig stuff
%include "numpy.i"
%init %{
import_array();
%}
%include <eigen.i>
%include <np_vector.i>

// ================================================================
// stl vector typemaps
// ================================================================

%np_vector_typemaps(int16_t, NPY_INT)
%np_vector_typemaps(int32_t, NPY_INT)
%np_vector_typemaps(int64_t, NPY_LONG)
%np_vector_typemaps(uint16_t, NPY_UINT)
%np_vector_typemaps(uint32_t, NPY_UINT)
%np_vector_typemaps(uint64_t, NPY_ULONG)
%np_vector_typemaps(int, NPY_INT)
%np_vector_typemaps(long, NPY_LONG)
%np_vector_typemaps(float, NPY_FLOAT)
%np_vector_typemaps(double, NPY_DOUBLE)
// %np_vector_typemaps(SimpleStruct*, NPY_OBJECT) // breaks

%np_vector_typemaps(length_t, NPY_INT)

// ================================================================
// eigen typemaps
// ================================================================

%eigen_typemaps(MatrixXd)
%eigen_typemaps(VectorXd)
// %eigen_typemaps(ArrayXd) // breaks
%eigen_typemaps(MatrixXf)
%eigen_typemaps(VectorXf)
%eigen_typemaps(MatrixXi)
%eigen_typemaps(VectorXi)

// ================================================================
// raw c array typemaps
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
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* X, int d, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(const double* X, int n, int d)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* A, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int m, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int d, int n)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* X, int n, int d)};

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
// %include "../../cpp/src/neighbors/tree.hpp"
