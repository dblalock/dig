

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

// pairings taken from the bottom of numpy.i
%np_vector_typemaps(signed char       , NPY_BYTE     )
%np_vector_typemaps(unsigned char     , NPY_UBYTE    )
%np_vector_typemaps(short             , NPY_SHORT    )
%np_vector_typemaps(unsigned short    , NPY_USHORT   )
%np_vector_typemaps(int               , NPY_INT      )
%np_vector_typemaps(unsigned int      , NPY_UINT     )
%np_vector_typemaps(long              , NPY_LONG     )
%np_vector_typemaps(unsigned long     , NPY_ULONG    )
%np_vector_typemaps(long long         , NPY_LONGLONG )
%np_vector_typemaps(unsigned long long, NPY_ULONGLONG)
%np_vector_typemaps(float             , NPY_FLOAT    )
%np_vector_typemaps(double            , NPY_DOUBLE   )

// apparently these are also necessary...
%np_vector_typemaps(int16_t, NPY_INT)
%np_vector_typemaps(int32_t, NPY_INT)
%np_vector_typemaps(int64_t, NPY_LONGLONG)
%np_vector_typemaps(uint16_t, NPY_UINT)
%np_vector_typemaps(uint32_t, NPY_UINT)
%np_vector_typemaps(uint64_t, NPY_ULONGLONG)
%np_vector_typemaps(int, NPY_INT)
%np_vector_typemaps(long, NPY_LONG)
%np_vector_typemaps(float, NPY_FLOAT)
%np_vector_typemaps(double, NPY_DOUBLE)
// %np_vector_typemaps(SimpleStruct*, NPY_OBJECT) // breaks

%np_vector_typemaps(length_t, NPY_LONGLONG)

// ================================================================
// eigen typemaps
// ================================================================

//%eigen_typemaps(Eigen::MatrixXd)
//%eigen_typemaps(VectorXd)
// %eigen_typemaps(ArrayXd) // breaks

// ------------------------ matrices

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;

typedef Matrix<double, Dynamic, Dynamic> FMatrix;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> CMatrix;
%eigen_typemaps(FMatrix);
%eigen_typemaps(CMatrix);
%eigen_typemaps(MatrixXd);
%eigen_typemaps(VectorXd);
%eigen_typemaps(RowVectorXd);
typedef Array<double, Dynamic, Dynamic> FArray;
typedef Array<double, Dynamic, Dynamic, RowMajor> CArray;
%eigen_typemaps(FArray);
%eigen_typemaps(CArray);
%eigen_typemaps(ArrayXd);  // 1d array
%eigen_typemaps(ArrayXXd); // 2d array

//%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>)
// %eigen_typemaps(Matrix<double, Dynamic, Dynamic>)
//%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)
//%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, 1>)
//%eigen_typemaps(Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>)
//
//%eigen_typemaps(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>)
//%eigen_typemaps(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)
//%eigen_typemaps(Eigen::Matrix<float, Eigen::Dynamic, 1>)
//%eigen_typemaps(Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>)
//
//%eigen_typemaps(Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>)
//%eigen_typemaps(Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)
//%eigen_typemaps(Eigen::Matrix<short, Eigen::Dynamic, 1>)
//%eigen_typemaps(Eigen::Matrix<short, 1, Eigen::Dynamic, Eigen::RowMajor>)
//
//%eigen_typemaps(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>)
//%eigen_typemaps(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)
//%eigen_typemaps(Eigen::Matrix<int, Eigen::Dynamic, 1>)
//%eigen_typemaps(Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>)
//
//%eigen_typemaps(Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>)
//%eigen_typemaps(Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)
//%eigen_typemaps(Eigen::Matrix<long, Eigen::Dynamic, 1>)
//%eigen_typemaps(Eigen::Matrix<long, 1, Eigen::Dynamic, Eigen::RowMajor>)
//
//%eigen_typemaps(Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic>)
//%eigen_typemaps(Eigen::Matrix<long long, Eigen::Dynamic, Eigen::Dynamic, //Eigen::RowMajor>)
//%eigen_typemaps(Eigen::Matrix<long long, Eigen::Dynamic, 1>)
//%eigen_typemaps(Eigen::Matrix<long long, 1, Eigen::Dynamic, Eigen::RowMajor>)

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
%apply (double* IN_ARRAY1, int DIM1) {(const double* seq, int seqLen)};

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
