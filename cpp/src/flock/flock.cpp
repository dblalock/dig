

#include "flock.hpp"

#include <Dense>

#include "type_defs.h"
#include "eigen_utils.hpp"
#include "pimpl_impl.hpp"

using Eigen::MatrixXd;

// ================================================================ pimpl

template<class T>
class FlockLearner<T>::Impl {
private:
	MatrixXd _X;
	length_t _m_min;
	length_t _m_max;
public:
	Impl(const T* X, int n, int d, int m_min, int m_max):
		_X(eigenWrap2D_aligned(X, n, d)),
		_m_min(m_min),
		_m_max(m_max)
	{}
	
	Impl(const T* X, int n, int d, double m_min, double m_max):
		Impl(X, m_min * n, m_max * n)
	{}
};
// ================================================================ public class

//template<class T>
//FlockLearner<T>::~FlockLearner() = default;

//template<class T>
//FlockLearner<T>::FlockLearner(const T* X, int n, int d, int m_min, int m_max):{
//	
//}
