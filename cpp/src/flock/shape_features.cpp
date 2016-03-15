
//#include "shape_features.hpp"



// using Eigen::Matrix;
// using Eigen::MatrixXd;


//template<class Derived1, class Derived2, class Derived3>
//void crossCorrelate(const EigenBase<Derived1>& shorter,
//					const EigenBase<Derived2>& longer,
//					EigenBase<Derived3>& out) {
//	auto n = longer.size();
//	auto m = shorter.size();
//	auto l = n - m + 1;
//	assert(n >= m);
//	for (size_t i = 0; i < l; i++) {
//		auto subseq = longer.segment(i, m);
//		out.noalias()(i) = subseq.dot(shorter);
//	}
//}
//
//template<class Derived1, class Derived2>
//auto crossCorrelate(const EigenBase<Derived1>& shorter,
//					const EigenBase<Derived2>& longer) ->
//	Matrix<decltype(shorter(0) * longer(0)), Dynamic, 1>
//{
//	auto l = longer.size(); - shorter.size() + 1;
////	VectorXd out(l);
//	Matrix<decltype(shorter(0) * longer(0)), Dynamic, 1> out(l);
//	crossCorrelate(shorter, longer, out);
//	return out;
//}



