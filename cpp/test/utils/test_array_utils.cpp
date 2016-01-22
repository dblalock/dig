//
//  test_array_utils.cpp
//  TimeKit
//
//  Created by DB on 10/24/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include <math.h>

#include "catch.hpp"
#include "testing_utils.hpp"
#include "array_utils.hpp"

#include "debug_utils.hpp"

using std::vector;
typedef double data_t;

TEST_CASE("stats", "array_utils") {
	int len = 4;
	vector<int> x = {1,4,2,3};
	vector<int> y = {3,2,4,1};
	REQUIRE(array_max(x) == 4);
	REQUIRE(array_min(x) == 1);
	REQUIRE(array_sum(x) == 10);
	REQUIRE(array_sumsquares(x) == (1+16+4+9));
	REQUIRE(array_variance(x) == 5.0 / 4);
	REQUIRE(array_std(x) == sqrt(5.0 / 4));
	REQUIRE(array_dot(x, y) == (3+8+8+3));
}

TEST_CASE("elementwise","array_utils") {
	vector<int>		x = {1,4,2,3};
	vector<int>		y = {3,2,4,1};
	vector<double>	z = {3,2,4,1.5};
	
	const int len = 4;
	int		a[] = {1,4,2,3};
	int		b[] = {3,2,4,1};
	double	c[] = {3,2,4,1.5};

	SECTION("add") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {4,6,6,4};
				auto v = array_add(x, y);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {4,6,6,4.5};
				auto v = array_add(x, z);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {4,6,6,4};
				auto v = array_add(a, b, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {4,6,6,4.5};
				auto v = array_add(a, c, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("sub") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {-2,2,-2,2};
				auto v = array_sub(x, y);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {-2,2,-2,1.5};
				auto v = array_sub(x, z);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {-2,2,-2,2};
				auto v = array_sub(a, b, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {-2,2,-2,1.5};
				auto v = array_sub(a, c, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("mul") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {3,8,8,3};
				auto v = array_mul(x, y);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {3,8,8,4.5};
				auto v = array_mul(x, z);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {3,8,8,3};
				auto v = array_mul(a, b, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {3,8,8,4.5};
				auto v = array_mul(a, c, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("div") {
		SECTION("vects") {
			SECTION("ints") {
				vector<double> ans {1.0/3, 2.0, 1.0/2, 3.0};
				auto v = array_div(x, y);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {1.0/3, 2.0, 1.0/2, 2.0};
				auto v = array_div(x, z);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				double ans[] = {1.0/3, 2.0, 1.0/2, 3.0};
				auto v = array_div(a, b, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {1.0/3, 2.0, 1.0/2, 2.0};
				auto v = array_div(a, c, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
}

TEST_CASE("scalar","array_utils") {
	vector<int>	x	= {1,4,2,3};
	int	a[]			= {1,4,2,3};
	int len = 4;
	int k = 2;
	float q = 1.5;
	
	SECTION("add") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {3,6,4,5};
				auto v = array_adds(x, k);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {2.5,5.5,3.5,4.5};
				auto v = array_adds(x, q);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {3,6,4,5};
				auto v = array_adds(a, k, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {2.5,5.5,3.5,4.5};
				auto v = array_adds(a, q, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("sub") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {-1,2,0,1};
				auto v = array_subs(x, k);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {-.5, 2.5, .5, 1.5};
				auto v = array_subs(x, q);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {-1,2,0,1};
				auto v = array_subs(a, k, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {-.5, 2.5, .5, 1.5};
				auto v = array_subs(a, q, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("mul") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {2,8,4,6};
				auto v = array_muls(x, k);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {1.5, 6, 3, 4.5};
				auto v = array_muls(x, q);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {2,8,4,6};
				auto v = array_muls(a, k, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {1.5, 6, 3, 4.5};
				auto v = array_muls(a, q, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("div") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {0, 2, 1, 1};
				auto v = array_divs(x, k);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {x[0]/q, x[1]/q, x[2]/q, x[3]/q};
				auto v = array_divs(x, q);
				short int equal = array_equal(ans, v);
				INFO("output = " + array_to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				double ans[] = {0, 2, 1, 1};
				auto v = array_divs(a, k, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {x[0]/q, x[1]/q, x[2]/q, x[3]/q};
				auto v = array_divs(a, q, len);
				short int equal = array_equal(ans, v.get(), len);
				INFO("output = " + array_to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
}

TEST_CASE("resample","array_utils") {
	
	SECTION("sampling rate correct") {
		unsigned int srcLen = 5;
		unsigned int destLen = 5;
		data_t src[] = {1,2,3,4,5};
		data_t destTruth[] = {1,2,3,4,5};
		data_t dest[destLen];
		
		array_resample(src,dest,srcLen,destLen);
		
		REQUIRE(array_equal(dest, destTruth, destLen));
	}

	SECTION("resample upsample by integer","array_utils") {
		unsigned int srcLen = 5;
		unsigned int destLen = 15;
		data_t src[] = {1,2,3,4,5};
		data_t destTruth[] = {1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5};
		data_t dest[destLen];
		
		array_resample(src,dest,srcLen,destLen);
		
		REQUIRE(array_equal(dest,destTruth,destLen) );
	}

	SECTION("resample upsample by fraction") {
		unsigned int srcLen = 5;
		unsigned int destLen = 7;
		data_t src[] = {1,2,3,4,5};
		data_t destTruth[] = {1,1,2,3,3,4,5,};
		data_t dest[destLen];
		
		array_resample(src,dest,srcLen,destLen);
		
		REQUIRE( array_equal(dest,destTruth,destLen) );
	}
	
	SECTION("resample downsample by integer") {
		unsigned int srcLen = 6;
		unsigned int destLen = 3;
		data_t src[] = {1,2,3,4,5,6};
		data_t destTruth[] = {1,3,5};
		data_t dest[destLen];
		
		array_resample(src,dest,srcLen,destLen);
		
		REQUIRE( array_equal(dest,destTruth,destLen) );
	}
	
	SECTION("resample downsample by fraction") {
		unsigned int srcLen = 7;
		unsigned int destLen = 3;
		data_t src[] = {1,2,3,4,5,6,7};
		data_t destTruth[] = {1,3,5};
		data_t dest[destLen];
		
		array_resample(src,dest,srcLen,destLen);
		
		REQUIRE( array_equal(dest,destTruth,destLen) );
	}
}

TEST_CASE("reverse", "array_utils") {
	SECTION("reverse even length") {
		unsigned int len = 6;
		data_t src[] = {1,2,3,4,5,6};
		data_t dest[len];
		data_t destTruth[] = {6,5,4,3,2,1};
		
		array_reverse(src, dest, len);
		
		REQUIRE( array_equal(dest,destTruth,len) );
	}
	
	SECTION("reverse odd length") {
		unsigned int len = 7;
		data_t src[] = {1,2,3,4,5,6,7};
		data_t dest[len];
		data_t destTruth[] = {7,6,5,4,3,2,1};
		
		array_reverse(src, dest, len);
		
		REQUIRE( array_equal(dest,destTruth,len) );
	}
}

TEST_CASE("reshape", "array_utils") {
	
	SECTION("dimensions not factor of length returns null") {
		unsigned int len = 7;
		data_t x[] = {1,2,3,4,5,6,7};
		unsigned int newNumDims = 2;
		
		data_t** newArrays = array_split(x, len, newNumDims);
		
		REQUIRE(newArrays == nullptr);
	}
	
	SECTION("dimensions = 1 returns pointer to copy of original array") {
		unsigned int len = 7;
		data_t x[] = {1,2,3,4,5,6,7};
		unsigned int newNumDims = 1;
		
		data_t** newArrays = array_split(x, len, newNumDims);
		
		REQUIRE( array_equal(x,newArrays[0],len) );
	}
	
	SECTION("split into 2 dims") {
		unsigned int len = 8;
		data_t x[] = {1,2,3,4,5,6,7,8};
		unsigned int newNumDims = 2;
		
		//answers
		unsigned int newLen = 4;
		data_t col1_truth[] = {1,3,5,7};
		data_t col2_truth[] = {2,4,6,8};
		
		//check if it worked
		data_t** newArrays = array_split(x, len, newNumDims);
		data_t* col1 = newArrays[0];
		data_t* col2 = newArrays[1];
		short int equal = array_equal(col1, col1_truth, newLen);
		equal = equal && array_equal(col2, col2_truth, newLen);
		
		REQUIRE(equal);
	}
	
	SECTION("split into 3 dims") {
		unsigned int len = 3;
		data_t x[] = {1,2,3};
		unsigned int newNumDims = 3;
		
		//answers
		unsigned int newLen = 1;
		data_t col1_truth[] = {1};
		data_t col2_truth[] = {2};
		data_t col3_truth[] = {3};
		
		//check if it worked
		data_t** newArrays = array_split(x, len, newNumDims);
		data_t* col1 = newArrays[0];
		data_t* col2 = newArrays[1];
		data_t* col3 = newArrays[2];
		short int equal = array_equal(col1, col1_truth, newLen);
		equal = equal &&  array_equal(col2, col2_truth, newLen);
		equal = equal &&  array_equal(col3, col3_truth, newLen);
		
		REQUIRE(equal);
	}
}

TEST_CASE("map", "array_utils") {
	
	int len = 4;
	vector<int> x = {1,3,5,7};
	
	SECTION("plus1") {
		vector<int> ans = {2,4,6,8};
		auto v = map( [](int z) {return ++z;}, x);
		short int equal = array_equal(v, ans);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	
	SECTION("times2") {
		vector<int> ans = {2,6,10,14};
		auto v = map( [](int z) {return z*2;}, x);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("setToConst") {
		vector<int> ans = {-3,-3,-3,-3};
		auto v = map( [](int z) {return -3;}, x);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("filter", "array_utils") {
	vector<int> x = {1,5,7,3};
	
	SECTION("lessThan4") {
		vector<int> ans = {1,3};
		auto v = filter( [](int z) {return z < 4;}, x);
		short int equal = array_equal(v, ans);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet") {
		vector<int> ans;
		auto v = filter( [](int z) {return false;}, x);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything") {
		vector<int> ans(x);
		auto v = filter( [](int z) {return true;}, x);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("where", "array_utils") {
	vector<int> x = {1,5,7,3};
	
	SECTION("lessThan4") {
		vector<int> ans = {0,3};
		auto v = where( [](int z) {return z < 4;}, x);
		short int equal = array_equal(v, ans);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet") {
		vector<int> ans;
		auto v = where( [](int z) {return false;}, x);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything") {
		vector<int> ans = {0,1,2,3};
		auto v = where( [](int z) {return true;}, x);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("at_idxs", "array_utils") {
	vector<int> x = {1,5,7,3};
	
	SECTION("lessThan4") {
		vector<int> idxs = {0,2};
		vector<int> ans = {x[0],x[2]};
		auto v = at_idxs(x, idxs);
		short int equal = array_equal(v, ans);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet") {
		vector<int> idxs;
		vector<int> ans;
		auto v = at_idxs(x, idxs);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything") {
		vector<int> idxs = {0,1,2,3};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet, no bounds check") {
		vector<int> idxs;
		vector<int> ans;
		auto v = at_idxs(x, idxs, false);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything, no bounds check") {
		vector<int> idxs = {0,1,2,3};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs, false);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("BoundsCheckCatchesNegative") {
		vector<int> idxs = {-1,0,1,2,3};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
	SECTION("BoundsCheckCatchesTooLarge") {
		vector<int> idxs = {0,1,2,3, 4,971};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs);
		short int equal = array_equal(ans, v);
		INFO("output = " + array_to_string(v))
		REQUIRE(equal);
	}
}
