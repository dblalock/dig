//
//  test_array_utils.cpp
//  Dig
//
//  Created by DB on 10/24/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include <algorithm>
#include <math.h>

#include "catch.hpp"
#include "testing_utils.hpp"
#include "array_utils.hpp"

#include "debug_utils.hpp"

using std::vector;
typedef double data_t;

using namespace ar;

TEST_CASE("stats", "array_utils") {
	int len = 4;
	vector<int> x = {1,4,2,3};
	vector<int> y = {3,2,4,1};
	REQUIRE(max(x) == 4);
	REQUIRE(min(x) == 1);
	REQUIRE(sum(x) == 10);
	REQUIRE(sumsquares(x) == (1+16+4+9));
	REQUIRE(variance(x) == 5.0 / 4);
	REQUIRE(stdev(x) == sqrt(5.0 / 4));
	REQUIRE(dot(x, y) == (3+8+8+3));
}

TEST_CASE("elementwise","utils") {
	vector<int>		x = {1,4,2,3};
	vector<int>		y = {3,2,4,1};
	vector<double>	z = {3,2,4,1.5};

//	const int len = 4;
	int len = 4;
	int		a[] = {1,4,2,3};
	int		b[] = {3,2,4,1};
	double	c[] = {3,2,4,1.5};

	SECTION("add") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {4,6,6,4};
				auto v = add(x, y);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {4,6,6,4.5};
				auto v = add(x, z);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {4,6,6,4};
				auto v = add(a, b, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {4,6,6,4.5};
				auto v = add(a, c, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("sub") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {-2,2,-2,2};
				auto v = sub(x, y);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {-2,2,-2,1.5};
				auto v = sub(x, z);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {-2,2,-2,2};
				auto v = sub(a, b, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {-2,2,-2,1.5};
				auto v = sub(a, c, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("mul") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {3,8,8,3};
				auto v = mul(x, y);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {3,8,8,4.5};
				auto v = mul(x, z);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {3,8,8,3};
				auto v = mul(a, b, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {3,8,8,4.5};
				auto v = mul(a, c, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("div") {
		SECTION("vects") {
			SECTION("ints") {
				// vector<double> ans {1.0/3, 2.0, 1.0/2, 3.0};
				vector<int> ans {0, 2, 0, 3};
				auto v = div(x, y);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {1.0/3, 2.0, 1.0/2, 2.0};
				auto v = div(x, z);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				// double ans[] = {1.0/3, 2.0, 1.0/2, 3.0};
				int ans[] = {0, 2, 0, 3};
				auto v = div(a, b, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {1.0/3, 2.0, 1.0/2, 2.0};
				auto v = div(a, c, len);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
}

TEST_CASE("scalar","utils") {
	vector<int>	x	= {1,4,2,3};
	int	a[]			= {1,4,2,3};
	int len = 4;
	int k = 2;
	float q = 1.5;

	SECTION("add") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {3,6,4,5};
				auto v = add(x, k);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {2.5,5.5,3.5,4.5};
				auto v = add(x, q);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {3,6,4,5};
				auto v = add(a, len, k);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {2.5,5.5,3.5,4.5};
				auto v = add(a, len, q);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("sub") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {-1,2,0,1};
				auto v = sub(x, k);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {-.5, 2.5, .5, 1.5};
				auto v = sub(x, q);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {-1,2,0,1};
				auto v = sub(a, len, k);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {-.5, 2.5, .5, 1.5};
				auto v = sub(a, len, q);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("mul") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {2,8,4,6};
				auto v = mul(x, k);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {1.5, 6, 3, 4.5};
				auto v = mul(x, q);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				int ans[] = {2,8,4,6};
				auto v = mul(a, len, k);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {1.5, 6, 3, 4.5};
				auto v = mul(a, len, q);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
	SECTION("div") {
		SECTION("vects") {
			SECTION("ints") {
				vector<int> ans {0, 2, 1, 1};
				auto v = div(x, k);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				vector<double> ans {x[0]/q, x[1]/q, x[2]/q, x[3]/q};
				auto v = div(x, q);
				short int equal = all_eq(ans, v);
				INFO("output = " + to_string(v))
				REQUIRE(equal);
			}
		}
		SECTION("arrays") {
			SECTION("ints") {
				double ans[] = {0, 2, 1, 1};
				auto v = div(a, len, k);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
			SECTION("int + float") {
				double ans[] = {x[0]/q, x[1]/q, x[2]/q, x[3]/q};
				auto v = div(a, len, q);
				short int equal = all_eq(ans, v.get(), len);
				INFO("output = " + to_string(v.get(), len))
				REQUIRE(equal);
			}
		}
	}
}

TEST_CASE("resample","utils") {

	SECTION("sampling rate correct") {
		unsigned int srcLen = 5;
		unsigned int destLen = 5;
		data_t src[] = {1,2,3,4,5};
		data_t destTruth[] = {1,2,3,4,5};
		data_t dest[destLen];

		resample(src,dest,srcLen,destLen);

		REQUIRE(all_eq(dest, destTruth, destLen));
	}

	SECTION("resample upsample by integer","utils") {
		unsigned int srcLen = 5;
		unsigned int destLen = 15;
		data_t src[] = {1,2,3,4,5};
		data_t destTruth[] = {1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5};
		data_t dest[destLen];

		resample(src,dest,srcLen,destLen);

		REQUIRE(all_eq(dest,destTruth,destLen) );
	}

	SECTION("resample upsample by fraction") {
		unsigned int srcLen = 5;
		unsigned int destLen = 7;
		data_t src[] = {1,2,3,4,5};
		data_t destTruth[] = {1,1,2,3,3,4,5,};
		data_t dest[destLen];

		resample(src,dest,srcLen,destLen);

		REQUIRE(all_eq(dest,destTruth,destLen) );
	}

	SECTION("resample downsample by integer") {
		unsigned int srcLen = 6;
		unsigned int destLen = 3;
		data_t src[] = {1,2,3,4,5,6};
		data_t destTruth[] = {1,3,5};
		data_t dest[destLen];

		resample(src,dest,srcLen,destLen);

		REQUIRE(all_eq(dest,destTruth,destLen) );
	}

	SECTION("resample downsample by fraction") {
		unsigned int srcLen = 7;
		unsigned int destLen = 3;
		data_t src[] = {1,2,3,4,5,6,7};
		data_t destTruth[] = {1,3,5};
		data_t dest[destLen];

		resample(src,dest,srcLen,destLen);

		REQUIRE(all_eq(dest,destTruth,destLen) );
	}
}

TEST_CASE("reverse", "utils") {
	SECTION("reverse even length") {
		unsigned int len = 6;
		data_t src[] = {1,2,3,4,5,6};
		data_t dest[len];
		data_t destTruth[] = {6,5,4,3,2,1};

		reverse(src, dest, len);

		REQUIRE(all_eq(dest,destTruth,len) );
	}

	SECTION("reverse odd length") {
		unsigned int len = 7;
		data_t src[] = {1,2,3,4,5,6,7};
		data_t dest[len];
		data_t destTruth[] = {7,6,5,4,3,2,1};

		reverse(src, dest, len);

		REQUIRE(all_eq(dest,destTruth,len) );
	}
}

TEST_CASE("reshape", "utils") {

//	SECTION("dimensions not factor of length returns null") {
//		unsigned int len = 7;
//		data_t x[] = {1,2,3,4,5,6,7};
//		unsigned int newNumDims = 2;
//
//		data_t** newArrays = split(x, len, newNumDims);
//
//		REQUIRE(newArrays == nullptr);
//	}

	SECTION("dimensions = 1 returns pointer to copy of original array") {
		unsigned int len = 7;
		data_t x[] = {1,2,3,4,5,6,7};
		unsigned int newNumDims = 1;

		data_t** newArrays = split(x, len, newNumDims);

		REQUIRE(all_eq(x,newArrays[0],len) );
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
		data_t** newArrays = split(x, len, newNumDims);
		data_t* col1 = newArrays[0];
		data_t* col2 = newArrays[1];
		short int equal = all_eq(col1, col1_truth, newLen);
		equal = equal && all_eq(col2, col2_truth, newLen);

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
		data_t** newArrays = split(x, len, newNumDims);
		data_t* col1 = newArrays[0];
		data_t* col2 = newArrays[1];
		data_t* col3 = newArrays[2];
		short int equal = all_eq(col1, col1_truth, newLen);
		equal = equal && all_eq(col2, col2_truth, newLen);
		equal = equal && all_eq(col3, col3_truth, newLen);

		REQUIRE(equal);
	}
}

TEST_CASE("map", "utils") {

	int len = 4;
	vector<int> x = {1,3,5,7};

	SECTION("plus1") {
		vector<int> ans = {2,4,6,8};
		auto v = map( [](int z) {return ++z;}, x);
		short int equal = all_eq(v, ans);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}

	SECTION("times2") {
		vector<int> ans = {2,6,10,14};
		auto v = map( [](int z) {return z*2;}, x);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("setToConst") {
		vector<int> ans = {-3,-3,-3,-3};
		auto v = map( [](int z) {return -3;}, x);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("filter", "utils") {
	vector<int> x = {1,5,7,3};

	SECTION("lessThan4") {
		vector<int> ans = {1,3};
		auto v = filter( [](int z) {return z < 4;}, x);
		short int equal = all_eq(v, ans);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet") {
		vector<int> ans;
		auto v = filter( [](int z) {return false;}, x);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything") {
		vector<int> ans(x);
		auto v = filter( [](int z) {return true;}, x);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("where", "utils") {
	vector<int> x = {1,5,7,3};

	SECTION("lessThan4") {
		vector<int> ans = {0,3};
		auto v = where( [](int z) {return z < 4;}, x);
		short int equal = all_eq(v, ans);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet") {
		vector<int> ans;
		auto v = where( [](int z) {return false;}, x);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything") {
		vector<int> ans = {0,1,2,3};
		auto v = where( [](int z) {return true;}, x);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("at_idxs", "utils") {
	vector<int> x = {1,5,7,3};

	SECTION("lessThan4") {
		vector<int> idxs = {0,2};
		vector<int> ans = {x[0],x[2]};
		auto v = at_idxs(x, idxs);
		short int equal = all_eq(v, ans);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet") {
		vector<int> idxs;
		vector<int> ans;
		auto v = at_idxs(x, idxs);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything") {
		vector<int> idxs = {0,1,2,3};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("EmptySet, no bounds check") {
		vector<int> idxs;
		vector<int> ans;
		auto v = at_idxs(x, idxs, false);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("Everything, no bounds check") {
		vector<int> idxs = {0,1,2,3};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs, false);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("BoundsCheckCatchesNegative") {
		vector<int> idxs = {-1,0,1,2,3};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs, true);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
	SECTION("BoundsCheckCatchesTooLarge") {
		vector<int> idxs = {0,1,2,3, 4,971};
		vector<int> ans(x);
		auto v = at_idxs(x, idxs, true);
		short int equal = all_eq(ans, v);
		INFO("output = " + to_string(v))
		REQUIRE(equal);
	}
}

TEST_CASE("unique", "utils") {
	vector<int> x = {10,20,20,20,30,30,20,20,10};

	auto uniq = unique(x);
	REQUIRE(uniq.size() == 3);
	REQUIRE(uniq[0] == 10);
	REQUIRE(uniq[1] == 20);
	REQUIRE(uniq[2] == 30);
}

TEST_CASE("random", "utils") {

	SECTION("intsWithReplace") {
		int howMany = 7;
		bool replace = true;
		auto ints = rand_ints(0, 5, howMany, replace); // must have duplicate
		std::sort(std::begin(ints), std::end(ints));
		bool worked = false;
		for (int i = 0; i < ints.size() - 1; i++) {
			worked = worked || ints[i] == ints[i+1];
		}
		INFO("Sampled ints with replacement = " + to_string(ints));
		REQUIRE(worked);
	}

	SECTION("intsNoReplace1") {
		int howMany = 100;
		bool replace = false;
		int maxVal = howMany;
		auto ints = rand_ints(1, maxVal, howMany, replace); // no duplicates

		INFO("Sampled ints without replacement = " + to_string(ints));
		REQUIRE(unique(ints).size() == howMany);
	}

	SECTION("intsNoReplace2") {
		int howMany = 100;
		bool replace = false;
		int maxVal = 10*1000;
		auto ints = rand_ints(0, maxVal, howMany, replace); // no duplicates
		INFO("Sampled ints without replacement = " + to_string(ints));
		REQUIRE(unique(ints).size() == howMany);
	}

	SECTION("intsNoReplaceReturn0Samples") {
		int howMany = 0;
		bool replace = false;
		int maxVal = howMany;
		auto ints = rand_ints(-5, maxVal, howMany, replace); // no duplicates

		INFO("Sampled ints without replacement = " + to_string(ints));
		REQUIRE(ints.size() == howMany);
	}

	SECTION("intsNoReplaceReturn1Sample") {
		int howMany = 1;
		bool replace = false;
		int maxVal = howMany;
		auto ints = rand_ints(0, maxVal, howMany, replace); // no duplicates

		INFO("Sampled ints without replacement = " + to_string(ints));
		REQUIRE(ints.size() == howMany);
	}

}

TEST_CASE("range", "array_utils") {
	SECTION("ints") {
		SECTION("length=1") {
			vector<int> ans {-7};
			auto out = range(-7, -6);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));
		}
		SECTION("step=1") {
			vector<int> ans {0, 1, 2, 3};

			auto out = range(0.0, 4);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));

			auto out2 = range(0, 4, 1);
			INFO(to_string(out2));
			REQUIRE(all_eq(ans, out2));
		}
		SECTION("step=3") {
			vector<int> ans {-2, 1, 4, 7};

			auto out = range(-2, 10, 3);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));
		}
		SECTION("step=-2") {
			vector<int> ans {4, 2, 0, -2};

			auto out = range(4, -2.5, -2);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));
		}
	}
	SECTION("floats") {
		SECTION("length=1") {
			vector<float> ans {-7};
			auto out = range(-7, -6);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));
		}
		SECTION("step=1") {
			vector<float> ans {0, 1, 2, 3};

			auto out = range(0, 4);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));

			auto out2 = range(0, 4, 1);
			INFO(to_string(out2));
			REQUIRE(all_eq(ans, out2));
		}
		SECTION("step=3") {
			vector<float> ans {-2, 1, 4, 7};

			auto out = range(-2, 10, 3);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));
		}
		SECTION("step=-2") {
			vector<float> ans {4, 2, 0, -2};

			auto out = range(4, -2.5, -2);
			INFO(to_string(out));
			REQUIRE(all_eq(ans, out));
		}
	}
}

TEST_CASE("exprange", "array_utils") {
	SECTION("length=1") {
		vector<int> ans {-7};
		auto out = exprange(-7, -8);
		INFO(to_string(out));
		REQUIRE(all_eq(ans, out));
	}
	SECTION("step=2") {
		vector<int> ans {3, 6, 12, 24};

		auto out = exprange(3, 25, 2);
		INFO(to_string(out));
		REQUIRE(all_eq(ans, out));

		auto out2 = exprange(3.0, 25, 2);
		INFO(to_string(out2));
		REQUIRE(all_eq(ans, out2));
	}
	SECTION("step=-.5") {
		vector<int> ans {24, -12, 6, -3};

		auto out = exprange(24, 2, -.5);
		INFO(to_string(out));
		REQUIRE(all_eq(ans, out));

		auto out2 = exprange(24, 2.0, -.5);
		INFO(to_string(out2));
		REQUIRE(all_eq(ans, out2));
	}
	SECTION("step=-.5, fraction in ans") {
		vector<float> ans {24, -12, 6, -3, 1.5, -.75};

		auto out = exprange(24, .5, -.5);
		INFO(to_string(out));
		REQUIRE(all_eq(ans, out));

		auto out2 = exprange(24, .5, -.5);
		INFO(to_string(out2));
		REQUIRE(all_eq(ans, out2));
	}
}

TEST_CASE("pad", "array_utils") {
	vector<float> x = {1,3,2};
	vector<float> ans = {0,0,1,3,2,0};

	SECTION("zero padding") {
		auto out = pad(x, 2, 1); // 2 on left, 1 on right
		INFO(to_string(out));
		REQUIRE(all_eq(out, ans));

		vector<float> ans2 = {1,3,2};
		auto out2 = pad(x, -1, 0); // 0 on left, 0 on right
		INFO(to_string(out2));
		REQUIRE(all_eq(out2, ans2));
	}

	// pad with a value, not just 0
	SECTION("constant padding") {
		vector<float> ans3 = {1,3,2, 99, 99, 99};
		auto out3 = pad(x, 0, 3, PAD_CONSTANT, 99); // 0 on left, 3 on right
		INFO(to_string(out3));
		REQUIRE(all_eq(out3, ans3));
	}
}

TEST_CASE("normalize_mean", "array_utils") {
	SECTION("vector") {
		vector<float> x {0, 1, 2, 3};
		vector<float> ans {-1.5, -.5, .5, 1.5};
		auto out = normalize_mean(x);
		REQUIRE(all_eq(out, ans));
	}
	SECTION("array") {
		int x[] {0, 1, 2, 3};
		// double ans[] {-1.5, -.5, .5, 1.5};
		int ans[] {-1, 0, 0, 1}; // int truncates towards 0
		auto out = normalize_mean(x, 4);
		INFO(to_string(out.get(), 4));
		REQUIRE(all_eq(out.get(), ans, 4));
	}
	SECTION("array_inplace") {
		int x[] {0, 1, 2, 3};
		// double ans[] {-1.5, -.5, .5, 1.5};
		int ans[] {-1, 0, 0, 1}; // int truncates towards 0
		normalize_mean_inplace(x, 4);
		INFO(to_string(x, 4));
		REQUIRE(all_eq(x, ans, 4));
	}

}

