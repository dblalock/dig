//
//  slice.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __SLICE_HPP
#define __SLICE_HPP

#include <string>
#include <iostream>

#include "array_utils.hpp"

using std::string;

using ar::min;
using ar::max;

#include "debug_utils.hpp"

namespace ar {

template<class data_t>
void plot_array(data_t* data, length_t length, int nlevels=8, double yBin=0,
				string title=string(), char marker='x')
{
	// compute vertical bin size and number of bins
	auto minVal = min(data, length);
	auto maxVal = max(data, length);
	auto range = maxVal - minVal;

	if (yBin <= 0) {
		yBin = (maxVal - minVal) / nlevels;
	}
	nlevels = static_cast<int>(range / yBin) + 2;
	
	// create string in which to write plot
	int nRows = nlevels;
	int nCols = static_cast<int>(length);
	int stringLen = nRows * nCols;
	string s(stringLen, ' '); // whitespace

	// compute offset so that y axis labels are multiples of yBin
	double offset = yBin * static_cast<int>(minVal / yBin);
	
	// populate plot body
	for (int i = 0; i < length; i++) {
		int level = static_cast<int>(floor((data[i] - offset) / yBin));
		level = max(level, 0);
		int idx = level * nCols + i;
		s[idx] = marker;
	}
	
	// print title
	if (title.size()) {
		int totalLen = 5 + nCols;
		string pad((totalLen - title.size()) / 2, ' ');
		std::cout << pad << title << "\n";
	}
	
	// print the plot and y axis labels
	char axLabel[32];
	// iterate thru rows of mat, from high to low values
	for (int i = nlevels - 1; i >= 0; i--) {
		double lb = i * yBin - offset;
		snprintf(axLabel, 31, "%5.2f", lb);
		string lbl(axLabel);
		std::cout << lbl << " " << s.substr(i * nCols, nCols) << "\n";
	}
}

} // namespace ar

#endif
