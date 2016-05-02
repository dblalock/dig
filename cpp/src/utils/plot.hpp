//
//  slice.hpp
//
//  Created By Davis Blalock on 3/15/16.
//  Copyright (c) 2016 Davis Blalock. All rights reserved.
//

#ifndef __PLOT_HPP
#define __PLOT_HPP

#include <string>
#include <iostream>

#include "array_utils.hpp"

using std::string;

using ar::min;
using ar::max;
using ar::unique;

namespace ar {

static inline void _print_title(string title, length_t ncols) {
	if (title.size()) {
		int totalLen = static_cast<int>(5 + ncols);
		string pad((totalLen - title.size()) / 2, ' ');
		std::cout << pad << title << "\n";
	}
}

template<class data_t>
static void plot_array(data_t* data, length_t length, int nlevels=8,
	double yBin=0, string title=string(), char marker='x') {
	// compute vertical bin size and number of bins
	auto minVal = min(data, length);
	auto maxVal = max(data, length);
	auto range = maxVal - minVal;

	if (yBin <= 0) {
		yBin = (maxVal - minVal) / nlevels;
	}
	nlevels = static_cast<int>(range / yBin) + 2;

	// create string in which to write plot
	int nrows = nlevels;
	int ncols = static_cast<int>(length);
	int stringLen = nrows * ncols;
	string s(stringLen, ' '); // whitespace

	// compute offset so that y axis labels are multiples of yBin
	double offset = yBin * static_cast<int>(floor(minVal / yBin));

	// populate plot body
	for (int i = 0; i < length; i++) {
		int level = static_cast<int>(floor((data[i] - offset) / yBin));
		level = max(level, 0);
		int idx = level * ncols + i;
		s[idx] = marker;
	}

	_print_title(title, ncols);

	// print the plot and y axis labels
	char axLabel[32];
	// iterate thru rows of mat, from high to low values
	for (int i = nlevels - 1; i >= 0; i--) {
		double lb = i * yBin + offset;
		snprintf(axLabel, 31, "%6.2f", lb);
		string lbl(axLabel);
		std::cout << lbl << " " << s.substr(i * ncols, ncols) << "\n";
	}
}

template<typename data_t>
static void imshow(data_t* data, length_t nrows, length_t ncols, string title=string(),
	bool rowmajor=true, bool twoEnded=false, int downsample_ncols=90) {

	downsample_ncols = static_cast<int>(min(ncols, downsample_ncols));

	static const char symbols2[2] = {' ', '+'};
	static const char symbols3[3] = {'.', ' ', '+'};
	static const char symbols4[4] = {'=', '-', 'o', 'O'};
	static const char symbols5[5] = {'=', '-', ' ', 'o', 'O'};
	static const char symbols6[6] = {'#', '=', '-', '.', 'o', 'O'};
	static const char symbols7[7] = {'#', '=', '-', ' ', '.', 'o', 'O'};

	static const char symbolsOneEnded[7] = {' ', '.',':','o','x','O','X'};

	int sz = static_cast<int>(nrows * downsample_ncols);
	auto minVal = min(data, sz);
	auto maxVal = max(data, sz);
	auto range = maxVal - minVal;

	string s(sz, ' '); // whitespace

	// determine how many levels to use
	size_t nuniqs = unique(data, sz).size();

	if (nuniqs <= 1) {
		std::cout << "imshow(): array had constant value " << data[0];
		return;
	}
	length_t nlevels = min(nuniqs, 7);
	const char* symbols;
	if (twoEnded) {
		switch (nlevels) {
			case 2: symbols = symbols2; break;
			case 3: symbols = symbols3; break;
			case 4: symbols = symbols4; break;
			case 5: symbols = symbols5; break;
			case 6: symbols = symbols6; break;
			default:
				symbols = symbols7;
		}
	} else {
		symbols = symbolsOneEnded;
		// nlevels = min(nuniqs, 8); // 8 symbols in this progression
	}
	// const char* symbols = symbolsTbl[nlevels-1];


	double binWidth = static_cast<double>(range) / nlevels;
	double binMin = binWidth * static_cast<int>(floor(minVal / binWidth));

	_print_title(title, downsample_ncols);

	// print each row of the data; lower row indices print at the top
	int colStride = static_cast<int>(ceil(ncols / downsample_ncols));
	// int colStride = 1;
	for (int i = 0; i < nrows; i++) {
		length_t strRowIdx = i * downsample_ncols;
		for (int j = 0; j < downsample_ncols; j++) {
			length_t strIdx = strRowIdx + j;
			length_t dataIdx;
			if (rowmajor) {
				dataIdx = (i * ncols) + (j * colStride);
			} else {
				dataIdx = (j * nrows * colStride) + i;
			}
			length_t level = static_cast<length_t>(floor((data[dataIdx] - binMin) / binWidth));
			level = max(level, 0);
			level = min(level, nlevels-1);
			// if (level) {
			// 	PRINT_VAR(level);
			// }
			// PRINT_VAR(level);
			// char symbol = symbols[level]; // TODO remove
			s[strIdx] = symbols[level];
		}
		std::cout << s.substr(strRowIdx, ncols) << "\n";
	}

	// print a legend showing what min value each symbol corresponds to
	char binValStr[32];
	for (int i = 0; i < nlevels; i++) {
		double binStart = binMin + i * binWidth;
		snprintf(binValStr, 31, "%5.2f", binStart);
		std::cout << "'" << symbols[i] << "':" << binValStr;
		if (i < nlevels - 1) {
			std::cout << "\t";
		}
	}
	std::cout << "+\n";
}

} // namespace ar

#endif // __PLOT_HPP
