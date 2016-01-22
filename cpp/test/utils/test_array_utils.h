//
//  test_array_utils.h
//  edtw
//
//  Created By <Anonymous> on 1/15/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#ifndef timekit_test_array_utils_h
#define timekit_test_array_utils_h

#ifdef __cplusplus
extern "C" {
#endif

void test_array_utils_all();
void resample_sameSampRate_correct();
void resample_upsampleByInteger_correct();
void resample_upsampleByFraction_correct();
void resample_downsampleByInteger_correct();
void resample_downsampleByFraction_correct();
void copyReverse_evenLen_correct();
void copyReverse_oddLen_correct();
void reshape_dimsNotFactorOfLen_returnsNull();
void reshape_stillOneDim_unchanged();
void reshape_twoDims_correct();
void reshape_threeDims_correct();
	
#ifdef __cplusplus
}
#endif

#endif
