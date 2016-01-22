//
//  test_ucr_funcs.h
//  edtw
//
//  Created By <Anonymous> on 1/12/14.
//  Copyright (c) 2014 University of <Anonymous>. All rights reserved.
//

#ifndef timekit_test_ucr_funcs_h
#define timekit_test_ucr_funcs_h

#ifdef __cplusplus
extern "C" {
#endif

void test_ucr_funcs_all(); /// calls all unit tests defined in this file

void Znormalize_varianceIsOne();
void Znormalize_meanIsZero();

void Envelope_Warp0_Correct();
void Envelope_Warp2_Correct();
	
void USEnvelope_noScaling_correct();
void USEnvelope_scaleDownOnly_correct();
void USEnvelope_scaleUpOnly_correct();
void USEnvelope_scaleUpAndDown_correct();

void EuclideanDist_FullComparison_correctDistance();
void EuclideanDist_EarlyAbandon_correctDistance();
	
void EuclideanSearch_BufferShorterThanQuery_ReturnsFailure();
void EuclideanSearch_QueryLenZero_ReturnsFailure();
void EuclideanSearch_QueryLenNegative_ReturnsFailure();
void EuclideanSearch_BufferLenZero_ReturnsFailure();
void EuclideanSearch_BufferLenNegative_ReturnsFailure();
void EuclideanSearch_NullQuery_ReturnsFailure();
void EuclideanSearch_NullBuffer_ReturnsFailure();
void EuclideanSearch_NullResult_ReturnsFailure();
void EuclideanSearch_EqualLenArrays_Correct();
void EuclideanSearch_DifferentLenArrays_Correct();

void USDist_OneLen_ReturnsEuclideanDist();
void USDist_MaxLenEqualsM_CorrectDistance();
void USDist_MinLenEqualsM_CorrectDistance();
void USDist_MinLessAndMaxGreater_CorrectDistance();

void DTWDist_NoWarpFullComparison_correctDistance();
void DTWDist_Warp1FullComparison_correctDistance();
void DTWDist_Warp2FullComparison_correctDistance();
void DTWDist_Warp1FullComparison_NeedsMoreWarp_correctDistance();
void DTWDist_NoWarpEarlyAbandon_correctDistance();
	
void DTWSearch_BufferShorterThanQuery_ReturnsFailure();
void DTWSearch_QueryLenZero_ReturnsFailure();
void DTWSearch_QueryLenNegative_ReturnsFailure();
void DTWSearch_BufferLenZero_ReturnsFailure();
void DTWSearch_BufferLenNegative_ReturnsFailure();
void DTWSearch_NullQuery_ReturnsFailure();
void DTWSearch_NullBuffer_ReturnsFailure();
void DTWSearch_NullResult_ReturnsFailure();
void DTWSearch_NegativeWarp_ReturnsFailure();
void DTWSearch_EqualLenArrays_CorrectLocation();
void DTWSearch_EqualLenArrays_CorrectDistance();
void DTWSearch_NoWarp_EqualLenArrays_Correct();
void DTWSearch_Warp_EqualLenArrays_Correct();
void DTWSearch_DifferentLenArrays_Correct();

#ifdef __cplusplus
}
#endif

#endif
