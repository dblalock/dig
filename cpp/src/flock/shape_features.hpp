//
//  shape_features.hpp
//  Dig
//
//  Created by DB on 3/9/16.
//  Copyright (c) 2016 DB. All rights reserved.
//


#ifndef DIG_SHAPE_FEATURES_HPP
#define DIG_SHAPE_FEATURES_HPP

#include <Dense>

using Eigen::VectorXd;

VectorXd crossCorrelate(VectorXd shorter, VectorXd longer);

#endif
