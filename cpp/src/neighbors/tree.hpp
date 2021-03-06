//
//  tree.hpp
//  Dig
//
//  Created by DB on 1/24/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#ifndef tree_hpp
#define tree_hpp

#include <stdio.h>
//#include <unordered_map>
//#include <map>
#include <vector>

#include "Dense"

#include "neighbors.hpp"
#include "intmap.hpp"

// ================================================================
// Typedefs and usings
// ================================================================

using std::pair;
using std::make_pair;
using std::unique_ptr;

//using std::unordered_map;
//using std::map;
using std::vector;

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::MatrixXi;
using Eigen::ArrayXi;

// typedef int16_t hash_t;
typedef int8_t hash_t; // note: causes weird debug output since is signed char
typedef int32_t length_t; // only handle up to length 2 billion
typedef int16_t depth_t;
typedef int64_t obj_id_t;

// ================================================================
// Structs
// ================================================================

// ------------------------------------------------ Node


// TODO abstract node class that knows if internal, and child classes
// for internal and leaf nodes

typedef struct Node {
	//	std::map<hash_t, struct Node> children;
	intmap64<std::unique_ptr<struct Node> > children;
	//	std::array<length_t, MAX_POINTS_PER_LEAF> points;
	vector<length_t> points; // TODO array of fixed size
	bool is_internal; /// flag indicating that this is not a leaf node
public:
	Node();
	//	Node(const Node& other);
} Node;

//typedef std::map<hash_t, Node> map_t;
typedef std::unique_ptr<Node> node_ptr_t;
typedef intmap64<node_ptr_t> map_t;


// ================================================================
// Utility funcs
// ================================================================

double squaredL2Dist(const VectorXd& x, const VectorXd& y);

VectorXd distsToVector(const MatrixXd& X, const VectorXd& v);
ArrayXd distsToVectors(const MatrixXd& X, const MatrixXd& V);

VectorXd squaredDistsToVector(const MatrixXd& X, const VectorXd& v);
ArrayXd squaredDistsToVectors(const MatrixXd& X, const MatrixXd& V);

// ================================================================
// Tree funcs
// ================================================================

MatrixXd computeProjectionVects(const MatrixXd& X, depth_t numVects=16);

unique_ptr<Node> constructIndex(const MatrixXd& X, const MatrixXd& projectionVects,
								double& binWidth);

vector<length_t> findNeighbors(const VectorXd& q, const MatrixXd& X,
	Node& root, const MatrixXd& projectionVects, double radiusL2,
	double binWidth);

Neighbor find1nn(const VectorXd& q, const MatrixXd& X, Node& root,
				 const MatrixXd& projectionVects, double binWidth);

vector<Neighbor> findKnn(const VectorXd& q, const MatrixXd& X, uint16_t k,
						Node& root, const MatrixXd& projectionVects,
						double binWidth, uint8_t numGuessesMultipleOfK=5);










#endif /* tree_hpp */
