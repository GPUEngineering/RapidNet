/*
 * ScenarioTree.cuh
 *
 *  Created on: Mar 22, 2017
 *      Author: Ajay Kumar, P. Sopasakis
 */

#ifndef SCENARIOTREE_CUH_
#define SCENARIOTREE_CUH_
#define VARNAME_N  "N"
#define VARNAME_K  "K"
#define VARNAME_NODES "nodes"
#define VARNAME_NUM_NONLEAF "nNonLeafNodes"
#define VARNAME_NUM_CHILD_TOT "nChildrenTot"
#define VARNAME_STAGES "stages"
#define VARNAME_NODES_PER_STAGE "nodesPerStage"
#define VARNAME_NODES_PER_STAGE_CUMUL "nodesPerStageCumul"
#define VARNAME_LEAVES "leaves"
#define VARNAME_CHILDREN "children"
#define VARNAME_ANCESTOR "ancestor"
#define VARNAME_NUM_CHILDREN "nChildren"
#define VARNAME_NUM_CHILD_CUMUL "nChildrenCumul"
#define VARNAME_PROB_NODE "probNode"
#define VARNAME_DIM_DEMAND "dimDemand"
#define VARNAME_DIM_PRICE "dimPrices"
#define VARNAME_DEMAND_NODE "valueDemandNode"
#define VARNAME_PRICE_NODE "valuePriceNode"
#include "Configuration.h"


/* The nominal values have
 * error with respect to the actual values. In scenario-based MPC,
 * the error is represented with a scenario tree.
 *
 * - scenario tree used to represent the error in the predictions
 *       - nodes at a stage
 *       - children of a node
 *       - ancestor of a node
 *       - probability of a node
 *       - value of a node both the price and the water demand
 */

class ScenarioTree{
public:
	/**
	 * Constructor for the Scenario tree generated from the given JSON file.
	 *
	 * @param pathToFile filename of a JSON file containing
	 * 		     a representation of the scenario tree structure.
	 */
	ScenarioTree(string pathToFileName);

	/*
	 * Default destructor.
	 */
	~ScenarioTree();

private:
	/**
	 *  Prediction horizon
	 */
	uint_t nPredHorizon;
	/**
	 * Number of scenarios
	 */
	uint_t nScenario;
	/**
	 * Total number of nodes.
	 */
	uint_t nNodes;
	/**
	 * Total number of children.
	 */
	uint_t nChildrenTot;
	/**
	 * Number of non-leaf tree nodes.
	 */
	uint_t nNonleafNodes;
	/**
	 * Vector of length nNodes and represents at stage of each node
	 */
	uint_t *stages;
	/**
	 * Vector of length N and represents how many nodes at given stage.
	 */
	uint_t *nodesPerStage;
	/**
	 * Vector of length N and represents the number of nodes past nodes
	 */
	uint_t *nodesPerStageCumul;
	/**
	 * Vector of length K and contains the indexes of the leaf nodes.
	 */
	uint_t *leaves;
	/**
	 * Indices of the children nodes for each node.
	 */
	uint_t *children;
	/**
	 * Vector of length nNodes and contain the index of the ancestor of the node
	 * with root node having the index zero
	 */
	uint_t *ancestor;
	/**
	 * Vector of length nNonLeafNodes and contain the number of children of
	 * each node.
	 */
	uint_t *nChildren;
	/**
	 * Vector of length nNonLeafNodes and contain the sum of past children at node
	 */
	uint_t *nChildrenCumul;
	/**
	 * Probability of a node.
	 */
	real_t *probNode;
	/**
	 * Dimension of the demand
	 */
	uint_t dimDemand;
	/**
	 * Dimension of the demand
	 */
	uint_t dimPrice;
	/**
	 * Demand error at a node.
	 */
	real_t *valueDemandNode;
	/**
	 * Price error at a node
	 */
	real_t *valuePriceNode;
};


#endif /* SCENARIOTREE_CUH_ */
