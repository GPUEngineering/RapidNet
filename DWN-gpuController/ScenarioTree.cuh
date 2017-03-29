/*
 *    GPU-accelerated scenario-based stochastic MPC for the operational
 *    management of drinking water networks.
 *    Copyright (C) 2017 Ajay. K. Sampathirao and P. Sopasakis
 *
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
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
#define VARNAME_DIM_PRICE "dimPrice"
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
	 * Return the prediction horizon
	 */
	uint_t getPredHorizon();

	/*
	 * Return the number of scenarios (leaves)
	 */
	uint_t getNumScenarios();

	/*
	 * Return the number of nodes
	 */
	uint_t getNumNodes();

	/*
	 * Return the number of children
	 */
	uint_t getNumChildrenTot();

	/*
	 * Return the number of number of non leaf nodes
	 */
	uint_t getNumNonleafNodes();

	/*
	 * Return the pointer to the array representing the stages of the
	 * node
	 */
	uint_t* getStageNodes();

	/*
	 * Return the pointer to the array for the nodes at each stage
	 */
	uint_t* getNodesPerStage();

	/*
	 * Return the pointer to the cumulative nodes at each stage
	 */
	uint_t* getNodesPerStageCumul();

	/*
	 * Return the pointer to the array for the list of leaves
	 */
	uint_t* getLeaveArray();

	/*
	 * Return the pointer to the array of children
	 */
	uint_t* getChildArray();

	/*
	 * Return the pointer to the array of ancestors
	 */
	uint_t* getAncestorArray();

	/*
	 * Return the pointer to the array of number of children at each node
	 */
	uint_t* getNumChildren();

	/*
	 * Return the pointer to the array of number of children at the past node
	 */
	uint_t* getNumChildrenCumul();

	/*
	 * Return the pointer to the array of probabilities of the node
	 */
	real_t* getProbArray();

	/*
	 * Return the pointer to the array of the error in demand at every node
	 */
	real_t* getErrorDemandArray();

	/*
	 * Return the pointer to the array of the error in prices at every node
	 */
	real_t* getErrorPriceArray();
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
	uint_t *stageArray;
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
	uint_t *leaveArray;
	/**
	 * Indices of the children nodes for each node.
	 */
	uint_t *childArray;
	/**
	 * Vector of length nNodes and contain the index of the ancestor of the node
	 * with root node having the index zero
	 */
	uint_t *ancestorArray;
	/**
	 * Vector of length nNonLeafNodes and contain the number of children of
	 * each node.
	 */
	uint_t *nChildArray;
	/**
	 * Vector of length nNonLeafNodes and contain the sum of past children at node
	 */
	uint_t *nChildCumulArray;
	/**
	 * Probability of a node.
	 */
	real_t *probNodeArray;
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
	real_t *errorDemandArray;
	/**
	 * Price error at a node
	 */
	real_t *errorPriceArray;
};


#endif /* SCENARIOTREE_CUH_ */
