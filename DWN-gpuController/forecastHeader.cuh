/*
 * Forecaster.cuh
 *
 *  Created on: Feb 23, 2017
 *      Author: Ajay K. Sampathirao, P. Sopasakis
 */

#ifndef FORECASTCLASS_CUH_
#define FORECASTCLASS_CUH_


//TODO forecast function, dHat variable allocation

/*TODO Rename this file into `Forecaster` */
/*TODO Add documentation; explain what this class is about. */
class Forecaster{

public:

	/**
	 *
	 */
	Forecaster(
		string pathToFile);
	/**
	 *
	 */
	~Forecaster();

	/*TODO REMOVE Friendship */
	friend class Engine;

	/*TODO REMOVE Friendship */
	friend class SMPCController;
private:
	/**
	 * ?
	 */
	uint_t N;
	/**
	 * ?
	 */
	uint_t K;
	/**
	 * Total number of nodes.
	 */
	uint_t N_NODES; /*TODO Fields should not be all-caps, but camelCase.*/
	/**
	 * Total number of children.
	 */
	uint_t N_CHILDREN_TOT;
	/**
	 * Number of non-leaf tree nodes.
	 */
	uint_t N_NONLEAF_NODES;
	/**
	 *
	 */
	uint_t DIM_NODE;
	/**
	 *
	 */
	uint_t *stages;
	/**
	 * Nodes of a given stage.
	 */
	uint_t *nodesPerStage;
	/**
	 *
	 */
	uint_t *nodesPerStageCumul;
	/**
	 * Indexes of the leaf nodes.
	 */
	uint_t *leaves;
	/**
	 * Indices of the children nodes for each node.
	 */
	uint_t *children;
	/**
	 * Ancestor of a node.
	 */
	uint_t *ancestor;
	/**
	 * Number of children of each node.
	 */
	uint_t *nChildren;
	/**
	 *
	 */
	uint_t *nChildrenCumul;
	/**
	 * Probability of a node.
	 */
	real_t *probNode;
	/**
	 * Value on a node.
	 */
	real_t *valueNode;
	/**
	 *
	 */
	real_t *dHat;
};



#endif /* FORECASTCLASS_CUH_ */
