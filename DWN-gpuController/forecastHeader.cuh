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
	 *
	 */
	uint_t N;
	/**
	 *
	 */
	uint_t K;
	/**
	 *
	 */
	uint_t N_NODES; /*TODO Fields should not be all-caps, but camelCase.*/
	uint_t N_CHILDREN_TOT;
	uint_t N_NONLEAF_NODES;
	uint_t DIM_NODE;
	uint_t *stages;
	uint_t *nodesPerStage;
	uint_t *nodesPerStageCumul;
	uint_t *leaves;
	uint_t *children;
	uint_t *ancestor;
	uint_t *nChildren;
	uint_t *nChildrenCumul;
	real_t *probNode;
	real_t *valueNode;
	real_t *dHat;
};



#endif /* FORECASTCLASS_CUH_ */
