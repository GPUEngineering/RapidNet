/*
 * forecastClass.cuh
 *
 *  Created on: Feb 23, 2017
 *      Author: control
 */

#ifndef FORECASTCLASS_CUH_
#define FORECASTCLASS_CUH_
#include "DefinitionHeader.h"

//TODO forecast function, dHat variable allocation
class Forecaster{
public:
	Forecaster( string pathToFile );
	~Forecaster();
	friend class Engine;
	friend class SMPCController;
private:
	uint_t N, K, N_NODES, N_CHILDREN_TOT, N_NONLEAF_NODES, DIM_NODE;
	uint_t *stages, *nodesPerStage, *nodesPerStageCumul, *leaves, *children,
	*ancestor, *nChildren, *nChildrenCumul;
	real_t *probNode, *valueNode, *dHat;
};



#endif /* FORECASTCLASS_CUH_ */
