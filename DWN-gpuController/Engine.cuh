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

/*TODO Add proper copyright notice here and in every file */

#ifndef ENGINE_CUH_
#define ENGINE_CUH_

#include "Configuration.h"
#include "DwnNetwork.cuh"
#include "Forecaster.cuh"
#include "unitTestHeader.cuh"
#include "Utilities.cuh"

/*TODO IF A METHOD IS NOT TO BE INVOKED FROM THE OUTSIDE OF THIS CLASS, MAKE IT PRIVATE */
/*TODO INTRODUCE GETTERS FOR THOSE FIELDS WHICH NEED TO BE ACCESSIBLE FROM THE OUTSIDE */

/**
 *TODO Document this class (what is its purpose)
 */
class Engine{
public:

	/**
	 *
	 * @param network
	 * @param forecaster
	 * @todo remove `unitTest` from here
	 */
	Engine( DwnNetwork *network,
			ScenarioTree *scenarioTree,
			SmpcConfiguration *smpcConfig);

	/**
	 * @todo It seems to me that this method should become private
	 */
	void allocateForecastDevice();

	/**
	 * @todo It seems to me that this method should become private
	 */
	void allocateSystemDevice();

	/**
	 * @todo It seems to me that this method should become private
	 */
	void initialiseForecastDevice();

	/**
	 * @todo It seems to me that this method should become private
	 */
	void initialiseSystemDevice();

	/**
	 * @todo It seems to me that this method should become private
	 */
	void factorStep();

	/**
	 * @todo explain how this works - does it read from a file?
	 */
	void updateDistubance();

	/**
	 * Eliminate input-demand coupling equations
	 * @param   nominalDemand    demand predicted
	 * @param   nominalPrice     price prediction
	 */
	void eliminateInputDistubanceCoupling(real_t* nominalDemand, real_t *nominalPrices);

	/**
	 * ?
	 */
	void updateStateControl();

	/**
	 *
	 * @param src
	 * @param dst
	 * @param n
	 * @param batchSize
	 */
	/*TODO float** --> real_t ** */
	/*TODO int --> uint_t */
	void inverseBatchMat(
		float** src,
		float** dst,
		int n,
		int batchSize);

	/**
	 * @todo There should be no test methods in classes
	 */
	void testInverse();

	/**
	 * @todo There should be no test methods in classes
	 */
	void testPrecondtioningFunciton();

	/**
	 * @todo make private
	 */
	void deallocateForecastDevice();

	/**
	 * @todo make private
	 */
	void deallocateSystemDevice();

	/**
	 * @todo remove method
	 */
	void testStupidFunction();

	/**
	 * @todo remove Friendship
	 */
	friend class SmpcController;

	/**
	 * Destructor
	 */
	~Engine();

private:
	/**
	 * Pointer to the network
	 */
	DwnNetwork *ptrMyNetwork;
	/**
	 * Pointer to the scenario tree
	 */
	ScenarioTree *ptrMyScenarioTree;
	/**
	 * Pointer to the Smpc configuration
	 */
	SmpcConfiguration *ptrMySmpcConfig;
	/* --- NETWORK --- */
	/**
	 * matrix B
	 */
	real_t *devSysMatB;
	/**
	 * constraints matrix F
	 */
	real_t *devSysMatF;
	/**
	 * constraints matrix G
	 */
	real_t *devSysMatG;
	/**
	 * matrix L
	 */
	real_t *devSysMatL;
	/**
	 * matrix Lhat
	 */
	real_t  *devSysMatLhat;
	/**
	 * pointer to Matrix B
	 */
	real_t **devPtrSysMatB;
	/**
	 * pointer to matrix F
	 */
	real_t  **devPtrSysMatF;
	/**
	 * pointer to matrix G
	 */
	real_t  **devPtrSysMatG;
	/**
	 * pointer to matrix L
	 */
	real_t  **devPtrSysMatL;
	/**
	 * pointer to matrix Lhat
	 */
	real_t  **devPtrSysMatLhat;
	/**
	 * previous control
	 */
	real_t *devVecPreviousControl;
	/**
	 * current state
	 */
	real_t  *devVecCurrentState;
	/**
	 * previous uhat
	 */
	real_t  *devVecPreviousUhat;




	/* --- NETWORK CONSTRAINTS --- */

	/**
	 * state/volume minimum
	 */
	real_t *devSysXmin;
	/**
	 * state/volume maximum
	 */
	real_t *devSysXmax;
	/**
	 * state/volume safe level
	 */
	real_t  *devSysXs;
	/**
	 * dummy state/volume safe level
	 */
	real_t  *devSysXsUpper;
	/**
	 * actuator/control minimum
	 */
	real_t  *devSysUmin;
	/**
	 * actuator/cotrol maximum
	 */
	real_t  *devSysUmax;




	/* --- COST FUNCTION --- */

	/**
	 * smooth operation cost W
	 */
	real_t *devSysCostW;
	/**
	 * pointer to smooth operation cost W
	 */
	real_t **devPtrSysCostW;


	/* --- SCENARIO TREE --- */

	/**
	 * Array of stages
	 */
	uint_t *devTreeStages;
	/**
	 * Array of nodes per stage
	 */
	uint_t *devTreeNodesPerStage;
	/**
	 * Array of past nodes
	 */
	uint_t *devTreeNodesPerStageCumul;
	/**
	 * Array of the leaves
	 */
	uint_t *devTreeLeaves;
	/**
	 * Array number of children
	 */
	uint_t *devTreeNumChildren;
	/**
	 * Array of ancestor
	 */
	uint_t *devTreeAncestor;
	/**
	 * Array of past cumulative children
	 */
	uint_t *devTreeNumChildrenCumul;
	/**
	 * Array of the probability
	 */
	real_t *devTreeProb;
	/**
	 * Array of the error in the demand
	 */
	real_t *devTreeErrorDemand;
	/**
	 * Array of the error in the prices
	 */
	real_t *devTreeErrorPrices;


	/* --- FACTOR MATRICES --- */

	/**
	 *  matrix Phi
	 */
	real_t *devMatPhi;
	/**
	 * matrix Psi
	 */
	real_t *devMatPsi;
	/**
	 * matrix Theta
	 */
	real_t *devMatTheta;
	/**
	 * matrix Theta
	 */
	real_t *devMatOmega;
	/**
	 * matrix Sigma
	 */
	real_t *devMatSigma;
	/**
	 * matrix D
	 */
	real_t *devMatD;
	/**
	 * matrix F (Factor step)
	 */
	real_t *devMatF;
	/**
	 * matrix G (Facotr step)
	 */
	real_t *devMatG;
	/**
	 * pointer matrix Phi
	 */
	real_t **devPtrMatPhi;
	/**
	 * pointer matrix Psi
	 */
	real_t **devPtrMatPsi;
	/**
	 * pointer matrix Theta
	 */
	real_t **devPtrMatTheta;
	/**
	 * pointer matrix Omega
	 */
	real_t **devPtrMatOmega;
	/**
	 * pointer matrix Sigma
	 */
	real_t **devPtrMatSigma;
	/**
	 * pointer matrix D
	 */
	real_t **devPtrMatD;
	/**
	 * pointer matrix F (Factor step)
	 */
	real_t **devPtrMatF;
	/**
	 * pointer matrix G (Factor step)
	 */
	real_t **devPtrMatG;
	/**
	 * uhat
	 */
	real_t *devVecUhat;
	/**
	 * beta control-distribution elimination
	 */
	real_t *devVecBeta;
	/**
	 * e control-disturbance elimination
	 */
	real_t *devVecE;
	/**
	 * cublas handler
	 */
	cublasHandle_t handle;
};

#endif /* ENGINE_CUH_ */
