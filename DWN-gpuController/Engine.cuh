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
	Engine(
	    DwnNetwork *network,
		  Forecaster *forecaster,
		  unitTest *ptrMyTestor);

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
	 * @todo It seems to me that this method should become private
	 */
	void eliminateInputDistubanceCoupling();

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
	 */
	 * @todo make private
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
	 *
	 */
	DwnNetwork *ptrMyNetwork;
	/**
	 *
	 */
	Forecaster *ptrMyForecaster;
	/**
	 *
	 */
	unitTest   *ptrMyTestor;

	/* --- NETWORK --- */
	/**
	 *
	 */
	real_t *devSysMatB;
	/**
	 *
	 */
	real_t *devSysMatF;
	/**
	 *
	 */
	real_t *devSysMatG;
	/**
	 *
	 */
	real_t *devSysMatL;
	/**
	 *
	 */
	real_t  *devSysMatLhat;
	/**
	 *
	 */
	real_t **devPtrSysMatB;
	/**
	 *
	 */
	real_t  **devPtrSysMatF;
	/**
	 *
	 */
	real_t  **devPtrSysMatG;
	/**
	 *
	 */
	real_t  **devPtrSysMatL;
	/**
	 *
	 */
	real_t  **devPtrSysMatLhat;
	/**
	 *
	 */
	real_t *devVecPreviousControl;
	/**
	 *
	 */
	real_t  *devVecCurrentState;
	/**
	 *
	 */
	real_t  *devVecPreviousUhat;




	/* --- NETWORK CONSTRAINTS --- */

	/**
	 *
	 */
	real_t *devSysXmin;
	/**
	 *
	 */
	real_t *devSysXmax;
	/**
	 *
	 */
	real_t  *devSysXs;
	/**
	 *
	 */
	real_t  *devSysXsUpper;
	/**
	 *
	 */
	real_t  *devSysUmin;
	/**
	 *
	 */
	real_t  *devSysUmax;




	/* --- COST FUNCTION --- */

	/**
	 *
	 */
	real_t *devSysCostW;
	/**
	 *
	 */
	real_t **devPtrSysCostW;


	/* --- FORECASTER --- */

	/**
	 *
	 */
	uint_t *devTreeStages;
	/**
	 *
	 */
	uint_t *devTreeNodesPerStage;
	/**
	 *
	 */
	uint_t *devTreeNodesPerStageCumul;
	/**
	 *
	 */
	uint_t *devTreeLeaves;
	/**
	 *
	 */
	uint_t *devTreeNumChildren;
	/**
	 *
	 */
	uint_t *devTreeAncestor;
	/**
	 *
	 */
	uint_t *devTreeNumChildrenCumul;
	/**
	 *
	 */
	real_t *devTreeProb;
	/**
	 *
	 */
	real_t *devTreeValue;
	/**
	 *
	 */
	real_t *devForecastValue;





	/* --- FACTOR MATRICES --- */

	/**
	 *
	 */
	real_t *devMatPhi;
	/**
	 *
	 */
	real_t *devMatPsi;
	/**
	 *
	 */
	real_t *devMatTheta;
	/**
	 *
	 */
	real_t *devMatOmega;
	/**
	 *
	 */
	real_t *devMatSigma;
	/**
	 *
	 */
	real_t *devMatD;
	/**
	 *
	 */
	real_t *devMatF;
	/**
	 *
	 */
	real_t *devMatG;
	/**
	 *
	 */
	real_t **devPtrMatPhi;
	/**
	 *
	 */
	real_t **devPtrMatPsi;
	/**
	 *
	 */
	real_t **devPtrMatTheta;
	/**
	 *
	 */
	real_t **devPtrMatOmega;
	/**
	 *
	 */
	real_t **devPtrMatSigma;
	/**
	 *
	 */
	real_t **devPtrMatD;
	/**
	 *
	 */
	real_t **devPtrMatF;
	/**
	 *
	 */
	real_t **devPtrMatG;
	/**
	 *
	 */
	real_t *devVecUhat;
	/**
	 *
	 */
	real_t *devVecBeta;
	/**
	 *
	 */
	real_t *devVecE;
	/**
	 *
	 */
	cublasHandle_t handle;
};

#endif /* ENGINE_CUH_ */
