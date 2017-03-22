/*
 * Engine.cuh
 *
 *  Created on: Mar 14, 2017
 *  Author: Ajay K. Sampathirao, P. Sopasakis
 */

/*TODO Add proper copyright notice here and in every file */

#ifndef ENGINE_CUH_
#define ENGINE_CUH_

#include "DefinitionHeader.h"
#include "DWNnetwork.cuh"
#include "Forecaster.cuh"
#include "unitTestHeader.cuh"

/*TODO IF A METHOD IS NOT TO BE INVOKED FROM THE OUTSIDE OF THIS CLASS, MAKE IT PRIVATE */
/*TODO INTRODUCE GETTERS FOR THOSE FIELDS WHICH NEED TO BE ACCESSIBLE FROM THE OUTSIDE */

/**
 *TODO Document this class (what is its purpose)
 *TODO Rename this file to Engine.cuh
 */
class Engine{
public:

	/**
	 *
	 * @param network
	 * @param forecaster
	 *TODO remove `unitTest` from here
	 */
	Engine(
		DWNnetwork *network,
		Forecaster *forecaster,
		unitTest *ptrMyTestor);

	/**
	 *
	 */
	 /*TODO It seems to me that this method should become private */
	void allocateForecastDevice();

	/**
	 *
	 */
	 /*TODO It seems to me that this method should become private */
	void allocateSystemDevice();

	/**
	 *
	 */
	 /*TODO It seems to me that this method should become private */
	void initialiseForecastDevice();

	/**
	 *
	 */
	 /*TODO It seems to me that this method should become private */
	void initialiseSystemDevice();

	/**
	 *
	 */
	void factorStep();

	/**
	 *
	 */
	void updateDistubance();

	/**
	 *
	 */
	void eliminateInputDistubanceCoupling();

	/**
	 *
	 */
	void updateStateControl();

	/**
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

	/*TODO There should be no test methods in classes */
	void testInverse();

	/*TODO There should be no test methods in classes */
	void testPrecondtioningFunciton();

	/**
	 *
	 */
	/*TODO It seems to me that this method should become private */
	void deallocateForecastDevice();

	/**
	 *
	 */
	/*TODO It seems to me that this method should become private */
	void deallocateSystemDevice();

	/*TODO REMOVE */
	void testStupidFunction();

	/*TODO REMOVE Friendship*/
	friend class SMPCController;

	/**
	 * Destructor
	 */
	~Engine();

private:
	/**
	 *
	 */
	DWNnetwork *ptrMyNetwork;
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
