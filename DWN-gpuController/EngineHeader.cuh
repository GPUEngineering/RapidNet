/*
 * Engine.cuh
 *
 *  Created on: Mar 14, 2017
 *      Author: Ajay Kumar
 */

#ifndef ENGINE_CUH_
#define ENGINE_CUH_

#include "DefinitionHeader.h"
#include "networkHeader.cuh"
#include "forecastHeader.cuh"
#include "unitTestHeader.cuh"
//#include "cudaKernalHeader.cuh"

class Engine{
public:
	Engine(DWNnetwork *myNetwork, Forecaster *myForecaster, unitTest *ptrMyTestor);
	void allocateForecastDevice();
	void allocateSystemDevice();
	void initialiseForecastDevice();
	void initialiseSystemDevice();
	void factorStep();
	void updateDistubance();
	void eliminateInputDistubanceCoupling();
	void updateStateControl();
	//void invertMat(real_t** src, real_t** dst, uint_t n, uint_t batchSize);
	void inverseBatchMat(float** src, float** dst, int n, int batchSize);
	void testInverse();
	void testPrecondtioningFunciton();
	void deallocateForecastDevice();
	void deallocateSystemDevice();
	void testStupidFunction();
	friend class SMPCController;
	~Engine();
private:
	DWNnetwork	*ptrMyNetwork;
	Forecaster  *ptrMyForecaster;
	unitTest *ptrMyTestor;
	// network
	real_t *devSysMatB, *devSysMatF, *devSysMatG, *devSysMatL, *devSysMatLhat;
	real_t **devPtrSysMatB, **devPtrSysMatF, **devPtrSysMatG, **devPtrSysMatL, **devPtrSysMatLhat;
	real_t *devVecPreviousControl, *devVecCurrentState, *devVecPreviousUhat;
	// network constraints
	real_t *devSysXmin, *devSysXmax, *devSysXs, *devSysXsUpper, *devSysUmin, *devSysUmax;
	// cost function
	real_t *devSysCostW;
	real_t **devPtrSysCostW;
	// forecaster memory
	uint_t *devTreeStages, *devTreeNodesPerStage, *devTreeNodesPerStageCumul, *devTreeLeaves,
	*devTreeNumChildren, *devTreeAncestor, *devTreeNumChildrenCumul;
	real_t *devTreeProb,*devTreeValue, *devForecastValue;
	// factor matrices
	real_t *devMatPhi, *devMatPsi, *devMatTheta, *devMatOmega, *devMatSigma, *devMatD,
	*devMatF, *devMatG;
	real_t **devPtrMatPhi, **devPtrMatPsi, **devPtrMatTheta, **devPtrMatOmega,
	**devPtrMatSigma, **devPtrMatD, **devPtrMatF, **devPtrMatG;
	real_t *devVecUhat, *devVecBeta, *devVecE;
	cublasHandle_t handle;
};

#endif /* ENGINE_CUH_ */
