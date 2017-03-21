/*
 * SMPControllerClass.cuh
 *
 *  Created on: Mar 1, 2017
 *      Author: control
 */

#ifndef SMPCONTROLLERCLASS_CUH_
#define SMPCONTROLLERCLASS_CUH_

#include "DefinitionHeader.h"
#include "EngineHeader.cuh"

class SMPCController{
public:
	SMPCController(Engine *myEngine);
	void dualExtrapolationStep(real_t lambda);
	void solveStep();
	void proximalFunG();
	void dualUpdate();
	void algorithmApg();
	void controllerSmpc();
	~SMPCController();
private:
	Engine* ptrMyEngine;
	real_t *devVecX, *devVecU, *devVecV, *devVecXi, *devVecPsi, *devVecAcceleratedXi, *devVecAcceleratedPsi,
	*devVecPrimalXi, *devVecPrimalPsi, *devVecDualXi, *devVecDualPsi, *devVecUpdateXi, *devVecUpdatePsi;
	real_t **devPtrVecX, **devPtrVecU, **devPtrVecV, **devPtrVecAcceleratedPsi, **devPtrVecAcceleratedXi,
	**devPtrVecPrimalPsi, **devPtrVecPrimalXi;
	real_t *devPrimalInfeasibilty;
	real_t *devVecQ, *devVecR;
	real_t **devPtrVecQ, **devPtrVecR;
	uint_t MAX_ITERATIONS  = 500;
	real_t stepSize;
};

__global__  void solveSumChildren(real_t *src, real_t *dst, uint_t *devTreeNumChildren, uint_t *devTreeNumChildCumul,
		  uint_t iStageCumulNodes, uint_t iStageNodes, uint_t iStage, uint_t dim);
__global__ void solveChildNodesUpdate(real_t *src, real_t *dst, uint_t *devTreeAncestor,uint_t nextStageCumulNodes, uint_t dim);


#endif /* SMPCONTROLLERCLASS_CUH_ */
