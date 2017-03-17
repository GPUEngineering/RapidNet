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
	void solveStep();

	~SMPCController();
private:
	Engine* ptrMyEngine;
	real_t *devVecX, *devVecU, *devVecV, *devVecXi, *devVecPsi, *devVecAcceleratedXi, *devVecAcceleratedPsi,
	*devVecPrimalXi, *devVecPrimalPsi, *devVecDualXi, *devVecDualPsi, *devVecUpdateXi, *devVecUpdatePsi;
	real_t **devPtrVecX, **devPtrVecU, **devPtrVecV, **devPtrVecAcceleratedPsi, **devPtrVecAcceleratedXi,
	**devPtrVecPrimalPsi, **devPtrVecPrimalXi;
	real_t *devVecQ, *devVecR;
	real_t **devPtrVecQ, **devPtrVecR;
};
#endif /* SMPCONTROLLERCLASS_CUH_ */
