/*
 * SMPControllerClass.cuh
 *
 *  Created on: Mar 1, 2017
 *  Author: Ajay K. Sampathirao, Pantelis Sopasakis
 */

#ifndef SMPCONTROLLERCLASS_CUH_
#define SMPCONTROLLERCLASS_CUH_

#include "Configuration.h"
#include "Engine.cuh"
//#include "cudaKernelHeader.cuh"

class SmpcController {
public:
	/**
	 * Construct a new Controller with a given engine.
	 */
	SmpcController(Engine *myEngine);
	/**
	 *
	 */
	void dualExtrapolationStep(real_t lambda);
	/**
	 *
	 */
	/*TODO solveStep should be private - nobody will need to compute a dual
	       gradient outside this class. This is an SMPC controller and its main
				 purpose is to compute control actions. */
	void solveStep();
	/**
	 *
	 */
	void proximalFunG();
	/**
	 *
	 */
	void dualUpdate();
	/**
	 *
	 */
	void algorithmApg();
	/**
	 *
	 */
	void controllerSmpc();
	/**
	 *
	 */
	~SmpcController();
private:
	/**
	 * Pointer to an Engine object.
	 * The Engine is responsible for the factor step.
	 */
	Engine* ptrMyEngine;
	/**
	 *
	 */
	real_t *devVecX;
	/**
	 *
	 */
	real_t *devVecU;
	/**
	 *
	 */
	real_t *devVecV;
	/**
	 *
	 */
	real_t *devVecXi;
	/**
	 *
	 */
	real_t *devVecPsi;
	/**
	 *
	 */
	real_t *devVecAcceleratedXi;
	/**
	 *
	 */
	real_t *devVecAcceleratedPsi;
	/**
	 *
	 */
	real_t *devVecPrimalXi;
	/**
	 *
	 */
	real_t *devVecPrimalPsi;
	/**
	 *
	 */
	real_t *devVecDualXi;
	/**
	 *
	 */
	real_t *devVecDualPsi;
	/**
	 *
	 */
	real_t *devVecUpdateXi;
	/**
	 *
	 */
	real_t *devVecUpdatePsi;
	/**
	 *
	 */
	real_t **devPtrVecX;
	/**
	 *
	 */
	real_t **devPtrVecU;
	/**
	 *
	 */
	real_t **devPtrVecV;
	/**
	 *
	 */
	real_t **devPtrVecAcceleratedPsi;
	/**
	 *
	 */
	real_t **devPtrVecAcceleratedXi;
	/**
	 *
	 */
	real_t **devPtrVecPrimalPsi;
	/**
	 *
	 */
	real_t **devPtrVecPrimalXi;
	/**
	 *
	 */
	real_t *devPrimalInfeasibilty;
	/**
	 *
	 */
	real_t *devVecQ;
	/**
	 *
	 */
	real_t *devVecR;
	/**
	 *
	 */
	real_t **devPtrVecQ;
	/**
	 *
	 */
	real_t **devPtrVecR;
	/**
	 *
	 */
	real_t stepSize;
	/**
	 * Maximum number of iterations
	 * Default: 500
	 */
	uint_t MAX_ITERATIONS;
};



#endif /* SMPCONTROLLERCLASS_CUH_ */
