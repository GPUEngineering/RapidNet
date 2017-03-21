/*
 * SMPControllerClass.cuh
 *
 *  Created on: Mar 1, 2017
 *  Author: Ajay K. Sampathirao, Pantelis Sopasakis
 */

#ifndef SMPCONTROLLERCLASS_CUH_
#define SMPCONTROLLERCLASS_CUH_

#include "DefinitionHeader.h"
#include "EngineHeader.cuh"

class SMPCController {
public:
	/**
	 * Construct a new Controller with a given engine.
	 */
	SMPCController(Engine *myEngine);
	/**
	 *
	 */
	void dualExtrapolationStep(real_t lambda);
	/**
	 *
	 */
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
	~SMPCController();
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
	uint_t MAX_ITERATIONS  = 500;
};

/**
 * Kernel function `solveSumChildren`
 *
 * @param src 
 * @param dst
 * @param devTreeNumChildren
 * @param devTreeNumChildCumul
 * @param iStageCumulNodes
 * @param iStageNodes
 * @param dim
 */
__global__  void solveSumChildren(
		real_t *src, 
		real_t *dst, 
		uint_t *devTreeNumChildren, 
		uint_t *devTreeNumChildCumul,
		uint_t iStageCumulNodes, 
		uint_t iStageNodes, 
		uint_t iStage, 
		uint_t dim);

/**
 * Kernel function `solveChildNodesUpdate`
 *
 * @param src 
 * @param dst
 * @param devTreeAncestor
 * @param nextStageCumulNodes
 * @param dim
 */
__global__ void solveChildNodesUpdate(
		real_t *src, 
		real_t *dst, 
		uint_t *devTreeAncestor,
		uint_t nextStageCumulNodes, 
		uint_t dim);


#endif /* SMPCONTROLLERCLASS_CUH_ */
