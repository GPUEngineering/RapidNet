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


#include <cuda_device_runtime_api.h>
#include "cuda_runtime.h"

#ifndef CUDAKERNELHEADER_CUH_
#define CUDAKERNELHEADER_CUH_

static cudaEvent_t start;
static cudaEvent_t stop;
static short timer_running = 0;
static short tic_called = 0;

/**
 *
 * @param matF
 * @param matG
 * @param dualDiagPrcnd
 * @param scaleVec
 * @param nx
 * @param nu
 */
__global__ void preconditionSystem(
		real_t *matF,
		real_t *matG,
		real_t *dualDiagPrcnd,
		real_t *scaleVec,
		uint_t nx,
		uint_t nu);

/**
 *
 * @param devDeltaUhat
 * @param devUhat
 * @param prevUhat
 * @param devTreeAncestor
 * @param nu
 * @param nodes
 */
__global__ void calculateDiffUhat(
		real_t *devDeltaUhat,
		real_t *devUhat,
		real_t *prevUhat,
		uint_t *devTreeAncestor,
		uint_t nu,
		uint_t nodes);

/**
 *
 * @param devZeta
 * @param devDeltaUhat
 * @param devTreeProb
 * @param devNumChildCuml
 * @param nu
 * @param numNonleafNodes
 * @param nodes
 */
__global__ void calculateZeta(
			real_t *devZeta,
			real_t *devDeltaUhat,
			real_t *devTreeProb,
			uint_t *devNumChildCuml,
			uint_t nu,
			uint_t numNonleafNodes,
			uint_t nodes);


/**
 *
 * @param src
 * @param dst
 * @param devTreeAncestor
 * @param iStageCumulNodes
 * @param dim
 */
__global__ void solveChildNodesUpdate(
			real_t *src,
			real_t *dst,
			uint_t *devTreeAncestor,
			uint_t iStageCumulNodes,
			uint_t dim);

/**
 *
 * @param src
 * @param dst
 * @param devTreeNumChildren
 * @param devTreeNumChildCumul
 * @param iStageCumulNodes
 * @param iStageNodes
 * @param iStage
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
 * @todo change int into uint_t
 * @param vecDualW
 * @param vecPrevDual
 * @param vecCurrentDual
 * @param alpha
 * @param size
 */
__global__ void kernelDualExtrapolationStep(
			real_t *vecDualW,
			real_t *vecPrevDual,
			real_t *vecCurrentDual,
			real_t alpha,
			uint_t size);



/**
 * @todo change int into uint_t
 * @param vecX
 * @param lowerbound
 * @param upperbound
 * @param dim
 * @param offset
 * @param size
 */
__global__ void projectionBox(
			real_t *vecX,
			real_t *lowerbound,
			real_t *upperbound,
			uint_t dim,
			uint_t offset,
			uint_t size);


/**
 * @todo change int into uint_t
 * @param dst
 * @param src
 * @param dimVec
 * @param numVec
 * @param numBlocks
 */
__global__ void shuffleVector(
				real_t *dst,
				real_t *src,
				uint_t dimVec,
				uint_t numVec,
				uint_t numBlocks);
/**
 * @param dst
 * @param src
 * @param scale
 * @param dim
 * @param offset
 * @param size
 */
__global__ void additionVectorOffset(
			real_t *dst,
			real_t *src,
			real_t scale,
			uint_t dim,
			uint_t offset,
			uint_t size);

/**
 * @param vecDualY
 * @param vecDualW
 * @param vecHX
 * @param vecZ
 * @param stepSize
 * @param size
 */
__global__ void kernelDualUpdate(
			real_t *vecDualY,
			real_t *vecDualW,
			real_t *vecHX,
			real_t *vecZ,
			real_t stepSize,
			uint_t size);

/**
 * @param vecU
 * @param lowerbound
 * @param upperbound
 * @param size
 */
__global__ void projectionControl(
			real_t *vecU,
			real_t *lowerbound,
			real_t *upperbound,
			uint_t size);

/**
 *
 * @param  devXmax
 * @param  devXmin
 * @param  devXsafe
 * @param  matPrcndDiag
 * @param  probNode
 * @param  dim
 * @param  numBlock
 */
__global__ void preconditionConstraintX(
		real_t *devXmax,
		real_t *devXmin,
		real_t *devXsafe,
		real_t *matPrcndDiag,
		real_t *probNode,
		uint_t dim,
		uint_t numBlock);

/**
 *
 * @param  devXmax
 * @param  devXmin
 * @param  devXsafe
 * @param  matPrcndDiag
 * @param  dim
 * @param  numBlock
 */
__global__ void preconditionConstraintU(
		real_t *devUmax,
		real_t *devUmin,
		real_t *matPrcndDiag,
		real_t *probNode,
		uint_t dim,
		uint_t numBlock);

/**
 * @param devVecU
 * @param devProbNode
 * @param dim
 * @param numBlock
 */
__global__ void scaleVecProbalitity(
		real_t* devVecU,
		real_t* devProbNode,
		uint_t dim,
		uint_t numBlock);

/**
 * Sets up the timer.
 *
 * Must be called before any invocation to
 * tic() or toc(), preferrably at the beginning of your
 * application.
 */
void startTicToc();

/**
 * Starts the timer.
 *
 * Use `toc()` to get the elapsed time; `tic()` must
 * be called before a `toc()`.
 */
void tic();

/**
 * Returns the elapsed time between its invocation
 * and a previous invocation of `toc()`. Returns `-1`
 * and prints a warning message if `toc()` was not
 * previously called. Returns `-2` and prints and error
 * message if `start_tictoc()` has not been called.
 *
 * @return Elapsed time between `tic()` and `toc()` in milliseconds
 * with a resolution of `0.5` microseconds.
 */
float toc();

/**
 * This function should be called when the
 * time will not be being used any more. It destroys
 * the events used to time CUDA kernels. If the timer
 * is not running, this function does nothing and
 * prints a warning message.
 */
void stopTicToc();

#endif /* CUDAKERNELHEADER_CUH_ */
