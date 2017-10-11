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


#include "Configuration.h"
#include "Utilities.cuh"
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
		uint_t nu){
	uint_t currentThread = threadIdx.x;
	uint_t currentBlock  = blockIdx.x;
	uint_t matFDim = 2*nx*nx;
	uint_t matGDim = nu*nu;
	real_t probSqrtValue = sqrt(scaleVec[ currentBlock ]);
	real_t currentScaleValue = dualDiagPrcnd[ currentThread ];
	if( currentThread < nu ){
		uint_t matGIdx = currentBlock*matGDim + nu*currentThread + currentThread;
		matG[ matGIdx ] = probSqrtValue * currentScaleValue;
	}else if( currentThread > nx & currentThread < nx + nu){
		uint_t rowIdx = currentThread - nu;
		uint_t matFIdx = currentBlock*matFDim + 2*nx*rowIdx + rowIdx;
		matF[matFIdx] = probSqrtValue * currentScaleValue;
	}else{
		uint_t rowIdx = currentThread - nu;
		uint_t matFIdx = currentBlock*matFDim + 2*nx*(rowIdx - nx) + rowIdx;
		matF[matFIdx] = probSqrtValue * currentScaleValue;
	}
}

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
		uint_t nodes){
	uint_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint_t currentBlock = blockIdx.x;
	uint_t currentThread = threadIdx.x;
	uint_t ancestorIdx = 0;
	if ( currentThread < nu & currentBlock < nodes){
		if (currentBlock == 0){
			devDeltaUhat[currentThread] = devUhat[currentThread] - prevUhat[currentThread];
		}else{
			ancestorIdx = (devTreeAncestor[currentBlock]-1)*nu + currentThread;
			devDeltaUhat[tid] = devUhat[tid] - devUhat[ancestorIdx];
		}
	}
}

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
			uint_t nodes){
	uint_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint_t currentBlock = blockIdx.x;
	uint_t currentThread = threadIdx.x;
	if ( currentThread < nu & currentBlock < nodes){
		devZeta[tid] = devTreeProb[currentBlock] * devDeltaUhat[tid];
		if(currentBlock < numNonleafNodes){
			if(currentBlock == 0){
				uint_t nChild = devNumChildCuml[currentBlock];
				uint_t childIdx;
				for (uint_t iChild = 0; iChild < nChild; iChild++){
					childIdx = (iChild + 1);
					devZeta[tid] = devZeta[tid] - devTreeProb[childIdx]*devDeltaUhat[childIdx*nu + currentThread];
				}
			}else{
				uint_t nChild = devNumChildCuml[currentBlock] - devNumChildCuml[currentBlock - 1];
				uint_t childIdx;
				for (uint_t iChild = 0; iChild < nChild; iChild++){
					childIdx = (devNumChildCuml[currentBlock - 1] + iChild + 1);
					devZeta[tid] = devZeta[tid] - devTreeProb[childIdx]*devDeltaUhat[childIdx*nu + currentThread];
				}
			}
		}
	}
}


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
			uint_t dim){
	uint_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	uint_t relativeNode = tid/dim;
	uint_t dimElement = tid - relativeNode*dim;
	uint_t previousAncestor = devTreeAncestor[iStageCumulNodes];
	uint_t ancestor = devTreeAncestor[iStageCumulNodes + relativeNode];
	dst[tid] = src[(ancestor-previousAncestor)*dim + dimElement] + dst[tid];

}

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
			uint_t dim){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint_t relativeNode = tid/dim;
	uint_t relativeParentNode = tid - relativeNode*dim;
	uint_t offset = 0;
	uint_t numChild = 0;
	if( tid < iStageNodes*dim){
		if(iStage > 0){
			offset = (devTreeNumChildCumul[iStageCumulNodes+relativeNode-1]
				- devTreeNumChildCumul[iStageCumulNodes-1])*dim;
			numChild = devTreeNumChildren[iStageCumulNodes + relativeNode];
		}else{
			numChild = devTreeNumChildren[relativeNode];
		}
		if( numChild > 1){
			for(uint_t iChild = 0; iChild < numChild-1; iChild++){
				if(iChild == 0)
					dst[tid] = src[offset + relativeParentNode] + src[offset + relativeParentNode + dim];
				if(iChild > 0)
					dst[tid] = dst[tid] + src[offset + relativeParentNode + (iChild+1)*dim];
			}
		}else{
			dst[tid] = src[offset + relativeParentNode];
		}
	}
}

/**
 *
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
			uint_t size){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	if( tid < size){
		vecDualW[tid] = vecCurrentDual[tid] + alpha*(vecCurrentDual[tid] - vecPrevDual[tid]);
		vecCurrentDual[tid] = vecPrevDual[tid];
		//accelerated_dual[tid]=dual_k_1[tid]+alpha*(dual_k_1[tid]-dual_k[tid]);
		//dual_k[tid]=dual_k_1[tid];
	}
}



/**
 *
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
			uint_t size){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint_t idVecX = blockIdx.x*dim + threadIdx.x + offset;
	if(idVecX < 0)
		printf(" %d %d %d ", blockIdx.x, threadIdx.x, idVecX);
	if( tid < size){
		if(vecX[idVecX] < lowerbound[tid])
			vecX[idVecX] = lowerbound[tid];
		else if( vecX[idVecX] > upperbound[tid])
			vecX[idVecX] = upperbound[tid];
	}
}


/**
 *
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
				uint_t numBlocks){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint_t dimBin = dimVec*numVec;
	if(tid < numBlocks*dimBin){
		uint_t binId = tid/dimBin;
		uint_t offsetBin = tid - binId*dimBin;
		uint_t vecId = offsetBin/dimVec;
		uint_t elemId = offsetBin - vecId*dimVec;
		uint_t shuffleVecId = vecId*(numBlocks*dimVec) + binId*dimVec + elemId;
		dst[shuffleVecId] = src[tid];
	}
}

/**
 *
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
			uint_t size){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint_t idVecX = blockIdx.x*dim + threadIdx.x + offset;
	if(tid < size){
		dst[idVecX] = dst[idVecX] +scale*src[idVecX];
	}
}

/**
 *
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
			uint_t size){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size){
		vecDualY[tid] = vecDualW[tid] + stepSize*( vecHX[tid]-vecZ[tid] );
	}
}


/**
 *
 * @param vecU
 * @param lowerbound
 * @param upperbound
 * @param size
 */
__global__ void projectionControl(
			real_t *vecU,
			real_t *lowerbound,
			real_t *upperbound,
			uint_t size){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size){
		if(vecU[tid] < lowerbound[tid]){
			vecU[tid] = lowerbound[tid];
		}else if(vecU[tid] > upperbound[tid]){
			vecU[tid] = upperbound[tid];
		}
	}
}

/**
 *
 * @param  devXmax
 * @param  devXmin
 * @param  devXsafe
 * @param  matPrcndDiag
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
		uint_t numBlock){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint_t currentThread = threadIdx.x;
	uint_t currentBlock = blockIdx.x;
	if( currentBlock < numBlock & currentThread < dim){
		real_t scaleX = sqrt(probNode[currentBlock])*matPrcndDiag[currentThread];
		real_t scaleXsafe = sqrt(probNode[currentBlock])*matPrcndDiag[currentThread + dim];
		devXmax[tid] = scaleX * devXmax[tid];
		devXmin[tid] = scaleX * devXmin[tid];
		devXsafe[tid] = scaleXsafe * devXsafe[tid];
	}
}


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
		uint_t numBlock){
	uint_t tid = blockIdx.x*blockDim.x + threadIdx.x;
	uint_t currentThread = threadIdx.x;
	uint_t currentBlock = blockIdx.x;
	if( currentBlock < numBlock & currentThread < dim){
		real_t scaleU = sqrt(probNode[currentBlock])*matPrcndDiag[currentThread];
		devUmax[tid] = scaleU * devUmax[tid];
		devUmin[tid] = scaleU * devUmin[tid];
	}

}



void startTicToc() {
	_CUDA(cudaEventCreate(&start));
	_CUDA(cudaEventCreate(&stop));
	timer_running = 1;
}

void tic() {
	if (timer_running) {
		_CUDA(cudaEventRecord(start, 0));
		tic_called = 1;
	} else {
		cout << "WARNING: tic() called without a timer running!\n";
	}
}

float toc() {
	float elapsed_time;
	if (tic_called == 0) {
		cout << "WARNING: toc() called without a previous tic()!\n";
		return -1;
	}
	if (timer_running == 1) {
		_CUDA(cudaEventRecord(stop, 0));
		_CUDA(cudaEventSynchronize(stop));
		_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
		tic_called = 0;
		return elapsed_time;
	} else {
		cout << "WARNING: toc() called without a timer running!\n";
		return -2;
	}

}

void stopTicToc()
{
	if (timer_running == 1){
		_CUDA(cudaEventDestroy(start));
		_CUDA(cudaEventDestroy(stop));
		timer_running = 0;
	} else{
		cout << "WARNING: stop_tictoc() called without a timer running!\n";
	}
}
