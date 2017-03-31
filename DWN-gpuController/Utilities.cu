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
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int currentThread = threadIdx.x;
	int currentBlock  = blockIdx.x;
	int matFDim = 2*nx*nx;
	int matGDim = nu*nu;
	real_t probSqrtValue = sqrt(scaleVec[ currentBlock ]);
	real_t currentScaleValue = dualDiagPrcnd[ currentThread ];
	/*if( currentThread < nx){
		uint_t matFIdx = currentBlock * matFDim + (nx*currentThread + currentThread);
		matF[ matFIdx ] = scaleValue * probValue;
	}else if( currentThread > nx & currentThread < 2*nx){
		uint_t matFIdx = currentBlock * matFDim + (nx*currentThread + currentThread - nx);
		matF[ matFIdx] = scaleValue * probValue;
	}else{
		uint_t rowIdx = currentThread -2*nx;
		uint_t matGIdx = currentBlock * matGDim + (nu*rowIdx + rowIdx );
		matG[ matGIdx ] = scaleValue * probValue;
	}*/
	if( currentThread < nu ){
		int matGIdx = currentBlock*matGDim + nu*currentThread + currentThread;
		matG[ matGIdx ] = probSqrtValue * currentScaleValue;
	}else if( currentThread > nx & currentThread < nx + nu){
		int rowIdx = currentThread - nu;
		int matFIdx = currentBlock*matFDim + 2*nx*rowIdx + rowIdx;
		matF[matFIdx] = probSqrtValue * currentScaleValue;
	}else{
		int rowIdx = currentThread - nu;
		//int matFIdx = currentBlock*matFDim + nx*rowIdx + rowIdx - nx;
		int matFIdx = currentBlock*matFDim + 2*nx*(rowIdx - nx) + rowIdx;
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
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int currentBlock = blockIdx.x;
	int currentThread = threadIdx.x;
	int ancestorIdx = 0;
	if ( currentThread < nu & currentBlock < nodes){
		if (currentBlock == 0){
			devDeltaUhat[currentThread] = devUhat[currentThread] - prevUhat[currentThread];
		}else{
			ancestorIdx = devTreeAncestor[currentBlock]*nu + currentThread;
			devDeltaUhat[tid] = devUhat[tid] - devUhat[currentThread];
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
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int currentBlock = blockIdx.x;
	int currentThread = threadIdx.x;
	if ( currentThread < nu & currentBlock < nodes){
		devZeta[tid] = devTreeProb[currentBlock] * devDeltaUhat[tid];
		if(currentBlock < numNonleafNodes){
			if(currentBlock == 0){
				int nChild = devNumChildCuml[currentBlock];
				int childIdx;
				for (int iChild = 0; iChild < nChild; iChild++){
					childIdx = (iChild + 1);
					devZeta[tid] = devZeta[tid] - devTreeProb[childIdx]*devDeltaUhat[childIdx*nu + currentThread];
				}
			}else{
				int nChild = devNumChildCuml[currentBlock] - devNumChildCuml[currentBlock - 1];
				int childIdx;
				for (int iChild = 0; iChild < nChild; iChild++){
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
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int relativeNode = tid/dim;
	int dimElement = tid - relativeNode*dim;
	int previousAncestor = devTreeAncestor[iStageCumulNodes];
	int ancestor = devTreeAncestor[iStageCumulNodes + relativeNode];
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
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int relativeNode = tid/dim;
	int relativeParentNode = tid - relativeNode*dim;
	int offset = 0;
	int numChild = 0;
	if( tid < iStageNodes*dim){
		if(iStage > 0){
			offset = (devTreeNumChildCumul[iStageCumulNodes+relativeNode-1]
				- devTreeNumChildCumul[iStageCumulNodes-1])*dim;
			numChild = devTreeNumChildren[iStageCumulNodes + relativeNode];
		}else{
			numChild = devTreeNumChildren[relativeNode];
		}
		if( numChild > 1){
			for(int iChild = 0; iChild < numChild-1; iChild++){
				if(iChild == 0)
					dst[tid] = src[offset + relativeParentNode] + src[offset + relativeParentNode + dim];
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
/*TODO int size --> uint_t size */
__global__ void kernelDualExtrapolationStep(
			real_t *vecDualW,
			real_t *vecPrevDual,
			real_t *vecCurrentDual,
			real_t alpha,
			int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
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
/*TODO int dim --> uint_t dim */
/*TODO int offset --> uint_t offset */
/*TODO int size --> uint_t size */
__global__ void projectionBox(
			real_t *vecX,
			real_t *lowerbound,
			real_t *upperbound,
			int dim,
			int offset,
			int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int idVecX = blockIdx.x*dim + threadIdx.x - offset;
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
/*TODO int dimVec --> uint_t dimVec */
/*TODO int numVec --> uint_t numVec */
/*TODO int numBlocks --> uint_t numBlocks */
__global__ void shuffleVector(
				real_t *dst,
				real_t *src,
				int dimVec,
				int numVec,
				int numBlocks){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int dimBin = dimVec*numVec;
	if(tid < numBlocks*dimBin){
		int binId = tid/dimBin;
		int offsetBin = tid - binId*dimBin;
		int vecId = offsetBin/dimVec;
		int elemId = offsetBin - vecId*dimVec;
		int shuffleVecId = vecId*(numBlocks*dimVec) + binId*dimVec + elemId;
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
 /*TODO int dim --> uint_t dim */
 /*TODO int offset --> uint_t offset */
 /*TODO int size --> uint_t size */
__global__ void additionVectorOffset(
			real_t *dst,
			real_t *src,
			real_t scale,
			int dim,
			int offset,
			int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int idVecX = blockIdx.x*dim + threadIdx.x - offset;
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
/*TODO int dim --> uint_t dim */
/*TODO int stepSize --> uint_t stepSize */
/*TODO int size --> uint_t size */
__global__ void kernelDualUpdate(
			real_t *vecDualY,
			real_t *vecDualW,
			real_t *vecHX,
			real_t *vecZ,
			float stepSize,
			int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size){
		vecDualY[tid] = vecDualW[tid] + stepSize*( vecHX[tid]-vecZ[tid] );
	}
}

/*TODO remove this test - there shouldn't be any tests here */
__global__ void testGPUAdd(
			real_t *matF,
			real_t *matG,
			uint_t k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	matG[tid] = tid;
	matF[tid] = k*matG[tid];
}

/**
 *
 * @param vecU
 * @param lowerbound
 * @param upperbound
 * @param size
 */
 /*TODO int size --> uint_t size */
__global__ void projectionControl(
			real_t *vecU,
			real_t *lowerbound,
			real_t *upperbound,
			int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size){
		if(vecU[tid] < lowerbound[tid]){
			vecU[tid] = lowerbound[tid];
		}else if(vecU[tid] > upperbound[tid]){
			vecU[tid] = upperbound[tid];
		}
	}
}
