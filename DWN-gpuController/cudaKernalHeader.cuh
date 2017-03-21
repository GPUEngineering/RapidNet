/*
 * cudaKernalHeader.cuh
 *
 *  Created on: Mar 14, 2017
 *      Author: Ajay Kumar Samparthirao
 */

#ifndef CUDAKERNALHEADER_CUH_
#define CUDAKERNALHEADER_CUH_

__global__ void preconditionSystem(real_t *matF, real_t *matG, real_t *dualDiagPrcnd, real_t *scaleVec,
		uint_t nx, uint_t nu){
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int currentThread = threadIdx.x;
	int currentBlock  = blockIdx.x;
	int matFDim = 2*nx*nx;
	int matGDim = nu*nu;
	real_t probValue = scaleVec[ currentBlock ];
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
		matG[ matGIdx ] = probValue * currentScaleValue;
	}else if( currentThread > nx & currentThread < nx + nu){
		int rowIdx = currentThread - nu;
		int matFIdx = currentBlock*matFDim + 2*nx*rowIdx + rowIdx;
		matF[matFIdx] = probValue * currentScaleValue;
	}else{
		int rowIdx = currentThread - nu;
		//int matFIdx = currentBlock*matFDim + nx*rowIdx + rowIdx - nx;
		int matFIdx = currentBlock*matFDim + 2*nx*(rowIdx - nx) + rowIdx;
		matF[matFIdx] = probValue * currentScaleValue;
	}
}

__global__ void calculateDiffUhat(real_t *devDeltaUhat, real_t *devUhat, real_t *prevUhat, uint_t *devTreeAncestor,
		uint_t nu, uint_t nodes){
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

__global__ void calculateZeta(real_t *devZeta, real_t *devDeltaUhat, real_t *devTreeProb, uint_t *devNumChildCuml,
		uint_t nu, uint_t numNonleafNodes, uint_t nodes){
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


__global__ void solveChildNodesUpdate(real_t *src, real_t *dst, uint_t *devTreeAncestor,uint_t iStageCumulNodes, uint_t dim){

	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int relativeNode = tid/dim;
	int dimElement = tid - relativeNode*dim;
	int previousAncestor = devTreeAncestor[iStageCumulNodes];
	int ancestor = devTreeAncestor[iStageCumulNodes + relativeNode];
	dst[tid] = src[(ancestor-previousAncestor)*dim + dimElement] + dst[tid];

}

__global__  void solveSumChildren(real_t *src, real_t *dst, uint_t *devTreeNumChildren, uint_t *devTreeNumChildCumul,
		  uint_t iStageCumulNodes, uint_t iStageNodes, uint_t iStage, uint_t dim){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int relativeNode = tid/dim;
	int relativeParentNode = tid - relativeNode*dim;
	int offset = 0;
	int numChild = 0;
	if( tid < iStageNodes*dim){
		if(iStage > 0){
			offset = (devTreeNumChildCumul[iStageCumulNodes+relativeNode-1] - devTreeNumChildCumul[iStageCumulNodes-1])*dim;
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


__global__ void kernalDualExtrapolationStep(real_t *vecDualW, real_t *vecPrevDual, real_t *vecCurrentDual,
		real_t alpha, int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if( tid < size){
		vecDualW[tid] = vecCurrentDual[tid] + alpha*(vecCurrentDual[tid] - vecPrevDual[tid]);
		vecCurrentDual[tid] = vecPrevDual[tid];
		//accelerated_dual[tid]=dual_k_1[tid]+alpha*(dual_k_1[tid]-dual_k[tid]);
		//dual_k[tid]=dual_k_1[tid];
	}
}




__global__ void projectionBox(real_t *vecX, real_t *lowerbound, real_t *upperbound, int dim, int offset, int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int idVecX = blockIdx.x*dim + threadIdx.x - offset;
	if( tid < size){
		if(vecX[idVecX] < lowerbound[tid])
			vecX[idVecX] = lowerbound[tid];
		else if( vecX[idVecX] > upperbound[tid])
			vecX[idVecX] = upperbound[tid];
	}
}

__global__ void shuffleVector(real_t *dst, real_t *src, int dimVec, int numVec, int numBlocks){
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

__global__ void additionVectorOffset(real_t *dst, real_t *src, real_t scale, int dim, int offset, int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int idVecX = blockIdx.x*dim + threadIdx.x - offset;
	if(tid < size){
		dst[idVecX] = dst[idVecX] +scale*src[idVecX];
	}
}

__global__ void kernalDualUpdate(real_t *vecDualY, real_t *vecDualW, real_t *vecHX, real_t *vecZ,
		float stepSize, int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size){
		vecDualY[tid] = vecDualW[tid] + stepSize*( vecHX[tid]-vecZ[tid] );
	}
}

__global__ void testGPUAdd(real_t *matF, real_t *matG, uint_t k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	matG[tid] = tid;
	matF[tid] = k*matG[tid];
}

__global__ void projectionControl(real_t *vecU, real_t *lowerbound, real_t *upperbound, int size){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < size){
		if(vecU[tid] < lowerbound[tid]){
			vecU[tid] = lowerbound[tid];
		}else if(vecU[tid] > upperbound[tid]){
			vecU[tid] = upperbound[tid];
		}
	}
}

__global__ void projection_state(real_t *x, real_t *lb, real_t *ub, real_t *safety_level, int size){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int tid_blck=threadIdx.x;
	int tid_box=blockIdx.x*NX;
	if(tid<size){
		if(tid_blck<NX){
			tid_box=tid_box+tid_blck;
			if(x[tid]<lb[tid_box]){
				x[tid]=lb[tid_box];
			}else if(x[tid]>ub[tid_box]){
				x[tid]=ub[tid_box];
			}
		}else{
			tid_box=tid_box+tid_blck-NX;
			if(x[tid]<safety_level[tid_box]){
				x[tid]=safety_level[tid_box];
			}
		}
	}
}


/*__global__ void preconditionSystem(real_t *matF, real_t *matG, real_t *dualDiagPrcnd, real_t *scaleVec,
		uint_t nx, uint_t nu);
__global__ void calculateDiffUhat(real_t *devDeltaUhat, real_t *devUhat, real_t *prevUhat, uint_t *devTreeAncestor,
		uint_t nu, uint_t nodes);
__global__ void calculateZeta(real_t *devZeta, real_t *devDeltaUhat, real_t *devTreeProb, uint_t *devNumChildCuml,
		uint_t nu, uint_t numNonleafNodes, uint_t nodes);
__global__  void solveSumChildren(real_t *src, real_t *dst, uint_t *devTreeNumChildren, uint_t *devTreeNumChildCumul,
		  uint_t iStageCumulNodes, uint_t iStageNodes, uint_t iStage, uint_t dim);
__global__ void solveChildNodesUpdate(real_t *src, real_t *dst, uint_t *devTreeAncestor,uint_t nextStageCumulNodes, uint_t dim);
__global__ void testGPUAdd(real_t *matF, real_t *matG, uint_t k);*/
#endif /* CUDAKERNALHEADER_CUH_ */
