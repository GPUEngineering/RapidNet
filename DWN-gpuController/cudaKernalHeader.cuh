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


__global__ void summation_children(real_t *x, real_t *y, uint_t* DEV_CONST_TREE_NODES_PER_STAGE
		,uint_t* DEV_CONSTANT_TREE_NUM_CHILDREN,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL
		,uint_t* DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,uint_t dim,uint_t stage){

	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int relative_node=tid/dim;
	int relative_parent_node=tid-relative_node*dim;
	int no_nodes=DEV_CONST_TREE_NODES_PER_STAGE[stage];
	int node_before=0;
	int off_set=0;
	if(tid<no_nodes*dim){
		if(stage>0){
			node_before=DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL[stage];
			off_set=(DEV_CONSTANT_TREE_N_CHILDREN_CUMUL[node_before+relative_node-1]-DEV_CONSTANT_TREE_N_CHILDREN_CUMUL[node_before-1])*dim;
		}
		int no_child=DEV_CONSTANT_TREE_NUM_CHILDREN[node_before+relative_node];
		if(no_child>1){
			for(int i=0;i<no_child-1;i++){
				if(i==0)
					y[tid]=x[off_set+relative_parent_node]+x[off_set+relative_parent_node+dim];
				if(i>0)
					y[tid]=y[tid]+x[off_set+relative_parent_node+(i+1)*dim];
			}
		}else{
			//printf("%d %d %d %f \n",no_child,dim,relative_node,relative_parent_node,x[tid]);
			y[tid]=x[off_set+relative_parent_node];
		}
	}
}

template<typename T>__global__ void child_nodes_update(T *x,T *y,uint_t* DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,
		uint_t* DEV_CONSTANT_TREE_ANCESTOR,int dim,int stage){

	int tid =blockDim.x*blockIdx.x+threadIdx.x;
	int relative_node=tid/dim;
	int dim_element=tid-relative_node*dim;
	int node_before=DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL[stage+1];
	int pre_ancestor=DEV_CONSTANT_TREE_ANCESTOR[node_before];
	int ancestor=DEV_CONSTANT_TREE_ANCESTOR[node_before+relative_node];
	y[tid]=x[(ancestor-pre_ancestor)*dim+dim_element]+y[tid];
	//y[(ancestor-1)*dim+tid]=y[(ancestor-1)*dim+tid]+x[bid];

}

__global__ void testGPUAdd(real_t *matF, real_t *matG, uint_t k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	matG[tid] = tid;
	matF[tid] = k*matG[tid];
}



__global__ void preconditionSystem(real_t *matF, real_t *matG, real_t *dualDiagPrcnd, real_t *scaleVec,
		uint_t nx, uint_t nu);
__global__ void calculateDiffUhat(real_t *devDeltaUhat, real_t *devUhat, real_t *prevUhat, uint_t *devTreeAncestor,
		uint_t nu, uint_t nodes);
__global__ void calculateZeta(real_t *devZeta, real_t *devDeltaUhat, real_t *devTreeProb, uint_t *devNumChildCuml,
		uint_t nu, uint_t numNonleafNodes, uint_t nodes);
__global__ void testGPUAdd(real_t *matF, real_t *matG, uint_t k);
#endif /* CUDAKERNALHEADER_CUH_ */
