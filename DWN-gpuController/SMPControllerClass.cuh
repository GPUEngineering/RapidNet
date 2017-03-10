/*
 * SMPControllerClass.cuh
 *
 *  Created on: Mar 1, 2017
 *      Author: control
 */

#ifndef SMPCONTROLLERCLASS_CUH_
#define SMPCONTROLLERCLASS_CUH_
#include "DefinitionHeader.h"

class SMPCController{
public:
	SMPCController(Engine *myEngine);
	void updateLinearCostBeta();
	void solveStep();

	~SMPCController();
private:
	Engine* ptrMyEngine;
	real_t *devVecX, *devVecU, *devVecV, *devVecXi, *devVecPsi, *devVecAcceleratedXi, *devVecAcceleratedPsi,
	*devVecPrimalXi, *devVecPrimalPsi, *devVecDualXi, *devVecDualPsi, *devVecUpdateXi, *devVecUpdatePsi;
	real_t **devPtrVecX, **devPtrVecU, **devPtrVecV, **devPtrVecAcceleratedPsi, **devPtrVecAcceleratedXi,
	**devPtrVecPrimalPsi, **devPtrVecPrimalXi;
};

SMPCController::SMPCController(Engine *myEngine){
	cout << "Allocation of controller memory" << endl;
	ptrMyEngine = myEngine;
	uint_t nx = ptrMyEngine->ptrMyNetwork->NX;
	uint_t nu = ptrMyEngine->ptrMyNetwork->NU;
	uint_t nv = ptrMyEngine->ptrMyNetwork->NV;
	uint_t ns = ptrMyEngine->ptrMyForecastor->K;
	uint_t N = ptrMyEngine->ptrMyForecastor->N;
	uint_t nodes = ptrMyEngine->ptrMyForecastor->N_NODES;

	_CUDA( cudaMalloc((void**)&devVecX, nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecU, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecV, nv*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecAcceleratedXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecAcceleratedPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrimalXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrimalPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDualXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDualPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecUpdateXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecUpdatePsi, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrVecX, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecU, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecV, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecAcceleratedXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecAcceleratedPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalPsi, nodes*sizeof(real_t*)) );

	real_t** ptrVecX = new real_t*[nodes];
	real_t** ptrVecU = new real_t*[nodes];
	real_t** ptrVecV = new real_t*[nodes];
	real_t** ptrVecAcceleratedXi = new real_t*[nodes];
	real_t** ptrVecAcceleratedPsi = new real_t*[nodes];
	real_t** ptrVecPrimalXi = new real_t*[nodes];
	real_t** ptrVecPrimalPsi = new real_t*[nodes];

	for(int i = 0; i < nodes; i++){
		ptrVecX[i] = &devVecX[i*nx];
		ptrVecU[i] = &devVecU[i*nu];
		ptrVecV[i] = &devVecV[i*nv];
		ptrVecAcceleratedXi[i] = &devVecAcceleratedXi[2*i*nx];
		ptrVecAcceleratedPsi[i] = &devVecAcceleratedPsi[i*nu];
		ptrVecPrimalXi[i] = &devVecPrimalXi[2*i*nx];
		ptrVecPrimalPsi[i] = &devVecPrimalPsi[i*nu];
	}

	_CUDA( cudaMemset(devVecU, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecAcceleratedXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecAcceleratedPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecUpdateXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecUpdatePsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecDualXi, 0, 2*nx*nodes*sizeof(real_t)));
	_CUDA( cudaMemset(devVecDualPsi, 0, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecX, ptrVecX, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecU, ptrVecU, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecV, ptrVecV, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecAcceleratedXi, ptrVecAcceleratedXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecAcceleratedPsi, ptrVecAcceleratedPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalXi, ptrVecPrimalXi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalPsi, ptrVecPrimalPsi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );

	delete [] ptrVecX;
	delete [] ptrVecU;
	delete [] ptrVecV;
	delete [] ptrVecAcceleratedXi;
	delete [] ptrVecAcceleratedPsi;
	delete [] ptrVecPrimalXi;
	delete [] ptrVecPrimalPsi;
}

void SMPCController::updateLinearCostBeta(){
	//TODO
}

void SMPCController::solveStep(){
	/*
	real_t scale[2]={-0.5,1};

	_CUDA(cudaMemcpy(ptrMyEngine->devMatSigma, dev_linear_cost_b,NV*N_NODES*sizeof(real_t),cudaMemcpyDeviceToDevice));
	_CUDA(cudaMemcpy(dev_Effinet_SIGMA,dev_linear_cost_b,NV*N_NODES*sizeof(real_t),cudaMemcpyDeviceToDevice));

	for(int k=N-1;k>-1;k--){

		if(k<N-1){
			// sigma=sigma+r
			_CUBLAS(cublasSaxpy(handle,TREE_NODES_PER_STAGE[k]*NV,&alpha,dev_r,1,&dev_Effinet_SIGMA[TREE_NODES_PER_STAGE_CUMUL[k]*NV],1));
		}

		// v=Omega*sigma
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NV,&scale[0],(const float**)&dev_ptr_Effinet_OMEGA[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_Effinet_SIGMA[TREE_NODES_PER_STAGE_CUMUL[k]],NV,&beta,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

		if(k<N-1){
			// v=Theta*q+v
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NX,&alpha,(const float**)&dev_ptr_Effinet_THETA[TREE_NODES_PER_STAGE_CUMUL[k]],
					NV,(const float**)dev_ptr_q,NX,&alpha,&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));
		}
		// v=Psi*psi+v
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NU,&alpha,(const float**)&dev_ptr_Effinet_PSI[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_psi[TREE_NODES_PER_STAGE_CUMUL[k]],NU,&alpha,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));
		// v=Phi*xi+v
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,2*NX,&alpha,(const float**)&dev_ptr_Effinet_PHI[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,
				&dev_ptr_v[TREE_NODES_PER_STAGE_CUMUL[k]],NV,TREE_NODES_PER_STAGE[k]));

		// r=sigma
		_CUDA(cudaMemcpy(dev_r,&dev_Effinet_SIGMA[TREE_NODES_PER_STAGE_CUMUL[k]*NV],NV*TREE_NODES_PER_STAGE[k]*sizeof(real_t),cudaMemcpyDeviceToDevice));

		// r=D*xi+r
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,2*NX,&alpha,(const float**)&dev_ptr_Effinet_D[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,dev_ptr_r,NV,TREE_NODES_PER_STAGE[k]));

		// r=f*psi+r
		_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NU,&alpha,(const float**)&dev_ptr_Effinet_F[TREE_NODES_PER_STAGE_CUMUL[k]],
				NV,(const float**)&dev_ptr_accelerated_psi[TREE_NODES_PER_STAGE_CUMUL[k]],NU,&alpha,dev_ptr_r,NV,TREE_NODES_PER_STAGE[k]));

		if(k<N-1){
			// r=g*q+r
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,NV,1,NX,&alpha,(const float**)&dev_ptr_Effinet_G[TREE_NODES_PER_STAGE_CUMUL[k]],
					NV,(const float**)dev_ptr_q,NX,&alpha,dev_ptr_r,NV,TREE_NODES_PER_STAGE[k]));
		}

		if(k<N-1)
			// q=F'xi+q
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,NX,1,2*NX,&alpha,(const float**)&dev_ptr_F[TREE_NODES_PER_STAGE_CUMUL[k]],
					2*NX,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&alpha,dev_ptr_q,NX,TREE_NODES_PER_STAGE[k]));
		else
			// q=F'xi
			_CUBLAS(cublasSgemmBatched(handle,CUBLAS_OP_T,CUBLAS_OP_N,NX,1,2*NX,&alpha,(const float**)&dev_ptr_F[TREE_NODES_PER_STAGE_CUMUL[k]],
					2*NX,(const float**)&dev_ptr_accelerated_xi[TREE_NODES_PER_STAGE_CUMUL[k]],2*NX,&beta,dev_ptr_q,NX,TREE_NODES_PER_STAGE[k]));

		if(k>0){
			if((TREE_NODES_PER_STAGE[k]-TREE_NODES_PER_STAGE[k-1])>0){
				summation_children<real_t><<<TREE_NODES_PER_STAGE[k-1],NX>>>(dev_q,dev_temp_q,DEV_CONST_TREE_NODES_PER_STAGE,
						DEV_CONSTANT_TREE_NUM_CHILDREN,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,NX,k-1);
				summation_children<real_t><<<TREE_NODES_PER_STAGE[k-1],NV>>>(dev_r,dev_temp_r,DEV_CONST_TREE_NODES_PER_STAGE,
						DEV_CONSTANT_TREE_NUM_CHILDREN,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_N_CHILDREN_CUMUL,NV,k-1);
				_CUDA(cudaMemcpy(dev_r,dev_temp_r,TREE_NODES_PER_STAGE[k-1]*NV*sizeof(real_t),cudaMemcpyDeviceToDevice));
				_CUDA(cudaMemcpy(dev_q,dev_temp_q,TREE_NODES_PER_STAGE[k-1]*NX*sizeof(real_t),cudaMemcpyDeviceToDevice));
			}
		}
	}

	// Forward substitution
	_CUDA(cudaMemcpy(dev_u,dev_vhat,N_NODES*NU*sizeof(real_t),cudaMemcpyDeviceToDevice));

	for(int k=0;k<N;k++){
		if(k==0){
			// x=p, u=h
			_CUBLAS(cublasSaxpy_v2(handle,NV,&alpha,dev_prev_v,1,dev_v,1));
			_CUDA(cudaMemcpy(dev_x,dev_current_state,NX*sizeof(real_t),cudaMemcpyDeviceToDevice));
			// x=x+w
			_CUBLAS(cublasSaxpy_v2(handle,NX,&alpha,dev_disturb_w,1,dev_x,1));
			// u=Lv+\hat{u}
			_CUBLAS(cublasSgemv_v2(handle,CUBLAS_OP_N,NU,NV,&alpha,dev_L,NU,dev_v,1,&alpha,dev_u,1));
			// x=x+Bu
			_CUBLAS(cublasSgemv_v2(handle,CUBLAS_OP_N,NX,NU,&alpha,dev_B,NX,dev_u,1,&alpha,dev_x,1));

		}else{
			if((TREE_NODES_PER_STAGE[k]-TREE_NODES_PER_STAGE[k-1])>0){
				// v_k=v_{k-1}+v_k
				child_nodes_update<real_t><<<TREE_NODES_PER_STAGE[k],NV>>>(&dev_v[TREE_NODES_PER_STAGE_CUMUL[k-1]*NV],&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV]
				                                                                                                             ,DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_ANCESTOR,NV,k-1);
				// u_k=Lv_k+\hat{u}_k
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NU,TREE_NODES_PER_STAGE[k],NV,&alpha,dev_L,
						NU,&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV],NV,&alpha,&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU));
				// x=w
				_CUDA(cudaMemcpy(&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],&dev_disturb_w[TREE_NODES_PER_STAGE_CUMUL[k]*NX],TREE_NODES_PER_STAGE[k]*NX*
						sizeof(real_t),cudaMemcpyDeviceToDevice));
				// x=x+Bu
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NX,TREE_NODES_PER_STAGE[k],NU,&alpha,dev_B,NX,
						&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU,&alpha,&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],NX));
				// x_{k+1}=x_k
				child_nodes_update<real_t><<<TREE_NODES_PER_STAGE[k],NX>>>(&dev_x[TREE_NODES_PER_STAGE_CUMUL[k-1]*NX],
						&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],DEV_CONSTANT_TREE_NODES_PER_STAGE_CUMUL,DEV_CONSTANT_TREE_ANCESTOR,NX,k-1);
			}else{
				// v_k=v_{k-1}+v_k
				_CUBLAS(cublasSaxpy_v2(handle,NV*TREE_NODES_PER_STAGE[k],&alpha,&dev_v[TREE_NODES_PER_STAGE_CUMUL[k-1]*NV],1,
						&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV],1));
				// u_k=Lv_k+\hat{u}_k
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NU,TREE_NODES_PER_STAGE[k],NV,&alpha,dev_L,
						NU,&dev_v[TREE_NODES_PER_STAGE_CUMUL[k]*NV],NV,&alpha,&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU));
				// x_{k+1}=x_{k}
				//_CUBLAS(cublasSaxpy_v2(handle,NX*TREE_NODES_PER_STAGE[k],&alpha,&dev_x[TREE_NODES_PER_STAGE_CUMUL[k-1]*NX],1,
				//	&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],1));
				_CUDA(cudaMemcpy(&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],&dev_x[TREE_NODES_PER_STAGE_CUMUL[k-1]*NX],NX*TREE_NODES_PER_STAGE[k]*sizeof(real_t),
						cudaMemcpyDeviceToDevice));
				// x=x+w
				_CUBLAS(cublasSaxpy_v2(handle,NX*TREE_NODES_PER_STAGE[k],&alpha,&dev_disturb_w[TREE_NODES_PER_STAGE_CUMUL[k]*NX],1,
						&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],1));
				// x=x+Bu
				_CUBLAS(cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,NX,TREE_NODES_PER_STAGE[k],NU,&alpha,dev_B,
						NX,&dev_u[TREE_NODES_PER_STAGE_CUMUL[k]*NU],NU,&alpha,&dev_x[TREE_NODES_PER_STAGE_CUMUL[k]*NX],NX));
			}
		}
	}



	//free(ptr_x_c);
	//free(x_c);
	//free(y_c);
	*/
}

SMPCController::~SMPCController(){
	cout << "removing the memory of the controller" << endl;
	_CUDA( cudaFree(devVecX) );
	_CUDA( cudaFree(devVecU) );
	_CUDA( cudaFree(devVecV) );
	_CUDA( cudaFree(devVecXi) );
	_CUDA( cudaFree(devVecPsi) );
	_CUDA( cudaFree(devVecAcceleratedXi) );
	_CUDA( cudaFree(devVecAcceleratedPsi) );
	_CUDA( cudaFree(devVecPrimalXi) );
	_CUDA( cudaFree(devVecPrimalPsi) );
	_CUDA( cudaFree(devVecDualXi) );
	_CUDA( cudaFree(devVecDualPsi) );
	_CUDA( cudaFree(devVecUpdateXi) );
	_CUDA( cudaFree(devVecUpdatePsi) );

	_CUDA( cudaFree(devPtrVecX) );
	_CUDA( cudaFree(devPtrVecU) );
	_CUDA( cudaFree(devPtrVecV) );
	_CUDA( cudaFree(devPtrVecAcceleratedXi) );
	_CUDA( cudaFree(devPtrVecAcceleratedPsi) );
	_CUDA( cudaFree(devPtrVecPrimalXi) );
	_CUDA( cudaFree(devPtrVecPrimalPsi) );
}
#endif /* SMPCONTROLLERCLASS_CUH_ */
