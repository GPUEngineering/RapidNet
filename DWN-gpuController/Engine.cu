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
#include "cublas_v2.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"
#include "Configuration.h"
#include "Engine.cuh"

Engine::Engine(DwnNetwork *myNetwork, ScenarioTree *myScenarioTree, SmpcConfiguration *mySmpcConfig){
	ptrMyNetwork = myNetwork;
	ptrMyScenarioTree = myScenarioTree;
	ptrMySmpcConfig = mySmpcConfig;
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	allocateSystemDevice();
	allocateScenarioTreeDevice();
	cublasCreate(&handle);
	priceUncertaintyFlag = true;
	demandUncertaintyFlag = true;

	_CUDA( cudaMalloc((void**)&devMatPhi, 2*nodes*nv*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatPsi, nodes*nu*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatTheta, nodes*nx*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatOmega, nodes*nv*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatSigma, nodes*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatD, 2*nodes*nv*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatF, nodes*nv*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatG, nodes*nv*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecUhat, nodes*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecBeta, nodes*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecE, nodes*nx*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrMatPhi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatTheta, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatOmega, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatSigma, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatD, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatF, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatG, nodes*sizeof(real_t*)) );

	real_t** ptrMatPhi = new real_t*[nodes];
	real_t** ptrMatPsi = new real_t*[nodes];
	real_t** ptrMatTheta = new real_t*[nodes];
	real_t** ptrMatSigma = new real_t*[nodes];
	real_t** ptrMatOmega = new real_t*[nodes];
	real_t** ptrMatD = new real_t*[nodes];
	real_t** ptrMatF = new real_t*[nodes];
	real_t** ptrMatG = new real_t*[nodes];

	for(int i = 0;i < nodes; i++){
		ptrMatPhi[i] = &devMatPhi[2*i*nv*nx];
		ptrMatPsi[i] = &devMatPsi[i*nv*nu];
		ptrMatTheta[i] = &devMatTheta[i*nx*nv];
		ptrMatOmega[i] = &devMatOmega[i*nv*nv];
		ptrMatSigma[i] = &devMatSigma[i*nv];
		ptrMatD[i] = &devMatD[2*i*nv*nx];
		ptrMatF[i] = &devMatF[i*nv*nu];
		ptrMatG[i] = &devMatG[i*nv*nx];
	}

	_CUDA( cudaMemcpy(devPtrMatPhi, ptrMatPhi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatPsi, ptrMatPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatTheta, ptrMatTheta, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatSigma, ptrMatSigma, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatOmega, ptrMatOmega, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatD, ptrMatD, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatF, ptrMatF, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatG, ptrMatG, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );

	delete [] ptrMatPhi;
	delete [] ptrMatPsi;
	delete [] ptrMatTheta;
	delete [] ptrMatSigma;
	delete [] ptrMatOmega;
	delete [] ptrMatD;
	delete [] ptrMatF;
	delete [] ptrMatG;
	ptrMatPhi = NULL;
	ptrMatPsi = NULL;
	ptrMatTheta = NULL;
	ptrMatSigma = NULL;
	ptrMatD = NULL;
	ptrMatF = NULL;
	ptrMatG = NULL;
}

Engine::Engine(SmpcConfiguration *smpcConfig){
	ptrMySmpcConfig = smpcConfig;
	string pathToNetwork = ptrMySmpcConfig->getPathToNetwork();
	string pathToScenarioTree = ptrMySmpcConfig->getPathToScenarioTree();
	ptrMyNetwork = new DwnNetwork( pathToNetwork );
	ptrMyScenarioTree = new ScenarioTree( pathToScenarioTree );

	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	allocateSystemDevice();
	allocateScenarioTreeDevice();
	cublasCreate(&handle);
	priceUncertaintyFlag = true;
	demandUncertaintyFlag = true;

	_CUDA( cudaMalloc((void**)&devMatPhi, 2*nodes*nv*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatPsi, nodes*nu*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatTheta, nodes*nx*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatOmega, nodes*nv*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatSigma, nodes*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatD, 2*nodes*nv*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatF, nodes*nv*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatG, nodes*nv*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecUhat, nodes*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecBeta, nodes*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecE, nodes*nx*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrMatPhi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatTheta, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatOmega, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatSigma, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatD, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatF, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatG, nodes*sizeof(real_t*)) );

	real_t** ptrMatPhi = new real_t*[nodes];
	real_t** ptrMatPsi = new real_t*[nodes];
	real_t** ptrMatTheta = new real_t*[nodes];
	real_t** ptrMatSigma = new real_t*[nodes];
	real_t** ptrMatOmega = new real_t*[nodes];
	real_t** ptrMatD = new real_t*[nodes];
	real_t** ptrMatF = new real_t*[nodes];
	real_t** ptrMatG = new real_t*[nodes];

	for(int i = 0;i < nodes; i++){
		ptrMatPhi[i] = &devMatPhi[2*i*nv*nx];
		ptrMatPsi[i] = &devMatPsi[i*nv*nu];
		ptrMatTheta[i] = &devMatTheta[i*nx*nv];
		ptrMatOmega[i] = &devMatOmega[i*nv*nv];
		ptrMatSigma[i] = &devMatSigma[i*nv];
		ptrMatD[i] = &devMatD[2*i*nv*nx];
		ptrMatF[i] = &devMatF[i*nv*nu];
		ptrMatG[i] = &devMatG[i*nv*nx];
	}

	_CUDA( cudaMemcpy(devPtrMatPhi, ptrMatPhi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatPsi, ptrMatPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatTheta, ptrMatTheta, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatSigma, ptrMatSigma, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatOmega, ptrMatOmega, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatD, ptrMatD, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatF, ptrMatF, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatG, ptrMatG, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );


	delete [] ptrMatPhi;
	delete [] ptrMatPsi;
	delete [] ptrMatTheta;
	delete [] ptrMatSigma;
	delete [] ptrMatOmega;
	delete [] ptrMatD;
	delete [] ptrMatF;
	delete [] ptrMatG;
	ptrMatPhi = NULL;
	ptrMatPsi = NULL;
	ptrMatTheta = NULL;
	ptrMatSigma = NULL;
	ptrMatD = NULL;
	ptrMatF = NULL;
	ptrMatG = NULL;
}

void Engine::allocateScenarioTreeDevice(){
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t N = ptrMyScenarioTree->getPredHorizon();
	uint_t K = ptrMyScenarioTree->getNumScenarios();
	uint_t ND = ptrMyNetwork->getNumDemands();
	uint_t NU = ptrMyNetwork->getNumControls();
	uint_t nNumNonLeafNodes = ptrMyScenarioTree->getNumNonleafNodes();
	_CUDA( cudaMalloc((void**)&devTreeStages, nodes*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeNodesPerStage, (N + 1)*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeNodesPerStageCumul, (N + 2)*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeLeaves, K*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeNumChildren, nNumNonLeafNodes*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeAncestor, nodes*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeNumChildrenCumul, nodes*sizeof(uint_t)) );
	_CUDA( cudaMalloc((void**)&devTreeProb, nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devTreeErrorDemand, nodes*ND*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devTreeErrorPrices, nodes*NU*sizeof(real_t)) );
	//_CUDA( cudaMalloc((void**)&devForecastValue, N*ND*sizeof(real_t)) );
}

void Engine::initialiseScenarioTreeDevice(){
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t N = ptrMyScenarioTree->getPredHorizon();
	uint_t K = ptrMyScenarioTree->getNumScenarios();
	uint_t nd = ptrMyNetwork->getNumDemands();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t numNonLeafNodes = ptrMyScenarioTree->getNumNonleafNodes();
	_CUDA( cudaMemcpy(devTreeStages, ptrMyScenarioTree->getStageNodes(), nodes*sizeof(uint_t), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeNodesPerStage, ptrMyScenarioTree->getNodesPerStage(), (N + 1)*sizeof(uint_t), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeNodesPerStageCumul, ptrMyScenarioTree->getNodesPerStageCumul(), (N + 2)*sizeof(uint_t),
			cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeLeaves, ptrMyScenarioTree->getLeaveArray(), K*sizeof(uint_t), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeNumChildren, ptrMyScenarioTree->getNumChildren(), numNonLeafNodes*sizeof(uint_t),
			cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeAncestor, ptrMyScenarioTree->getAncestorArray(), nodes*sizeof(uint_t), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeNumChildrenCumul, ptrMyScenarioTree->getNumChildrenCumul(), nodes*sizeof(uint_t),
			cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeProb, ptrMyScenarioTree->getProbArray(), nodes*sizeof(real_t), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeErrorDemand, ptrMyScenarioTree->getErrorDemandArray(), nodes*nd*sizeof(real_t),
			cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devTreeErrorPrices, ptrMyScenarioTree->getErrorPriceArray(), nodes*nu*sizeof(real_t),
				cudaMemcpyHostToDevice) );
	//_CUDA( cudaMemcpy(devForecastValue, ptrMyForecaster->dHat, N*ND*sizeof(real_t), cudaMemcpyHostToDevice) );
}

void Engine::allocateSystemDevice(){
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t nd = ptrMyNetwork->getNumDemands();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t N = ptrMyScenarioTree->getPredHorizon();
	uint_t *nodesPerStage = ptrMyScenarioTree->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyScenarioTree->getNodesPerStageCumul();
	uint_t iStage, iNode;
	_CUDA( cudaMalloc((void**)&devSysMatB, ns*nx*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysMatL, ns*nv*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysMatLhat, ns*nu*nd*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysMatF, 2*nodes*nx*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysMatG, nodes*nu*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysXmin, nodes*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysXmax, nodes*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysXs, nodes*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysXsUpper, nodes*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysUmin, nodes*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysUmax, nodes*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devSysCostW, nodes*nv*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecCurrentState, nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPreviousControl, nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPreviousUhat, nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPreviousDemand, nd*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatWv, nu*nv*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrSysMatB, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrSysMatL, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrSysMatLhat, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrSysMatF, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrSysMatG, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrSysCostW, nodes*sizeof(real_t*)) );

	real_t **ptrSysMatB = new real_t*[nodes];
	real_t **ptrSysMatL = new real_t*[nodes];
	real_t **ptrSysMatLhat = new real_t*[nodes];
	real_t **ptrSysMatF = new real_t*[nodes];
	real_t **ptrSysMatG = new real_t*[nodes];
	real_t **ptrSysCostW = new real_t*[nodes];

	for(uint_t iNode = 0; iNode < nodes; iNode++ ){
		ptrSysMatF[iNode] = &devSysMatF[iNode*2*nx*nx];
		ptrSysMatG[iNode] = &devSysMatG[iNode*nu*nu];
		ptrSysCostW[iNode] = &devSysCostW[iNode*nv*nv];
	}
	for(iStage = 0; iStage < N; iStage++){
		for(iNode = 0; iNode < nodesPerStage[iStage]; iNode++){
			ptrSysMatB[nodesPerStageCumul[iStage] + iNode] = &devSysMatB[iNode*nx*nu];
			ptrSysMatL[nodesPerStageCumul[iStage] + iNode] = &devSysMatL[iNode*nu*nv];
			ptrSysMatLhat[nodesPerStageCumul[iStage] + iNode] = &devSysMatLhat[iNode*nu*nd];
		}
	}

	_CUDA( cudaMemcpy(devPtrSysMatB, ptrSysMatB, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrSysMatL, ptrSysMatL, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrSysMatLhat, ptrSysMatLhat, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrSysMatF, ptrSysMatF, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrSysMatG, ptrSysMatG, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrSysCostW, ptrSysCostW, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );

	delete [] ptrSysMatB;
	delete [] ptrSysMatL;
	delete [] ptrSysMatLhat;
	delete [] ptrSysMatF;
	delete [] ptrSysMatG;
	delete [] ptrSysCostW;
	ptrSysMatB = NULL;
	ptrSysMatL = NULL;
	ptrSysMatLhat = NULL;
	ptrSysMatF = NULL;
	ptrSysMatG = NULL;
	ptrSysCostW = NULL;
	nodesPerStage = NULL;
	nodesPerStageCumul = NULL;
}

void Engine::initialiseSystemDevice(){
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nd = ptrMyNetwork->getNumDemands();
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t N = ptrMyScenarioTree->getPredHorizon();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t *nodesPerStage = ptrMyScenarioTree->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyScenarioTree->getNodesPerStageCumul();
	uint_t numBlock, prevNodes;
	uint_t matFIdx, matGIdx;
	uint_t stateIdx, controlIdx;
	real_t *devMatDiagPrcnd;
	real_t *devCostMatW, *devMatvariable;
	real_t alpha = 1, beta = 0;

	_CUDA( cudaMalloc((void**)&devMatDiagPrcnd, N*(2*nx + nu)*sizeof(real_t)) );
	_CUDA( cudaMemcpy(devMatDiagPrcnd, ptrMySmpcConfig->getMatPrcndDiag(), N*(2*nx + nu)*sizeof(real_t),
			cudaMemcpyHostToDevice) );
	for (uint_t iScen = 0; iScen < ns; iScen++){
		_CUDA( cudaMemcpy(&devSysMatB[iScen*nx*nu], ptrMyNetwork->getMatB(), nx*nu*sizeof(real_t), cudaMemcpyHostToDevice) );
		_CUDA( cudaMemcpy(&devSysMatL[iScen*nu*nv], ptrMySmpcConfig->getMatL(), nu*nv*sizeof(real_t), cudaMemcpyHostToDevice) );
		_CUDA( cudaMemcpy(&devSysMatLhat[iScen*nu*nd], ptrMySmpcConfig->getMatLhat(), nu*nd*sizeof(real_t),
				cudaMemcpyHostToDevice) );
	}
	_CUDA( cudaMalloc((void**)&devCostMatW, nu*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatvariable, nu*nv*sizeof(real_t)) );
	_CUDA( cudaMemcpy(devCostMatW, ptrMySmpcConfig->getCostW(), nu*nu*sizeof(real_t), cudaMemcpyHostToDevice) );
	_CUBLAS( cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, nu, nv, nu, &alpha, (const real_t*) devCostMatW, nu,
			(const real_t*) devSysMatL, nu, &beta, devMatvariable, nu) );
	_CUDA( cudaMemcpy( devMatWv, devMatvariable, nu*nv*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUBLAS( cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nu, &alpha, (const real_t*) devSysMatL, nu,
			(const real_t*) devMatvariable, nu, &beta, devCostMatW, nv) );

	_CUDA( cudaMemset(devSysMatF, 0, nodes*2*nx*nx*sizeof(real_t)) );
	_CUDA( cudaMemset(devSysMatG, 0, nodes*nu*nu*sizeof(real_t)) );

	for (uint_t iNodes = 0; iNodes < nodes; iNodes++){
		_CUDA( cudaMemcpy(&devSysXmin[iNodes*nx], ptrMyNetwork->getXmin(), nx*sizeof(real_t), cudaMemcpyHostToDevice) );
		_CUDA( cudaMemcpy(&devSysXmax[iNodes*nx], ptrMyNetwork->getXmax(), nx*sizeof(real_t), cudaMemcpyHostToDevice) );
		_CUDA( cudaMemcpy(&devSysXs[iNodes*nx], ptrMyNetwork->getXsafe(), nx*sizeof(real_t), cudaMemcpyHostToDevice) );
		_CUDA( cudaMemcpy(&devSysUmin[iNodes*nu], ptrMyNetwork->getUmin(), nu*sizeof(real_t), cudaMemcpyHostToDevice) );
		_CUDA( cudaMemcpy(&devSysUmax[iNodes*nu], ptrMyNetwork->getUmax(), nu*sizeof(real_t), cudaMemcpyHostToDevice) );
		/*_CUBLAS( cublasSscal_v2(handle, nx, &ptrMyScenarioTree->getProbArray()[iNodes], &devSysXmax[iNodes*nx], 1) );
			_CUBLAS( cublasSscal_v2(handle, nx, &ptrMyScenarioTree->getProbArray()[iNodes], &devSysXmin[iNodes*nx], 1) );
			_CUBLAS( cublasSscal_v2(handle, nx, &ptrMyScenarioTree->getProbArray()[iNodes], &devSysXs[iNodes*nx], 1) );
			_CUBLAS( cublasSscal_v2(handle, nu, &ptrMyScenarioTree->getProbArray()[iNodes], &devSysUmax[iNodes*nu], 1) );
			_CUBLAS( cublasSscal_v2(handle, nu, &ptrMyScenarioTree->getProbArray()[iNodes], &devSysUmin[iNodes*nu], 1) );*/
		_CUDA( cudaMemcpy(&devSysCostW[iNodes*nv*nv], devCostMatW, nv*nv*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		_CUBLAS( cublasSscal_v2(handle, nv*nv, &ptrMyScenarioTree->getProbArray()[iNodes], &devSysCostW[iNodes*nv*nv], 1) );
	}

	for (uint_t iStage = 0; iStage < N; iStage++){
		numBlock = nodesPerStage[iStage];
		prevNodes = nodesPerStageCumul[iStage];
		matFIdx = prevNodes*(2*nx * nx);
		matGIdx = prevNodes*(nu * nu);
		stateIdx = prevNodes*nx;
		controlIdx = prevNodes*nu;
		preconditionSystem<<<numBlock, 2*nx+nu>>>(&devSysMatF[matFIdx], &devSysMatG[matGIdx],
				&devMatDiagPrcnd[iStage*(2*nx + nu)], &devTreeProb[prevNodes], nx, nu );
		preconditionConstraintU<<<numBlock, nu>>>(&devSysUmax[controlIdx], &devSysUmin[controlIdx],
				&devMatDiagPrcnd[iStage*(2*nx + nu)], &devTreeProb[prevNodes], nu, numBlock);
		preconditionConstraintX<<<numBlock, nx>>>(&devSysXmax[stateIdx], &devSysXmin[stateIdx], &devSysXs[stateIdx],
				&devMatDiagPrcnd[iStage*(2*nx + nu) + nu], &devTreeProb[prevNodes], nx, numBlock);
	}

	//_CUDA(cudaMemcpy(devSysXsUpper, devSysXmax, nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	uint_t scaleMax = pow(2, 7) - 1;
	_CUDA( cudaMemset(devSysXsUpper, scaleMax, nx*nodes*sizeof(real_t)) );
	//_CUBLAS(cublasSscal_v2(handle, nx*nodes, &scaleMax, devSysXsUpper, 1));
	_CUDA( cudaFree(devMatDiagPrcnd) );
	_CUDA( cudaFree(devMatvariable) );
	_CUDA( cudaFree(devCostMatW) );
	devMatDiagPrcnd = NULL;
	devMatvariable = NULL;
	devCostMatW = NULL;
}

void  Engine::factorStep(){
	initialiseScenarioTreeDevice();
	initialiseSystemDevice();
	real_t scale[2] = {-0.5, 1};
	real_t alpha = 1.0;
	real_t beta = 0.0;
	uint_t iStageCumulNodes, iStageNodes;
	real_t *devMatBbar, *devMatGbar;
	real_t **devPtrMatBbar, **devPtrMatGbar, **ptrMatBbar, **ptrMatGbar;
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t N = ptrMyScenarioTree->getPredHorizon();
	uint_t *nodesPerStage = ptrMyScenarioTree->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyScenarioTree->getNodesPerStageCumul();

	_CUDA( cudaMalloc((void**)&devMatBbar, nv*nx*ns*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatGbar, nu*nv*ns*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devPtrMatBbar, ns*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrMatGbar, ns*sizeof(real_t*)) );
	ptrMatBbar = new real_t*[ns];
	ptrMatGbar = new real_t*[ns];
	for(uint_t i = 0; i < ns; i++){
		ptrMatBbar[i] = &devMatBbar[i*nx*nv];
		ptrMatGbar[i] = &devMatGbar[i*nu*nv];
	}
	_CUDA( cudaMemcpy(devPtrMatGbar, ptrMatGbar, ns*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrMatBbar, ptrMatBbar, ns*sizeof(real_t*), cudaMemcpyHostToDevice) );
	// Bbar'
	_CUBLAS( cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, nv, nx, nu, &alpha, (const real_t**)devPtrSysMatL, nu,
			(const real_t**)devPtrSysMatB, nx, &beta, devPtrMatBbar, nv, ns));

	for(uint_t iStage = N-1; iStage > -1; iStage--){
		iStageCumulNodes = nodesPerStageCumul[iStage];
		iStageNodes = nodesPerStage[iStage];
		// omega=(p_k\bar{R})^{-1}
		inverseBatchMat( &devPtrSysCostW[iStageCumulNodes], &devPtrMatOmega[iStageCumulNodes], nv, iStageNodes );
		// effinet_f=GBar
		_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, nv, nu, nu, &alpha, (const real_t**)devPtrSysMatL, nu,
				(const real_t**)&devPtrSysMatG[iStageCumulNodes], nu, &beta, &devPtrMatF[iStageCumulNodes], nv, iStageNodes) );
		// effinet_g=\bar{B}'
		_CUDA( cudaMemcpy(&devMatG[nx*nv*iStageCumulNodes], devMatBbar, nx*nv*iStageNodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		// effinet_d=\bar{B}'F'
		_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nv, 2*nx, nx, &alpha, (const real_t**)devPtrMatBbar, nv,
				(const real_t**)&devPtrSysMatF[iStageCumulNodes], 2*nx, &beta, &devPtrMatD[iStageCumulNodes], nv, iStageNodes));
		// phi=\omega \bar{B}'F'
		_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nv, 2*nx, nv, &scale[0], (const real_t**)
				&devPtrMatOmega[iStageCumulNodes], nv, (const real_t**)&devPtrMatD[iStageCumulNodes], nv, &beta,
				&devPtrMatPhi[iStageCumulNodes], nv, iStageNodes));
		// theta=\omega \bar{B}'
		_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nv, nx, nv, &scale[0], (const real_t**)
				&devPtrMatOmega[iStageCumulNodes], nv, (const real_t**)devPtrMatBbar, nv, &beta,
				&devPtrMatTheta[iStageCumulNodes], nv , iStageNodes));
		// psi=\omega \bar{G}'
		_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nv, nu, nv, &scale[0],(const real_t**)
				&devPtrMatOmega[iStageCumulNodes], nv, (const real_t**)&devPtrMatF[iStageCumulNodes], nv, &beta,
				&devPtrMatPsi[iStageCumulNodes], nv, iStageNodes));
	}
	//cout << "Factor step is completed\n";

	delete [] ptrMatBbar;
	delete [] ptrMatGbar;
	_CUDA(cudaFree(devMatBbar));
	_CUDA(cudaFree(devMatGbar));
	_CUDA(cudaFree(devPtrMatBbar));
	_CUDA(cudaFree(devPtrMatGbar));
	ptrMatBbar = NULL;
	ptrMatGbar = NULL;
	devMatBbar = NULL;
	devMatGbar = NULL;
	devPtrMatBbar = NULL;
	devPtrMatGbar = NULL;
}

/**
 *  pointer to the scenario tree
 */
ScenarioTree* Engine::getScenarioTree(){
	return ptrMyScenarioTree;
}
/**
 *  pointer to the DWN network
 */
DwnNetwork* Engine::getDwnNetwork(){
	return ptrMyNetwork;
}
/** ----GETTER'S FOR THE FACTOR STEP---*/
/**
 *  matrix Phi
 */
real_t* Engine::getMatPhi(){
	return devMatPhi;
}
/**
 * matrix Psi
 */
real_t* Engine::getMatPsi(){
	return devMatPsi;
}
/**
 * matrix Theta
 */
real_t* Engine::getMatTheta(){
	return devMatTheta;
}
/**
 * matrix Theta
 */
real_t* Engine::getMatOmega(){
	return devMatOmega;
}
/**
 * matrix Sigma
 */
real_t* Engine::getMatSigma(){
	return devMatSigma;
}
/**
 * matrix D
 */
real_t* Engine::getMatD(){
	return devMatD;
}
/**
 * matrix F (Factor step)
 */
real_t* Engine::getMatF(){
	return devMatF;
}
/**
 * matrix G (Factor step)
 */
real_t* Engine::getMatG(){
	return devMatG;
}
/**
 * pointer matrix Phi
 */
real_t** Engine::getPtrMatPhi(){
	return devPtrMatPhi;
}
/**
 * pointer matrix Psi
 */
real_t** Engine::getPtrMatPsi(){
	return devPtrMatPsi;
}
/**
 * pointer matrix Theta
 */
real_t** Engine::getPtrMatTheta(){
	return devPtrMatTheta;
}
/**
 * pointer matrix Omega
 */
real_t** Engine::getPtrMatOmega(){
	return devPtrMatOmega;
}
/**
 * pointer matrix Sigma
 */
real_t** Engine::getPtrMatSigma(){
	return devPtrMatSigma;
}
/**
 * pointer matrix D
 */
real_t** Engine::getPtrMatD(){
	return devPtrMatD;
}
/**
 * pointer matrix F (Factor step)
 */
real_t** Engine::getPtrMatF(){
	return devPtrMatF;
}
/**
 * pointer matrix G (Factor step)
 */
real_t** Engine::getPtrMatG(){
	return devPtrMatG;
}
/**
 * uhat
 */
real_t* Engine::getVecUhat(){
	return devVecUhat;
}
/**
 * beta control-distribution elimination
 */
real_t* Engine::getVecBeta(){
	return devVecBeta;
}
/**
 * e control-disturbance elimination
 */
real_t* Engine::getVecE(){
	return devVecE;
}
/** ---GETTER'S FOR THE SYSTEM MATRICES */
real_t* Engine::getSysMatB(){
	return devSysMatB;
}
/**
 * constraints matrix F
 */
real_t* Engine::getSysMatF(){
	return devSysMatF;
}
/**
 * constraints matrix G
 */
real_t* Engine::getSysMatG(){
	return devSysMatG;
}
/**
 * matrix L
 */
real_t* Engine::getSysMatL(){
	return devSysMatL;
}
/**
 * matrix Lhat
 */
real_t* Engine::getSysMatLhat(){
	return devSysMatLhat;
}
/**
 * pointer to Matrix B
 */
real_t** Engine::getPtrSysMatB(){
	return devPtrSysMatB;
}
/**
 * pointer to matrix F
 */
real_t** Engine::getPtrSysMatF(){
	return devPtrSysMatF;
}
/**
 * pointer to matrix G
 */
real_t** Engine::getPtrSysMatG(){
	return devPtrSysMatG;
}
/**
 * pointer to matrix L
 */
real_t** Engine::getPtrSysMatL(){
	return devPtrSysMatLhat;
}
/**
 * pointer to matrix Lhat
 */
real_t** Engine::getPtrSysMatLhat(){
	return devPtrSysMatLhat;
}
/**
 * previous control
 */
real_t* Engine::getVecPreviousControl(){
	return devVecPreviousControl;
}
/**
 * current state
 */
real_t* Engine::getVecCurrentState(){
	return devVecCurrentState;
}
/**
 * previous uhat
 */
real_t* Engine::getVecPreviousUhat(){
	return devVecPreviousUhat;
}
/**
 * previous demand
 */
real_t* Engine::getVecDemand(){
	return devVecPreviousDemand;
}
/** ----GETTER'S FOR THE SCENARIO TREE----*/
/**
 *  Array of the stage of the nodes at the tree
 */
uint_t* Engine::getTreeStages(){
	return devTreeStages;
}
/**
 * Array of nodes per stage
 */
uint_t* Engine::getTreeNodesPerStage(){
	return devTreeNodesPerStage;
}
/**
 * Array of past nodes
 */
uint_t* Engine::getTreeNodesPerStageCumul(){
	return devTreeNodesPerStageCumul;
}
/**
 * Array of the leaves
 */
uint_t* Engine::getTreeLeaves(){
	return devTreeLeaves;
}
/**
 * Array number of children
 */
uint_t* Engine::getTreeNumChildren(){
	return devTreeNumChildren;
}
/**
 * Array of ancestor
 */
uint_t* Engine::getTreeAncestor(){
	return devTreeAncestor;
}
/**
 * Array of past cumulative children
 */
uint_t* Engine::getTreeNumChildrenCumul(){
	return devTreeNumChildrenCumul;
}
/**
 * Array of the probability
 */
real_t* Engine::getTreeProb(){
	return devTreeProb;
}
/**
 * Array of the error in the demand
 */
real_t* Engine::getTreeErrorDemand(){
	return devTreeErrorDemand;
}
/**
 * Array of the error in the prices
 */
real_t* Engine::getTreeErrorPrices(){
	return devTreeErrorPrices;
}
/** ----GETTER'S OF NETWORK CONSTRAINTS----*/
/**
 * state/volume minimum
 */
real_t* Engine::getSysXmin(){
	return devSysXmin;
}
/**
 * state/volume maximum
 */
real_t* Engine::getSysXmax(){
	return devSysXmax;
}
/**
 * state/volume safe level
 */
real_t* Engine::getSysXs(){
	return devSysXs;
}
/**
 * dummy state/volume safe level
 */
real_t* Engine::getSysXsUpper(){
	return devSysXsUpper;
}
/**
 * actuator/control minimum
 */
real_t* Engine::getSysUmin(){
	return devSysUmin;
}
/**
 * actuator/control maximum
 */
real_t* Engine::getSysUmax(){
	return devSysUmax;
}
/**
 * cublasHandle
 */
cublasHandle_t Engine::getCublasHandle(){
	return handle;
}

/**
 * status of price uncertainty
 */
bool Engine::getPriceUncertainty(){
	return priceUncertaintyFlag;
}
/**
 * status of the demand uncertanity
 */
bool Engine::getDemandUncertantiy(){
	return demandUncertaintyFlag;
}
/*  SETTER'S IN THE ENGINE  */
/*
 * Option for uncertainty in price
 * @param    priceUncertaintyFlag    true to include uncertainty (default)
 *                                   false to include uncertainty (default)
 */
void Engine::setPriceUncertaintyFlag(bool inputFlag){
	priceUncertaintyFlag = inputFlag;
}
/*
 * Option for uncertainty in demand
 * @param    demandUncertaintyFlag    true to include uncertainty (default)
 *                                    false to include uncertainty (default)
 */
void Engine::setDemandUncertaintyFlag(bool inputFlag){
	demandUncertaintyFlag = inputFlag;
}

void Engine::eliminateInputDistubanceCoupling(real_t* nominalDemand, real_t *nominalPrices){
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t nd = ptrMyNetwork->getNumDemands();
	uint_t N =  ptrMyScenarioTree->getPredHorizon();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t numNonleafNodes =  ptrMyScenarioTree->getNumNonleafNodes();
	real_t alpha = 1, beta = 0;
	real_t *devMatGd;
	real_t **devPtrMatGd, **devPtrVecE, **devPtrVecDemand;
	real_t **ptrMatGd = new real_t*[ns];
	real_t **ptrVecE = new real_t*[nodes];
	real_t **ptrVecDemand = new real_t*[nodes];
	real_t **ptrVecUhat = new real_t*[nodes];
	real_t *devVecDemand, *devVecDemandHat;
	real_t **devPtrVecUhat, *devVecDeltaUhat, *devVecZeta;
	real_t *devVecAlphaHat;
	real_t *devVecAlpha1;
	real_t *devVecAlpha;
	real_t *devVecAlphaBar;
	real_t *devMatRhat;
	uint_t *nodeStage = ptrMyScenarioTree->getStageNodes();
	uint_t *nodesPerStage = ptrMyScenarioTree->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyScenarioTree->getNodesPerStageCumul();
	uint_t iStageNodes, iStageCumulNodes, jNodes, iStage;


	_CUDA( cudaMalloc((void**)&devVecDemand, nodes*nd*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDemandHat, N*nd*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatGd, ns*nx*nd*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devPtrMatGd, ns*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecE, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecDemand, nodes*sizeof(real_t*)));
	_CUDA( cudaMalloc((void**)&devPtrVecUhat, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devVecAlphaHat, N*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecAlpha1, nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecAlpha, nodes*nu*sizeof(real_t)));
	_CUDA( cudaMalloc((void**)&devVecAlphaBar, nodes*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devMatRhat, nu*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDeltaUhat, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecZeta, nu*nodes*sizeof(real_t)) );


	for (uint_t iScenario = 0; iScenario < ns; iScenario++){
		_CUDA( cudaMemcpy(&devMatGd[iScenario*nx*nd], ptrMyNetwork->getMatGd(), nx*nd*sizeof(real_t), cudaMemcpyHostToDevice) );
		ptrMatGd[iScenario] = &devMatGd[iScenario*nx*nd];
	}
	for( uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecE[iNode] = &devVecE[iNode*nx];
		ptrVecDemand[iNode] = &devVecDemand[iNode*nd];
		ptrVecUhat[iNode] = &devVecUhat[iNode*nu];
	}
	_CUDA( cudaMemcpy(devPtrMatGd, ptrMatGd, ns*sizeof(real_t*), cudaMemcpyHostToDevice));
	_CUDA( cudaMemcpy(devPtrVecDemand, ptrVecDemand, nodes*sizeof(real_t*), cudaMemcpyHostToDevice));
	_CUDA( cudaMemcpy(devPtrVecE, ptrVecE, nodes*sizeof(real_t*), cudaMemcpyHostToDevice));
	_CUDA( cudaMemcpy(devPtrVecUhat, ptrVecUhat, nodes*sizeof(real_t*), cudaMemcpyHostToDevice));
	// d(node) = dhat(stage) + d(node)
	// e = Gd*d
	_CUDA( cudaMemcpy(devVecDemand, ptrMyScenarioTree->getErrorDemandArray(), nodes*nd*sizeof(real_t),
			cudaMemcpyHostToDevice ));
	if(!demandUncertaintyFlag){
		_CUBLAS( cublasSscal_v2(handle, nu*nodes, &beta, devVecDemand, 1) );
	}
	_CUDA( cudaMemcpy(devVecDemandHat, nominalDemand, N*nd*sizeof(real_t), cudaMemcpyHostToDevice ));
	for (iStage = 0 ; iStage < N; iStage++){
		iStageCumulNodes = nodesPerStageCumul[iStage];
		iStageNodes = nodesPerStage[iStage];
		for(uint_t j = 0; j < iStageNodes; j++){
			jNodes = iStageCumulNodes + j;
			_CUBLAS( cublasSaxpy_v2(handle, nd, &alpha, &devVecDemandHat[iStage*nd], 1, &devVecDemand[jNodes*nd],1) );
		}
		_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx, 1, nd, &alpha, (const real_t**)devPtrMatGd,
				nx, (const real_t**)&devPtrVecDemand[iStageCumulNodes], nd, &beta, &devPtrVecE[iStageCumulNodes],
				nx , iStageNodes));
	}
	// uhat = Lhat*d
	_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nu, 1, nd, &alpha, (const real_t**)
			devPtrSysMatLhat, nu, (const real_t**)devPtrVecDemand, nd, &beta, devPtrVecUhat, nu , nodes));
	// alpha = alphaHat + alpha1 + errorprice
	_CUDA( cudaMemcpy(devVecAlphaHat, nominalPrices, N*nu*sizeof(real_t), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devVecAlpha1, ptrMyNetwork->getAlpha(), nu*sizeof(real_t), cudaMemcpyHostToDevice));
	_CUDA(cudaMemcpy(devVecAlpha, ptrMyScenarioTree->getErrorPriceArray(), nu*nodes*sizeof(real_t), cudaMemcpyHostToDevice));
	//_CUBLAS(cublasSscal(handle,n,&al,d x,1));
	if(!priceUncertaintyFlag){
		_CUBLAS( cublasSscal_v2(handle, nu*nodes, &beta, devVecAlpha, 1) );
	}

	for(iStage = 0; iStage < N; iStage++){
		_CUBLAS( cublasSaxpy_v2(handle, nu, &alpha, devVecAlpha1, 1, &devVecAlphaHat[iStage*nu], 1) );
	}
	for(uint_t iNode = 0; iNode < nodes; iNode++){
		iStage = nodeStage[iNode];
		_CUBLAS( cublasSaxpy_v2(handle, nu, &alpha, &devVecAlphaHat[iStage*nu], 1 , &devVecAlpha[iNode*nu], 1));
	}
	//scaling with the weight
	real_t weightEconomical = ptrMySmpcConfig->getWeightEconomical();
	_CUBLAS( cublasSscal_v2(handle, nu*nodes, &weightEconomical, devVecAlpha, 1) );
	// alphaBar = L* (alpha)
	_CUBLAS( cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, nv, nodes, nu, &alpha, (const real_t*) devSysMatL, nu,
			(const real_t*)devVecAlpha, nu, &beta, devVecAlphaBar, nv));
	_CUBLAS( cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, nu, nv, nu, &alpha, (const real_t*) devSysCostW, nu,
			(const real_t*)devSysMatL, nu, &beta, devMatRhat, nu));
	// Beta
	calculateDiffUhat<<<nodes, nu>>>(devVecDeltaUhat, devVecUhat, devVecPreviousUhat, devTreeAncestor, nu, nodes);
	calculateZeta<<<nodes, nu>>>(devVecZeta, devVecDeltaUhat, devTreeProb, devTreeNumChildrenCumul, nu, numNonleafNodes, nodes);

	alpha = 2;
	_CUBLAS( cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, nv, nodes, nu, &alpha, (const real_t *) devMatWv, nu,
			(const real_t *) devVecZeta, nu, &beta, devVecBeta, nv) );
	alpha = 1;

	for(uint_t iNode = 0; iNode < nodes; iNode++){
		real_t scale = ptrMyScenarioTree->getProbArray()[iNode];
		_CUBLAS( cublasSaxpy_v2(handle, nv, &scale, &devVecAlphaBar[nv*iNode], 1, &devVecBeta[nv*iNode],1) );
	}

	delete [] ptrMatGd;
	delete [] ptrVecE;
	delete [] ptrVecDemand;
	delete [] ptrVecUhat;
	_CUDA(cudaFree(devVecDemand));
	_CUDA(cudaFree(devVecDemandHat));
	_CUDA(cudaFree(devMatGd));
	_CUDA(cudaFree(devPtrMatGd));
	_CUDA(cudaFree(devPtrVecE));
	_CUDA(cudaFree(devPtrVecDemand));
	_CUDA(cudaFree(devPtrVecUhat));
	_CUDA(cudaFree(devVecAlpha));
	_CUDA(cudaFree(devVecAlphaHat));
	_CUDA(cudaFree(devVecAlpha1));
	_CUDA(cudaFree(devVecAlphaBar));
	_CUDA(cudaFree(devMatRhat));
	_CUDA(cudaFree(devVecDeltaUhat));
	_CUDA(cudaFree(devVecZeta));

	ptrMatGd = NULL;
	ptrVecE = NULL;
	ptrVecDemand = NULL;
	ptrVecUhat = NULL;
	devVecDemand = NULL;
	devVecDemandHat = NULL;
	devMatGd = NULL;
	devPtrMatGd = NULL;
	devPtrVecE = NULL;
	devPtrVecDemand = NULL;
	devPtrVecUhat = NULL;
	devVecAlphaHat = NULL;
	devVecAlpha = NULL;
	devVecAlpha1 = NULL;
	devVecAlphaBar = NULL;
	devMatRhat = NULL;
	devVecDeltaUhat = NULL;
	devVecZeta = NULL;
}

void Engine::updateStateControl(real_t* currentX, real_t* prevU, real_t* prevDemand){
	real_t alpha = 1;
	real_t beta = 0;
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nd = ptrMyNetwork->getNumDemands();
	uint_t nv = ptrMySmpcConfig->getNV();

	_CUDA( cudaMemcpy(devVecCurrentState, currentX, ptrMyNetwork->getNumTanks()*sizeof(real_t),
			cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devVecPreviousControl, prevU, ptrMyNetwork->getNumControls()*sizeof(real_t),
			cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devVecPreviousDemand, prevDemand, ptrMyNetwork->getNumDemands()*sizeof(real_t),
			cudaMemcpyHostToDevice));
	_CUBLAS(cublasSgemv_v2(handle, CUBLAS_OP_N, nu, nd, &alpha, devSysMatLhat, nu, devVecPreviousDemand, 1,
			&beta, devVecPreviousUhat, 1) );
}

void Engine::inverseBatchMat(real_t** src, real_t** dst, uint_t n, uint_t batchSize){
	uint_t *P, *INFO;

	_CUDA(cudaMalloc((void**)&P, n * batchSize * sizeof(uint_t)));
	_CUDA(cudaMalloc((void**)&INFO, batchSize * sizeof(uint_t)));

	uint_t lda = n;

	real_t** x=(real_t**)malloc(batchSize*sizeof(real_t*));
	real_t* y=(real_t*)malloc(n*n*sizeof(real_t));


	_CUBLAS(cublasSgetrfBatched(handle,n,src,lda,P,INFO,batchSize));

	uint_t INFOh[batchSize];

	_CUDA(cudaMemcpy(INFOh,INFO,batchSize*sizeof(uint_t),cudaMemcpyDeviceToHost));
	for (uint_t i = 0; i < batchSize; i++){
		if(INFOh[i] != 0)
		{
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	}

	_CUBLAS(cublasSgetriBatched(handle,n,(const real_t **)src,lda,P,dst,lda,INFO,batchSize));
	_CUDA(cudaMemcpy(INFOh,INFO,batchSize*sizeof(uint_t),cudaMemcpyDeviceToHost));

	for (uint_t i = 0; i < batchSize; i++)
		if(INFOh[i] != 0)
		{
			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	_CUDA(cudaFree(P));
	_CUDA(cudaFree(INFO));
	P = NULL;
	INFO = NULL;
}


void Engine::deallocateSystemDevice(){
	_CUDA( cudaFree(devSysMatB) );
	_CUDA( cudaFree(devSysMatL) );
	_CUDA( cudaFree(devSysMatLhat) );
	_CUDA( cudaFree(devSysMatF) );
	_CUDA( cudaFree(devSysMatG) );
	_CUDA( cudaFree(devSysXmin) );
	_CUDA( cudaFree(devSysXmax) );
	_CUDA( cudaFree(devSysXs) );
	_CUDA( cudaFree(devSysXsUpper) );
	_CUDA( cudaFree(devSysUmin) );
	_CUDA( cudaFree(devSysUmax) );
	_CUDA( cudaFree(devVecCurrentState));
	_CUDA( cudaFree(devVecPreviousControl));
	_CUDA( cudaFree(devVecPreviousUhat));
	_CUDA( cudaFree(devVecPreviousDemand));
	_CUDA( cudaFree(devMatWv) );

	_CUDA( cudaFree(devPtrSysMatB) );
	_CUDA( cudaFree(devPtrSysMatL) );
	_CUDA( cudaFree(devPtrSysMatLhat) );
	_CUDA( cudaFree(devPtrSysMatF) );
	_CUDA( cudaFree(devPtrSysMatG) );

	devSysMatB = NULL;
	devSysMatL = NULL;
	devSysMatLhat = NULL;
	devSysMatF = NULL;
	devSysMatG = NULL;
	devSysXmin = NULL;
	devSysXmax = NULL;
	devSysXs = NULL;
	devSysXsUpper = NULL;
	devSysUmin = NULL;
	devSysUmax = NULL;
	devVecCurrentState = NULL;
	devVecPreviousControl = NULL;
	devVecPreviousUhat = NULL;
	devVecPreviousDemand = NULL;
	devMatWv = NULL;

	devPtrSysMatB = NULL;
	devPtrSysMatL = NULL;
	devPtrSysMatLhat = NULL;
	devPtrSysMatF = NULL;
	devPtrSysMatG = NULL;
}

void Engine::deallocateScenarioTreeDevice(){
	_CUDA( cudaFree(devTreeStages) );
	_CUDA( cudaFree(devTreeNodesPerStage));
	_CUDA( cudaFree(devTreeLeaves) );
	_CUDA( cudaFree(devTreeNodesPerStageCumul) );
	_CUDA( cudaFree(devTreeNumChildren) );
	_CUDA( cudaFree(devTreeNumChildrenCumul) );
	_CUDA( cudaFree(devTreeErrorDemand) );
	_CUDA( cudaFree(devTreeErrorPrices) );
	//_CUDA( cudaFree(devForecastValue) );

	devTreeStages = NULL;
	devTreeNodesPerStage = NULL;
	devTreeLeaves = NULL;
	devTreeNodesPerStageCumul = NULL;
	devTreeNumChildren = NULL;
	devTreeNumChildrenCumul = NULL;
	devTreeErrorDemand = NULL;
	devTreeErrorPrices = NULL;
	//devForecastValue = NULL;
}
Engine::~Engine(){
	deallocateSystemDevice();
	deallocateScenarioTreeDevice();
	//delete ptrmyForecaster;
	//delete ptrMyNetwork;
	_CUDA(cudaFree(devMatPhi));
	_CUDA(cudaFree(devMatPsi));
	_CUDA(cudaFree(devMatTheta));
	_CUDA(cudaFree(devMatOmega));
	_CUDA(cudaFree(devMatSigma));
	_CUDA(cudaFree(devMatD));
	_CUDA(cudaFree(devMatF));
	_CUDA(cudaFree(devMatG));
	_CUDA(cudaFree(devVecUhat));
	_CUDA(cudaFree(devVecBeta));
	_CUDA(cudaFree(devVecE));

	_CUDA(cudaFree(devPtrMatPhi));
	_CUDA(cudaFree(devPtrMatPsi));
	_CUDA(cudaFree(devPtrMatTheta));
	_CUDA(cudaFree(devPtrMatOmega));
	_CUDA(cudaFree(devPtrMatSigma));
	_CUDA(cudaFree(devPtrMatD));
	_CUDA(cudaFree(devPtrMatF));
	_CUDA(cudaFree(devPtrMatG));

	devMatPhi = NULL;
	devMatPsi = NULL;
	devMatTheta = NULL;
	devMatOmega = NULL;
	devMatSigma = NULL;
	devMatD = NULL;
	devMatF = NULL;
	devMatG = NULL;
	devVecUhat = NULL;
	devVecBeta = NULL;
	devVecE = NULL;

	devPtrMatPhi = NULL;
	devPtrMatPsi = NULL;
	devPtrMatTheta = NULL;
	devPtrMatOmega = NULL;
	devPtrMatSigma = NULL;
	devPtrMatD = NULL;
	devPtrMatF = NULL;
	devPtrMatG = NULL;
	//_CUBLAS(cublasDestroy(handle));
	_CUBLAS(cublasDestroy_v2(handle));
}
