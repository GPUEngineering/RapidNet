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
#include "SmpcController.cuh"


SmpcController::SmpcController(Forecaster *myForecaster, Engine *myEngine, SmpcConfiguration *mySmpcConfig){
	ptrMyForecaster = myForecaster;
	ptrMyEngine = myEngine;
	ptrMySmpcConfig = mySmpcConfig;
	DwnNetwork* ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree* ptrMyScenarioTree = ptrMyEngine->getScenarioTree();

	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	stepSize = ptrMySmpcConfig->getStepSize();
	FlagFactorStep = false;

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
	_CUDA( cudaMalloc((void**)&devVecPrimalInfeasibilty, (2*nx + nu)*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecQ, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecR, ns*nv*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrVecX, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecU, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecV, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecAcceleratedXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecAcceleratedPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecQ, ns*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecR, ns*sizeof(real_t*)) );

	real_t** ptrVecX = new real_t*[nodes];
	real_t** ptrVecU = new real_t*[nodes];
	real_t** ptrVecV = new real_t*[nodes];
	real_t** ptrVecAcceleratedXi = new real_t*[nodes];
	real_t** ptrVecAcceleratedPsi = new real_t*[nodes];
	real_t** ptrVecPrimalXi = new real_t*[nodes];
	real_t** ptrVecPrimalPsi = new real_t*[nodes];
	real_t** ptrVecQ = new real_t*[ns];
	real_t** ptrVecR = new real_t*[ns];

	for(uint_t iScenario = 0; iScenario < ns; iScenario++){
		ptrVecQ[iScenario] = &devVecQ[iScenario*nx];
		ptrVecR[iScenario] = &devVecR[iScenario*nv];
	}
	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecX[iNode] = &devVecX[iNode*nx];
		ptrVecU[iNode] = &devVecU[iNode*nu];
		ptrVecV[iNode] = &devVecV[iNode*nv];
		ptrVecAcceleratedXi[iNode] = &devVecAcceleratedXi[2*iNode*nx];
		ptrVecAcceleratedPsi[iNode] = &devVecAcceleratedPsi[iNode*nu];
		ptrVecPrimalXi[iNode] = &devVecPrimalXi[2*iNode*nx];
		ptrVecPrimalPsi[iNode] = &devVecPrimalPsi[iNode*nu];
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
	_CUDA( cudaMemset(devVecPrimalInfeasibilty, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecR, 0, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecQ, 0, ns*nx*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecX, ptrVecX, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecU, ptrVecU, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecV, ptrVecV, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecAcceleratedXi, ptrVecAcceleratedXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecAcceleratedPsi, ptrVecAcceleratedPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalXi, ptrVecPrimalXi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalPsi, ptrVecPrimalPsi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecQ, ptrVecQ, ns*sizeof(real_t*), cudaMemcpyHostToDevice));
	_CUDA( cudaMemcpy(devPtrVecR, ptrVecR, ns*sizeof(real_t*), cudaMemcpyHostToDevice));

	vecPrimalInfs = new real_t[ptrMySmpcConfig->getMaxIterations()];

	delete [] ptrVecX;
	delete [] ptrVecU;
	delete [] ptrVecV;
	delete [] ptrVecAcceleratedXi;
	delete [] ptrVecAcceleratedPsi;
	delete [] ptrVecPrimalXi;
	delete [] ptrVecPrimalPsi;
	delete [] ptrVecQ;
	delete [] ptrVecR;
	ptrVecX = NULL;
	ptrVecU = NULL;
	ptrVecV = NULL;
	ptrVecAcceleratedXi = NULL;
	ptrVecAcceleratedPsi = NULL;
	ptrVecPrimalXi = NULL;
	ptrVecPrimalPsi = NULL;
	ptrVecQ = NULL;
	ptrVecR = NULL;
	ptrMyNetwork = NULL;
	ptrMyScenarioTree = NULL;
}

SmpcController::SmpcController(string pathToConfigFile){
	ptrMySmpcConfig = new SmpcConfiguration( pathToConfigFile );
	string pathToForecaster = ptrMySmpcConfig->getPathToForecaster();
	ptrMyForecaster = new Forecaster( pathToForecaster );
	ptrMyEngine = new Engine( ptrMySmpcConfig );

	DwnNetwork* ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree* ptrMyScenarioTree = ptrMyEngine->getScenarioTree();

	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	stepSize = ptrMySmpcConfig->getStepSize();
	FlagFactorStep = false;

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
	_CUDA( cudaMalloc((void**)&devVecPrimalInfeasibilty, (2*nx + nu)*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecQ, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecR, ns*nv*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrVecX, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecU, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecV, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecAcceleratedXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecAcceleratedPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecQ, ns*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecR, ns*sizeof(real_t*)) );

	real_t** ptrVecX = new real_t*[nodes];
	real_t** ptrVecU = new real_t*[nodes];
	real_t** ptrVecV = new real_t*[nodes];
	real_t** ptrVecAcceleratedXi = new real_t*[nodes];
	real_t** ptrVecAcceleratedPsi = new real_t*[nodes];
	real_t** ptrVecPrimalXi = new real_t*[nodes];
	real_t** ptrVecPrimalPsi = new real_t*[nodes];
	real_t** ptrVecQ = new real_t*[ns];
	real_t** ptrVecR = new real_t*[ns];

	for(uint_t iScenario = 0; iScenario < ns; iScenario++){
		ptrVecQ[iScenario] = &devVecQ[iScenario*nx];
		ptrVecR[iScenario] = &devVecR[iScenario*nv];
	}
	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecX[iNode] = &devVecX[iNode*nx];
		ptrVecU[iNode] = &devVecU[iNode*nu];
		ptrVecV[iNode] = &devVecV[iNode*nv];
		ptrVecAcceleratedXi[iNode] = &devVecAcceleratedXi[2*iNode*nx];
		ptrVecAcceleratedPsi[iNode] = &devVecAcceleratedPsi[iNode*nu];
		ptrVecPrimalXi[iNode] = &devVecPrimalXi[2*iNode*nx];
		ptrVecPrimalPsi[iNode] = &devVecPrimalPsi[iNode*nu];
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
	_CUDA( cudaMemset(devVecPrimalInfeasibilty, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecR, 0, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecQ, 0, ns*nx*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecX, ptrVecX, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecU, ptrVecU, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecV, ptrVecV, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecAcceleratedXi, ptrVecAcceleratedXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecAcceleratedPsi, ptrVecAcceleratedPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalXi, ptrVecPrimalXi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalPsi, ptrVecPrimalPsi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecQ, ptrVecQ, ns*sizeof(real_t*), cudaMemcpyHostToDevice));
	_CUDA( cudaMemcpy(devPtrVecR, ptrVecR, ns*sizeof(real_t*), cudaMemcpyHostToDevice));

	vecPrimalInfs = new real_t[ptrMySmpcConfig->getMaxIterations()];

	delete [] ptrVecX;
	delete [] ptrVecU;
	delete [] ptrVecV;
	delete [] ptrVecAcceleratedXi;
	delete [] ptrVecAcceleratedPsi;
	delete [] ptrVecPrimalXi;
	delete [] ptrVecPrimalPsi;
	delete [] ptrVecQ;
	delete [] ptrVecR;
	ptrVecX = NULL;
	ptrVecU = NULL;
	ptrVecV = NULL;
	ptrVecAcceleratedXi = NULL;
	ptrVecAcceleratedPsi = NULL;
	ptrVecPrimalXi = NULL;
	ptrVecPrimalPsi = NULL;
	ptrVecQ = NULL;
	ptrVecR = NULL;
	ptrMyNetwork = NULL;
	ptrMyScenarioTree = NULL;
}

void SmpcController::initialiseSmpcController(){
	FlagFactorStep = true;
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	currentX = NULL;
	prevU = NULL;
	prevDemand = NULL;
}

void SmpcController::dualExtrapolationStep(real_t lambda){
	DwnNetwork* ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree* ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	real_t alpha;
	// w = (1 + \lambda)y_k - \lambda y_{k-1}
	_CUDA(cudaMemcpy(devVecAcceleratedXi, devVecUpdateXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUDA(cudaMemcpy(devVecAcceleratedPsi, devVecUpdatePsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	alpha = 1 + lambda;
	_CUBLAS(cublasSscal_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &alpha, devVecAcceleratedXi, 1));
	_CUBLAS(cublasSscal_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &alpha, devVecAcceleratedPsi, 1));
	alpha = -lambda;
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &alpha, devVecXi, 1, devVecAcceleratedXi, 1));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &alpha, devVecPsi, 1, devVecAcceleratedPsi, 1));
	// y_{k} = y_{k-1}
	_CUDA(cudaMemcpy(devVecXi, devVecUpdateXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUDA(cudaMemcpy(devVecPsi, devVecUpdatePsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	ptrMyNetwork = NULL;
	ptrMyScenarioTree = NULL;
}

void SmpcController::solveStep(){
	DwnNetwork *ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree *ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	real_t *devTempVecR, *devTempVecQ, *devLv;
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyScenarioTree->getNumScenarios();
	uint_t N =  ptrMyScenarioTree->getPredHorizon();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t *nodesPerStage = ptrMyScenarioTree->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyScenarioTree->getNodesPerStageCumul();
	uint_t iStageCumulNodes, iStageNodes, prevStageNodes, prevStageCumulNodes;
	real_t scale[3] = {-0.5, 1, -1};
	real_t alpha = 1;
	real_t beta = 0;

	if(FlagFactorStep == false){
		initialiseSmpcController();
		FlagFactorStep = true;
	}

	_CUDA( cudaMalloc((void**)&devTempVecQ, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devTempVecR, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devLv, ns*nu*sizeof(real_t)) );
	_CUDA( cudaMemcpy(ptrMyEngine->getMatSigma(), ptrMyEngine->getVecBeta(), nv*nodes*sizeof(real_t),
			cudaMemcpyDeviceToDevice) );

	//Backward substitution
	for(uint_t iStage = N-1;iStage > -1;iStage--){
		iStageCumulNodes = nodesPerStageCumul[iStage];
		iStageNodes = nodesPerStage[iStage];

		if(iStage < N-1){
			// sigma=sigma+r
			_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), iStageNodes*nv, &alpha, devVecR, 1,
					&ptrMyEngine->getMatSigma()[iStageCumulNodes*nv],1));
		}

		// v=Omega*sigma
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nv,
				&scale[0], (const real_t**)&ptrMyEngine->getPtrMatOmega()[iStageCumulNodes], nv,
				(const real_t**)&ptrMyEngine->getPtrMatSigma()[iStageCumulNodes], nv, &beta,
				&devPtrVecV[iStageCumulNodes], nv, iStageNodes));

		if(iStage < N-1){
			// v=Theta*q+v
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nx,
					&alpha, (const real_t**)&ptrMyEngine->getPtrMatTheta()[iStageCumulNodes], nv,
					(const real_t**)devPtrVecQ, nx, &alpha, &devPtrVecV[iStageCumulNodes], nv, iStageNodes));
		}

		// v=Psi*psi+v
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nu, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatPsi()[iStageCumulNodes], nv,
				(const real_t**)&devPtrVecAcceleratedPsi[iStageCumulNodes], nu, &alpha, &devPtrVecV
				[iStageCumulNodes], nv, iStageNodes));

		// v=Phi*xi+v
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, 2*nx, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatPhi()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecAcceleratedXi[iStageCumulNodes], 2*nx, &alpha, &devPtrVecV[iStageCumulNodes],
				nv, iStageNodes));

		// r=sigma
		_CUDA(cudaMemcpy(devVecR, &ptrMyEngine->getMatSigma()[iStageCumulNodes*nv], nv*iStageNodes*sizeof(real_t),
				cudaMemcpyDeviceToDevice));

		// r=D*xi+r
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, 2*nx, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatD()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecAcceleratedXi[iStageCumulNodes], 2*nx, &alpha, devPtrVecR, nv, iStageNodes));

		// r=f*psi+r
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nu, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatF()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecAcceleratedPsi[iStageCumulNodes], nu, &alpha, devPtrVecR, nv, iStageNodes));

		if(iStage < N-1){
			// r=g*q+r
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrMatG()[iStageCumulNodes], nv, (const real_t**)devPtrVecQ,
					nx, &alpha, devPtrVecR, nv, iStageNodes));
		}

		if(iStage < N-1){
			// q=F'xi+q
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nx, 1, 2*nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrSysMatF()[iStageCumulNodes], 2*nx, (const real_t**)
					&devPtrVecAcceleratedXi[iStageCumulNodes], 2*nx, &alpha, devPtrVecQ, nx, iStageNodes));
		}else{
			// q=F'xi
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nx, 1, 2*nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrSysMatF()[iStageCumulNodes], 2*nx, (const real_t**)
					&devPtrVecAcceleratedXi[iStageCumulNodes], 2*nx, &beta, devPtrVecQ, nx, iStageNodes));
		}

		if(iStage > 0){
			prevStageNodes = nodesPerStage[iStage - 1];
			prevStageCumulNodes = nodesPerStageCumul[iStage - 1];
			if( (iStageNodes - prevStageNodes) > 0 ){
				solveSumChildren<<<prevStageNodes, nx>>>(devVecQ, devTempVecQ, ptrMyEngine->getTreeNumChildren(),
						ptrMyEngine->getTreeNumChildrenCumul(), prevStageCumulNodes, prevStageNodes, iStage - 1, nx);
				solveSumChildren<<<prevStageNodes, nv>>>(devVecR, devTempVecR, ptrMyEngine->getTreeNumChildren(),
						ptrMyEngine->getTreeNumChildrenCumul(), prevStageCumulNodes, prevStageNodes, iStage - 1 , nv);
				_CUDA(cudaMemcpy(devVecR, devTempVecR, prevStageNodes*nv*sizeof(real_t), cudaMemcpyDeviceToDevice));
				_CUDA(cudaMemcpy(devVecQ, devTempVecQ, prevStageNodes*nx*sizeof(real_t), cudaMemcpyDeviceToDevice));
			}
		}
	}

	// Forward substitution
	_CUDA(cudaMemcpy(devVecU, ptrMyEngine->getVecUhat(), nodes*nu*sizeof(real_t), cudaMemcpyDeviceToDevice));

	for(uint_t iStage = 0;iStage < N;iStage++){
		iStageNodes = nodesPerStage[iStage];
		iStageCumulNodes = nodesPerStageCumul[iStage];
		if(iStage == 0){
			// u = prevU - prevUhat
			_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu, &alpha, ptrMyEngine->getVecPreviousControl(), 1,
					devVecU, 1));
			_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu, &scale[2], ptrMyEngine->getVecPreviousUhat(), 1,
					devVecU, 1));
			// x=p
			_CUDA( cudaMemcpy(devVecX, ptrMyEngine->getVecCurrentState(), nx*sizeof(real_t), cudaMemcpyDeviceToDevice) );
			// x=x+w
			_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nx, &alpha, ptrMyEngine->getVecE(), 1, devVecX, 1));
			// u=Lv+\hat{u}
			_CUBLAS(cublasSgemv_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, nu, nv, &alpha,
					ptrMyEngine->getSysMatL(), nu, devVecV, 1, &alpha, devVecU, 1) );
			// x=x+Bu
			_CUBLAS(cublasSgemv_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, nx, nu, &alpha,
					ptrMyEngine->getSysMatB(), nx, devVecU, 1, &alpha, devVecX, 1) );
		}else{
			prevStageCumulNodes = nodesPerStageCumul[iStage - 1];
			if((nodesPerStage[iStage] - nodesPerStage[iStage-1]) > 0){
				// u_k=Lv_k+\hat{u}_k
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, iStageNodes, nv,
						&alpha, ptrMyEngine->getSysMatL(), nu, &devVecV[iStageCumulNodes*nv], nv, &alpha,
						&devVecU[iStageCumulNodes*nu], nu));
				// prevLv = u_{k-1} - uHat_{k-1}
				_CUDA( cudaMemcpy(devLv, &devVecU[prevStageCumulNodes*nu], nu*nodesPerStage[iStage-1]*sizeof(real_t), cudaMemcpyDeviceToDevice));
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodesPerStage[iStage-1], &scale[2],
						&ptrMyEngine->getVecUhat()[prevStageCumulNodes*nu], 1, devLv, 1));
				// u_{k} = u_{k} + prevLu
				solveChildNodesUpdate<<<iStageNodes, nu>>>(devLv, &devVecU[iStageCumulNodes*nu], ptrMyEngine->getTreeAncestor(),
						iStageCumulNodes, nu);
				// x=w
				_CUDA(cudaMemcpy(&devVecX[iStageCumulNodes*nx], &ptrMyEngine->getVecE()[iStageCumulNodes*nx],
						iStageNodes*nx*sizeof(real_t), cudaMemcpyDeviceToDevice));
				// x=x+Bu
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nx, iStageNodes, nu, &alpha,
						ptrMyEngine->getSysMatB(), nx, &devVecU[iStageCumulNodes*nu], nu, &alpha, &devVecX[iStageCumulNodes*nx], nx));
				// x_{k+1}=x_k
				solveChildNodesUpdate<<<iStageNodes, nx>>>(&devVecX[prevStageCumulNodes*nx], &devVecX[iStageCumulNodes*nx],
						ptrMyEngine->getTreeAncestor(), iStageCumulNodes, nx);
			}else{
				// u_k = u_{k-1} - uHat_{k-1}
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*iStageNodes, &alpha, &devVecU[prevStageCumulNodes*nu], 1,
						&devVecU[iStageCumulNodes*nu], 1));
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*iStageNodes, &scale[2], &ptrMyEngine->getVecUhat()
						[prevStageCumulNodes*nu], 1, &devVecU[iStageCumulNodes*nu], 1));
				// u_k=Lv_k+\hat{u}_k + u_k
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, iStageNodes, nv, &alpha,
						ptrMyEngine->getSysMatL(), nu, &devVecV[iStageCumulNodes*nv], nv, &alpha, &devVecU[iStageCumulNodes*nu], nu));
				// x_{k+1}=x_{k}
				_CUDA(cudaMemcpy(&devVecX[iStageCumulNodes*nx], &devVecX[prevStageCumulNodes*nx], nx*iStageNodes*sizeof(real_t),
						cudaMemcpyDeviceToDevice));
				// x=x+w
				_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nx*iStageNodes, &alpha, &ptrMyEngine->getVecE()
						[iStageCumulNodes*nx], 1, &devVecX[iStageCumulNodes*nx], 1));
				// x=x+Bu
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nx, iStageNodes, nu, &alpha,
						ptrMyEngine->getSysMatB(), nx, &devVecU[iStageCumulNodes*nu], nu, &alpha, &devVecX[iStageCumulNodes*nx], nx));
			}

		}
	}
	/**/

	_CUDA(cudaFree(devTempVecQ));
	_CUDA(cudaFree(devTempVecR));
	devTempVecQ = NULL;
	devTempVecR = NULL;
	ptrMyNetwork = NULL;
	ptrMyScenarioTree = NULL;
}

void SmpcController::proximalFunG(){
	DwnNetwork *ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree *ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	real_t alpha = 1;
	real_t negAlpha = -1;
	real_t beta = 0;
	real_t penaltyScalar;
	real_t invLambda = 1/stepSize;
	real_t distanceXs, distanceXcst;
	real_t *devSuffleVecXi;
	real_t *devVecDiffXi;
	_CUDA( cudaMalloc((void**)&devSuffleVecXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDiffXi, 2*nx*nodes*sizeof(real_t)) );

	// primalX = Hx
	_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, 2*nx, 1, nx, &alpha, (const real_t**)
			ptrMyEngine->getPtrSysMatF(), 2*nx, (const real_t**)devPtrVecX, nx, &beta, devPtrVecPrimalXi, 2*nx, nodes) );
	_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, 1, nu, &alpha, (const real_t**)
			ptrMyEngine->getPtrSysMatG(), nu, (const real_t**)devPtrVecU, nu, &beta, devPtrVecPrimalPsi, nu, nodes) );

	// Hx + \lambda^{-1}w
	_CUDA( cudaMemcpy(devVecDualXi, devVecPrimalXi, 2*nodes*nx*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUDA( cudaMemcpy(devVecDualPsi, devVecPrimalPsi, nodes*nu*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &invLambda, devVecAcceleratedXi, 1, devVecDualXi, 1) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &invLambda, devVecAcceleratedPsi, 1, devVecDualPsi, 1) );

	_CUDA( cudaMemcpy(devVecDiffXi, devVecDualXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	// proj(xi|X), proj(xi|Xsafe)
	projectionBox<<<nodes, nx>>>(devVecDualXi, ptrMyEngine->getSysXmin(), ptrMyEngine->getSysXmax(), 2*nx, 0, nx*nodes);
	projectionBox<<<nodes, nx>>>(devVecDualXi, ptrMyEngine->getSysXs(), ptrMyEngine->getSysXsUpper(), 2*nx, nx, nx*nodes);

	// x-proj(x)
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &negAlpha, devVecDualXi, 1, devVecDiffXi, 1) );
	shuffleVector<<<nodes, 2*nx>>>(devSuffleVecXi, devVecDiffXi, nx, 2, nodes);
	//distance with constraints X
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nx*nodes, devSuffleVecXi, 1, &distanceXcst));
	if(distanceXcst > invLambda*ptrMySmpcConfig->getPenaltyState()){
		penaltyScalar = 1 - invLambda*ptrMySmpcConfig->getPenaltyState()/distanceXcst;
		additionVectorOffset<<<nodes, nx>>>(devVecDualXi, devVecDiffXi, penaltyScalar, 2*nx, 0, nx*nodes);
	}
	//distance with Xsafe
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nx*nodes, &devSuffleVecXi[nx*nodes], 1, &distanceXs));
	if(distanceXs > invLambda*ptrMySmpcConfig->getPenaltySafety()){
		penaltyScalar = 1-invLambda*ptrMySmpcConfig->getPenaltySafety()/distanceXs;
		additionVectorOffset<<<nodes, nx>>>(devVecDualXi, devVecDiffXi, penaltyScalar, 2*nx, nx, nx*nodes);
	}
	/**/
	projectionBox<<<nodes, nu>>>(devVecDualPsi, ptrMyEngine->getSysUmin(), ptrMyEngine->getSysUmax(), nu, 0, nu*nodes);
	_CUDA( cudaFree(devSuffleVecXi) );
	_CUDA( cudaFree(devVecDiffXi) );
	devSuffleVecXi = NULL;
	devVecDiffXi = NULL;
	ptrMyNetwork = NULL;
	ptrMyScenarioTree = NULL;
}

void SmpcController::dualUpdate(){
	DwnNetwork *ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree *ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
	real_t negAlpha = -1;
	//Hx - z
	_CUDA(cudaMemcpy(devVecPrimalInfeasibilty, devVecPrimalXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUDA(cudaMemcpy(&devVecPrimalInfeasibilty[2*nx*nodes], devVecPrimalPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &negAlpha, devVecDualXi, 1, devVecPrimalInfeasibilty, 1));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &negAlpha, devVecDualPsi, 1, &devVecPrimalInfeasibilty[2*nx*nodes], 1));
	// y = w + \lambda(Hx - z)
	_CUDA( cudaMemcpy(devVecUpdateXi, devVecAcceleratedXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUDA( cudaMemcpy(devVecUpdatePsi, devVecAcceleratedPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &stepSize, devVecPrimalInfeasibilty, 1, devVecUpdateXi, 1) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &stepSize, &devVecPrimalInfeasibilty[2*nx*nodes], 1,
			devVecUpdatePsi, 1) );

}


uint_t SmpcController::algorithmApg(){
	DwnNetwork *ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree *ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();
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

	real_t theta[2] = {1, 1};
	real_t lambda;
	uint_t maxIndex;

	for (uint_t iter = 0; iter < ptrMySmpcConfig->getMaxIterations(); iter++){
		lambda = theta[1]*(1/theta[0] - 1);
		dualExtrapolationStep(lambda);
		solveStep();
		proximalFunG();
		dualUpdate();
		theta[0] = theta[1];
		theta[1] = 0.5*(sqrt(pow(theta[0], 4) + 4*theta[0]) - pow(theta[0], 2));
		_CUBLAS( cublasIsamax_v2(ptrMyEngine->getCublasHandle(), 2*(nx + nu)*nodes, devVecPrimalInfeasibilty,
				1, &maxIndex));
		_CUDA( cudaMemcpy(&vecPrimalInfs[iter], &devVecPrimalInfeasibilty[maxIndex - 1], sizeof(real_t),
				cudaMemcpyDeviceToHost));
	}
	return 1;
}

void SmpcController::controllerSmpc(){
	ptrMyEngine->updateStateControl(ptrMySmpcConfig->getCurrentX(), ptrMySmpcConfig->getPrevU(),
			ptrMySmpcConfig->getPrevDemand());
	ptrMyEngine->eliminateInputDistubanceCoupling(ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	algorithmApg();
}

uint_t SmpcController::controlAction(real_t* u){
	uint_t status;
	//real_t hostPrimalInfs = new real_t[];
	ptrMyEngine->updateStateControl(ptrMySmpcConfig->getCurrentX(), ptrMySmpcConfig->getPrevU(),
			ptrMySmpcConfig->getPrevDemand());
	ptrMyEngine->eliminateInputDistubanceCoupling(ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	status = algorithmApg();
	_CUDA( cudaMemcpy(u, devVecU, ptrMySmpcConfig->getNU()*sizeof(real_t), cudaMemcpyDeviceToHost));
	return status;
}

/**
 * Get's the network object
 * @return  DwnNetwork
 */
DwnNetwork* SmpcController::getDwnNetwork(){
	return ptrMyEngine->getDwnNetwork();
}
/**
 * Get's the scenario tree object
 * @return scenarioTree
 */
ScenarioTree* SmpcController::getScenarioTree(){
	return ptrMyEngine->getScenarioTree();
}
/**
 * Get's the forecaster object
 * @return Forecaster
 */
Forecaster* SmpcController::getForecaster(){
	return ptrMyForecaster;
}
/**
 * Get's the Smpc controller configuration object
 * @return SmpcConfiguration
 */
SmpcConfiguration* SmpcController::getSmpcConfiguration(){
	return ptrMySmpcConfig;
}
/**
 * Get's the Engine object
 * @return Engine
 */
Engine* SmpcController::getEngine(){
	return ptrMyEngine;
}

SmpcController::~SmpcController(){
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
	_CUDA( cudaFree(devVecPrimalInfeasibilty) );

	_CUDA( cudaFree(devPtrVecX) );
	_CUDA( cudaFree(devPtrVecU) );
	_CUDA( cudaFree(devPtrVecV) );
	_CUDA( cudaFree(devPtrVecAcceleratedXi) );
	_CUDA( cudaFree(devPtrVecAcceleratedPsi) );
	_CUDA( cudaFree(devPtrVecPrimalXi) );
	_CUDA( cudaFree(devPtrVecPrimalPsi) );
	_CUDA( cudaFree(devPtrVecQ));
	_CUDA( cudaFree(devPtrVecR));

	free(vecPrimalInfs);
	devVecX = NULL;
	devVecU = NULL;
	devVecV = NULL;
	devVecXi = NULL;
	devVecPsi = NULL;
	devVecAcceleratedXi = NULL;
	devVecAcceleratedPsi = NULL;
	devVecPrimalXi = NULL;
	devVecPrimalPsi = NULL;
	devVecDualXi = NULL;
	devVecDualPsi = NULL;
	devVecUpdateXi = NULL;
	devVecUpdatePsi = NULL;
	devVecPrimalInfeasibilty = NULL;

	devPtrVecX = NULL;
	devPtrVecU = NULL;
	devPtrVecV = NULL;
	devPtrVecAcceleratedXi = NULL;
	devPtrVecAcceleratedPsi = NULL;
	devPtrVecPrimalXi = NULL;
	devPtrVecPrimalPsi = NULL;
	devPtrVecQ = NULL;
	devPtrVecR = NULL;

	vecPrimalInfs = NULL;
}
