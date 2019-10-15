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
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
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
	factorStepFlag = false;
	simulatorFlag = true;
	bool apgStatus = ptrMyEngine->getApgFlag();
	bool globalFbeStatus = ptrMyEngine->getGlobalFbeFlag();
	bool namaStatus = ptrMyEngine->getNamaFlag();

	vecPrimalInfs = new real_t[ptrMySmpcConfig->getMaxIterations() + 1];
	vecValueFbe = new real_t[ptrMySmpcConfig->getMaxIterations()];
	vecTau = new real_t[ptrMySmpcConfig->getMaxIterations()];

	economicKpi = 0;
	smoothKpi = 0;
	safeKpi = 0;
	networkKpi = 0;

	allocateSmpcController();

	if( globalFbeStatus ){
		allocateGlobalFbeAlgorithm();
	}else if( namaStatus ){
		allocateNamaAlgorithm();
	}else if( apgStatus ){
		allocateApgAlgorithm();
	}else{
		cerr << "No algorithm is selected flag is set " << __FILE__ << "\nLine = " << __LINE__;
	}

	//initialiseAlgorithmSpecificData();
}



SmpcController::SmpcController(string pathToConfigFile){
	ptrMySmpcConfig = new SmpcConfiguration( pathToConfigFile );
	string pathToForecaster = ptrMySmpcConfig->getPathToForecaster();
	ptrMyForecaster = new Forecaster( pathToForecaster );
	ptrMyEngine = new Engine( ptrMySmpcConfig );

	stepSize = ptrMySmpcConfig->getStepSize();
	factorStepFlag = false;
	simulatorFlag = true;
	bool apgStatus = ptrMyEngine->getApgFlag();
	bool globalFbeStatus = ptrMyEngine->getGlobalFbeFlag();
	bool namaStatus = ptrMyEngine->getNamaFlag();

	vecPrimalInfs = new real_t[ptrMySmpcConfig->getMaxIterations() + 1];
	vecValueFbe = new real_t[ptrMySmpcConfig->getMaxIterations()];
	vecTau = new real_t[ptrMySmpcConfig->getMaxIterations()];

	economicKpi = 0;
	smoothKpi = 0;
	safeKpi = 0;
	networkKpi = 0;

	allocateSmpcController();

	if( globalFbeStatus ){
		allocateGlobalFbeAlgorithm();
	}else if( namaStatus ){
		allocateNamaAlgorithm();
	}else if( apgStatus ){
		allocateApgAlgorithm();
	}else{
		cerr << "No algorithm is selected flag is set " << __FILE__ << "\nLine = " << __LINE__;
	}

	//initialiseAlgorithmSpecificData();
}



void SmpcController::allocateSmpcController(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	_CUDA( cudaMalloc((void**)&devVecX, nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecU, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecV, nv*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrimalXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrimalPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecFixedPointResidualXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecFixedPointResidualPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDualXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDualPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecAcceleratedXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecAcceleratedPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecQ, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecR, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devControlAction, nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devStateUpdate, nx*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrVecX, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecU, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecV, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalPsi, nodes*sizeof(real_t*)) );

	_CUDA( cudaMalloc((void**)&devPtrVecSolveStepXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecSolveStepPsi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecQ, ns*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecR, ns*sizeof(real_t*)) );

	ptrProximalXi = new real_t*[1];
	ptrProximalPsi = new real_t*[1];

	real_t** ptrVecX = new real_t*[nodes];
	real_t** ptrVecU = new real_t*[nodes];
	real_t** ptrVecV = new real_t*[nodes];
	real_t** ptrVecPrimalXi = new real_t*[nodes];
	real_t** ptrVecPrimalPsi = new real_t*[nodes];
	real_t** ptrVecQ = new real_t*[ns];
	real_t** ptrVecR = new real_t*[ns];


	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecX[iNode] = &devVecX[iNode*nx];
		ptrVecU[iNode] = &devVecU[iNode*nu];
		ptrVecV[iNode] = &devVecV[iNode*nv];
		ptrVecPrimalXi[iNode] = &devVecPrimalXi[2*iNode*nx];
		ptrVecPrimalPsi[iNode] = &devVecPrimalPsi[iNode*nu];
	}
	for(uint_t iScenario = 0; iScenario < ns; iScenario++){
		ptrVecQ[iScenario] = &devVecQ[iScenario*nx];
		ptrVecR[iScenario] = &devVecR[iScenario*nv];
	}

	_CUDA( cudaMemset(devVecU, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecDualXi, 0, 2*nx*nodes*sizeof(real_t)));
	_CUDA( cudaMemset(devVecDualPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecAcceleratedXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecAcceleratedPsi, 0, nu*nodes*sizeof(real_t)) );
	//_CUDA( cudaMemset(devVecResidual, 0, (2*nx + nu)*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecFixedPointResidualXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecFixedPointResidualPsi, 0, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMemset(devVecR, 0, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecQ, 0, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMemset(devControlAction, 0, nu*sizeof(real_t)) );
	_CUDA( cudaMemset(devStateUpdate, 0, nx*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecX, ptrVecX, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecU, ptrVecU, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecV, ptrVecV, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalXi, ptrVecPrimalXi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalPsi, ptrVecPrimalPsi, nodes*sizeof(real_t*),cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecQ, ptrVecQ, ns*sizeof(real_t*), cudaMemcpyHostToDevice));
	_CUDA( cudaMemcpy(devPtrVecR, ptrVecR, ns*sizeof(real_t*), cudaMemcpyHostToDevice));

	delete [] ptrVecX;
	delete [] ptrVecU;
	delete [] ptrVecV;
	delete [] ptrVecPrimalXi;
	delete [] ptrVecPrimalPsi;
	delete [] ptrVecQ;
	delete [] ptrVecR;
	ptrVecX = NULL;
	ptrVecU = NULL;
	ptrVecV = NULL;
	ptrVecPrimalXi = NULL;
	ptrVecPrimalPsi = NULL;
	ptrVecQ = NULL;
	ptrVecR = NULL;

}



void SmpcController::allocateApgAlgorithm(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	_CUDA( cudaMalloc((void**)&devVecUpdateXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecUpdatePsi, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMemset(devVecUpdateXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecUpdatePsi, 0, nu*nodes*sizeof(real_t)) );
}


void SmpcController::allocateLbfgsBuffer(){
	uint_t nx = ptrMySmpcConfig->getNX();
	uint_t nu = ptrMySmpcConfig->getNU();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t bufferSize = ptrMySmpcConfig->getLbfgsBufferSize();
	uint_t ny = 2*nx + nu;

	lbfgsBufferCol = 0;
	lbfgsBufferMemory = 0;
	lbfgsSkipCount = 0;
	lbfgsBufferHessian = 1;

	_CUDA( cudaMalloc((void**)&devLbfgsBufferMatS, ny*nodes*bufferSize*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devLbfgsBufferMatY, ny*nodes*bufferSize*sizeof(real_t)) );

	lbfgsBufferRho = new real_t[bufferSize];

	ptrLbfgsCurrentYvecXi = new real_t*[1];
	ptrLbfgsCurrentYvecPsi = new real_t*[1];
	ptrLbfgsPreviousYvecXi = new real_t*[1];
	ptrLbfgsPreviousYvecPsi = new real_t*[1];

	_CUDA( cudaMemset(devLbfgsBufferMatS, 0, ny*nodes*bufferSize*sizeof(real_t)) );
	_CUDA( cudaMemset(devLbfgsBufferMatY, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemcpy(lbfgsBufferRho, devLbfgsBufferMatY, bufferSize*sizeof(real_t), cudaMemcpyDeviceToHost));
}



void SmpcController::allocateLbfgsDirection(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	allocateLbfgsBuffer();

	_CUDA( cudaMalloc((void**)&devVecPrevXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrevPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecLbfgsDirXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecLbfgsDirPsi, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecXdir, nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecUdir, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrimalXiDir, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecPrimalPsiDir, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMalloc((void**)&devPtrVecXdir, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecUdir, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalXiDir, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecPrimalPsiDir, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecHessianOracleXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecHessianOraclePsi, nodes*sizeof(real_t*)) );

	real_t** ptrVecXdir = new real_t*[nodes];
	real_t** ptrVecUdir = new real_t*[nodes];
	real_t** ptrVecPrimalXiDir = new real_t*[nodes];
	real_t** ptrVecPrimalPsiDir =  new real_t*[nodes];

	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecXdir[iNode] = &devVecXdir[iNode*nx];
		ptrVecUdir[iNode] = &devVecUdir[iNode*nu];
		ptrVecPrimalXiDir[iNode] = &devVecPrimalXiDir[2*iNode*nx];
		ptrVecPrimalPsiDir[iNode] = &devVecPrimalPsiDir[iNode*nu];
	}

	_CUDA( cudaMemset(devVecPrevXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrevPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecLbfgsDirXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecLbfgsDirPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecXdir, 0, nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecUdir, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalXiDir, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalPsiDir, 0, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecXdir, ptrVecXdir, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecUdir, ptrVecUdir, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalXiDir, ptrVecPrimalXiDir, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecPrimalPsiDir, ptrVecPrimalPsiDir, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );

	delete [] ptrVecXdir;
	delete [] ptrVecUdir;
	delete [] ptrVecPrimalXiDir;
	delete [] ptrVecPrimalPsiDir;

	ptrVecXdir = NULL;
	ptrVecUdir = NULL;
	ptrVecPrimalXiDir = NULL;
	ptrVecPrimalPsiDir = NULL;

}



void SmpcController::allocateGlobalFbeAlgorithm(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	allocateLbfgsDirection();

	_CUDA( cudaMalloc((void**)&devVecGradientFbeXi, 2*nx*nodes*sizeof(real_t)));
	_CUDA( cudaMalloc((void**)&devVecGradientFbePsi, nu*nodes*sizeof(real_t)));
	_CUDA( cudaMalloc((void**)&devVecPrevGradientFbeXi, 2*nx*nodes*sizeof(real_t)));
	_CUDA( cudaMalloc((void**)&devVecPrevGradientFbePsi, nu*nodes*sizeof(real_t)));

	_CUDA( cudaMalloc((void**)&devPtrVecGradFbeXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecGradFbePsi, nodes*sizeof(real_t*)) );

	real_t** ptrVecGradFbeXi = new real_t*[nodes];
	real_t** ptrVecGradFbePsi = new real_t*[nodes];

	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecGradFbeXi[iNode] = &devVecGradientFbeXi[2*iNode*nx];
		ptrVecGradFbePsi[iNode] = &devVecGradientFbePsi[iNode*nu];
	}

	_CUDA( cudaMemset(devVecGradientFbeXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecGradientFbePsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrevGradientFbeXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrevGradientFbePsi, 0, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecGradFbeXi, ptrVecGradFbeXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecGradFbePsi, ptrVecGradFbePsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecHessianOracleXi, ptrVecGradFbeXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecHessianOraclePsi, ptrVecGradFbePsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );

	delete [] ptrVecGradFbeXi;
	delete [] ptrVecGradFbePsi;

	ptrVecGradFbeXi = NULL;
	ptrVecGradFbePsi = NULL;
}



void SmpcController::allocateNamaAlgorithm(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	allocateLbfgsDirection();

	_CUDA( cudaMalloc((void**)&devVecPrevFixedPointResidualXi, 2*nx*nodes*sizeof(real_t)));
	_CUDA( cudaMalloc((void**)&devVecPrevFixedPointResidualPsi, nu*nodes*sizeof(real_t)));

	_CUDA( cudaMalloc((void**)&devPtrVecFixedPointResidualXi, nodes*sizeof(real_t*)) );
	_CUDA( cudaMalloc((void**)&devPtrVecFixedPointResidualPsi, nodes*sizeof(real_t*)) );

	real_t** ptrVecFixedPointResidualXi = new real_t*[nodes];
	real_t** ptrVecFixedPointResidualPsi = new real_t*[nodes];

	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecFixedPointResidualXi[iNode] = &devVecFixedPointResidualXi[2*iNode*nx];
		ptrVecFixedPointResidualPsi[iNode] = &devVecFixedPointResidualPsi[iNode*nu];
	}

	_CUDA( cudaMemset(devVecPrevFixedPointResidualXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrevFixedPointResidualPsi, 0, nu*nodes*sizeof(real_t)) );

	_CUDA( cudaMemcpy(devPtrVecHessianOracleXi, ptrVecFixedPointResidualXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecHessianOraclePsi, ptrVecFixedPointResidualPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );

	delete [] ptrVecFixedPointResidualXi;
	delete [] ptrVecFixedPointResidualPsi;

	ptrVecFixedPointResidualXi = NULL;
	ptrVecFixedPointResidualPsi = NULL;
}



void SmpcController::initialiseAlgorithm(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();

	_CUDA( cudaMemset(devVecXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecAcceleratedXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecAcceleratedPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalXi, 0, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecPrimalPsi, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecDualXi, 0, 2*nx*nodes*sizeof(real_t)));
	_CUDA( cudaMemset(devVecDualPsi, 0, nu*nodes*sizeof(real_t)) );


	if( ptrMyEngine->getApgFlag()){
		_CUDA( cudaMemset(devVecUpdateXi, 0, 2*nx*nodes*sizeof(real_t)) );
		_CUDA( cudaMemset(devVecUpdatePsi, 0, nu*nodes*sizeof(real_t)) );
	}
	if(ptrMyEngine->getGlobalFbeFlag()){
		_CUDA( cudaMemset(devVecPrevXi, 0, 2*nx*nodes*sizeof(real_t)) );
		_CUDA( cudaMemset(devVecPrevPsi, 0, nu*nodes*sizeof(real_t)) );
		_CUDA( cudaMemset(devVecGradientFbeXi, 0, 2*nx*nodes*sizeof(real_t)));
		_CUDA( cudaMemset(devVecGradientFbePsi, 0, nu*nodes*sizeof(real_t)));
	}else if (ptrMyEngine->getNamaFlag()){
		_CUDA( cudaMemset(devVecPrevXi, 0, 2*nx*nodes*sizeof(real_t)) );
		_CUDA( cudaMemset(devVecPrevPsi, 0, nu*nodes*sizeof(real_t)) );
		_CUDA( cudaMemset(devVecFixedPointResidualXi, 0, 2*nx*nodes*sizeof(real_t)));
		_CUDA( cudaMemset(devVecFixedPointResidualPsi, 0, nu*nodes*sizeof(real_t)));
	}
}


void SmpcController::initaliseLbfgBuffer(){
	uint_t nx = ptrMySmpcConfig->getNX();
	uint_t nu = ptrMySmpcConfig->getNU();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t bufferSize = ptrMySmpcConfig->getLbfgsBufferSize();
	uint_t ny = 2*nx + nu;

	lbfgsBufferCol = 0;
	lbfgsBufferMemory = 0;
	lbfgsSkipCount = 0;
	lbfgsBufferHessian = 1;

	_CUDA( cudaMemset(devLbfgsBufferMatS, 0, ny*nodes*bufferSize*sizeof(real_t)) );
	_CUDA( cudaMemset(devLbfgsBufferMatY, 0, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemcpy(lbfgsBufferRho, devLbfgsBufferMatY, bufferSize*sizeof(real_t), cudaMemcpyDeviceToDevice));
}


/**
 * Performs the initialise the smpc controller
 *   - update the current state and previous controls in the device memory
 *   - perform the factor step
 */
void SmpcController::initialiseSmpcController(){
	factorStepFlag = true;
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	initialiseAlgorithmSpecificData();
}

/**
 * Function to copy the vector for solve step
 * apg - accelerated dual gradient
 * fbe - dual vector
 */
void SmpcController::initialiseAlgorithmSpecificData(){
	ScenarioTree* ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();

	real_t** ptrDevLbfgsYvec;
	real_t** ptrVecAcceleratedXi = new real_t*[nodes];
	real_t** ptrVecAcceleratedPsi = new real_t*[nodes];
	for(uint_t iNode = 0; iNode < nodes; iNode++){
		ptrVecAcceleratedXi[iNode] = &devVecAcceleratedXi[2*iNode*nx];
		ptrVecAcceleratedPsi[iNode] = &devVecAcceleratedPsi[iNode*nu];
	}

	_CUDA( cudaMemcpy(devPtrVecSolveStepXi, ptrVecAcceleratedXi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	_CUDA( cudaMemcpy(devPtrVecSolveStepPsi, ptrVecAcceleratedPsi, nodes*sizeof(real_t*), cudaMemcpyHostToDevice) );
	ptrProximalXi[0] = &devVecAcceleratedXi[0];
	ptrProximalPsi[0] = &devVecAcceleratedPsi[0];

	if (ptrMyEngine->getGlobalFbeFlag()){
		ptrLbfgsCurrentYvecXi[0] = &devVecGradientFbeXi[0];
		ptrLbfgsCurrentYvecPsi[0] = &devVecGradientFbePsi[0];
		ptrLbfgsPreviousYvecXi[0] = &devVecPrevGradientFbeXi[0];
		ptrLbfgsPreviousYvecPsi[0] = &devVecPrevGradientFbePsi[0];
	}else if (ptrMyEngine->getNamaFlag()){
		ptrLbfgsCurrentYvecXi[0] = &devVecFixedPointResidualXi[0];
		ptrLbfgsCurrentYvecPsi[0] = &devVecFixedPointResidualPsi[0];
		ptrLbfgsPreviousYvecXi[0] = &devVecPrevFixedPointResidualXi[0];
		ptrLbfgsPreviousYvecPsi[0] = &devVecPrevFixedPointResidualPsi[0];
	}
	ptrDevLbfgsYvec = NULL;

}

/**
 * Performs the dual extrapolation step with given parameter.
 * @param extrapolation parameter.
 */
void SmpcController::dualExtrapolationStep(real_t lambda){
	DwnNetwork* ptrMyNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree* ptrMyScenarioTree = ptrMyEngine->getScenarioTree();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	uint_t nx = ptrMyNetwork->getNumTanks();
	uint_t nu = ptrMyNetwork->getNumControls();

	if( ptrMyEngine->getApgFlag()){
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
	}else{
		// y_{update} = y_{k}
		_CUDA(cudaMemcpy(devVecAcceleratedXi, devVecXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA(cudaMemcpy(devVecAcceleratedPsi, devVecPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	}
}

/**
 * Computes the dual gradient.This is the main computational
 * algorithm for the proximal gradient algorithm
 */
void SmpcController::solveStep(){
	real_t *devTempVecR, *devTempVecQ, *devLv;
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t N =  ptrMyEngine->getScenarioTree()->getPredHorizon();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t *nodesPerStage = ptrMyEngine->getScenarioTree()->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyEngine->getScenarioTree()->getNodesPerStageCumul();
	uint_t iStageCumulNodes, iStageNodes, prevStageNodes, prevStageCumulNodes;
	real_t scale[3] = {-0.5, 1, -1};
	real_t alpha = 1;
	real_t beta = 0;
	real_t *xHost = new real_t[nu*nodes];

	if(factorStepFlag == false){
		initialiseSmpcController();
		factorStepFlag = true;
	}

	_CUDA( cudaMalloc((void**)&devTempVecQ, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devTempVecR, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devLv, ns*nu*sizeof(real_t)) );
	_CUDA( cudaMemcpy(ptrMyEngine->getMatSigma(), ptrMyEngine->getVecBeta(), nv*nodes*sizeof(real_t),
			cudaMemcpyDeviceToDevice) );

	real_t *x = new real_t[ns*nv*nv];

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
				(const real_t**)&devPtrVecSolveStepPsi[iStageCumulNodes], nu, &alpha, &devPtrVecV
				[iStageCumulNodes], nv, iStageNodes));

		// v=Phi*xi+v
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, 2*nx, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatPhi()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecSolveStepXi[iStageCumulNodes], 2*nx, &alpha, &devPtrVecV[iStageCumulNodes],
				nv, iStageNodes));

		// r=sigma
		_CUDA(cudaMemcpy(devVecR, &ptrMyEngine->getMatSigma()[iStageCumulNodes*nv], nv*iStageNodes*sizeof(real_t),
				cudaMemcpyDeviceToDevice));

		// r=D*xi+r
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, 2*nx, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatD()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecSolveStepXi[iStageCumulNodes], 2*nx, &alpha, devPtrVecR, nv, iStageNodes));

		// r=f*psi+r
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nu, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatF()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecSolveStepPsi[iStageCumulNodes], nu, &alpha, devPtrVecR, nv, iStageNodes));

		if(iStage < N-1){
			// r=g*q+r
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrMatG()[0], nv, (const real_t**)devPtrVecQ,
					nx, &alpha, devPtrVecR, nv, iStageNodes));
		}

		if(iStage < N-1){
			// q=F'xi+q
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nx, 1, 2*nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrSysMatF()[iStageCumulNodes], 2*nx, (const real_t**)
					&devPtrVecSolveStepXi[iStageCumulNodes], 2*nx, &alpha, devPtrVecQ, nx, iStageNodes));
		}else{
			// q=F'xi
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nx, 1, 2*nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrSysMatF()[iStageCumulNodes], 2*nx, (const real_t**)
					&devPtrVecSolveStepXi[iStageCumulNodes], 2*nx, &beta, devPtrVecQ, nx, iStageNodes));
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

	// primalX = Hx
	_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, 2*nx, 1, nx, &alpha, (const real_t**)
			ptrMyEngine->getPtrSysMatF(), 2*nx, (const real_t**)devPtrVecX, nx, &beta, devPtrVecPrimalXi, 2*nx, nodes) );
	_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, 1, nu, &alpha, (const real_t**)
			ptrMyEngine->getPtrSysMatG(), nu, (const real_t**)devPtrVecU, nu, &beta, devPtrVecPrimalPsi, nu, nodes) );

	_CUDA(cudaFree(devTempVecQ));
	_CUDA(cudaFree(devTempVecR));
	_CUDA(cudaFree(devLv) );
	devTempVecQ = NULL;
	devTempVecR = NULL;
	devLv = NULL;
}



void SmpcController::proximalFunG(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	real_t alpha = 1;
	real_t negAlpha = -1;
	real_t beta = 0;
	real_t penaltyScalar;
	real_t invLambda = 1/stepSize;
	real_t distanceXs, distanceXcst;
	real_t *devSuffleVecXi;
	real_t *devVecDiffXi;
	real_t flagCalculateDistance = 0;
	_CUDA( cudaMalloc((void**)&devSuffleVecXi, 2*nx*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDiffXi, 2*nx*nodes*sizeof(real_t)) );



	// Hx + \lambda^{-1}w
	_CUDA( cudaMemcpy(devVecDualXi, devVecPrimalXi, 2*nodes*nx*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUDA( cudaMemcpy(devVecDualPsi, devVecPrimalPsi, nodes*nu*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &invLambda, ptrProximalXi[0], 1, devVecDualXi, 1) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &invLambda, ptrProximalPsi[0], 1, devVecDualPsi, 1) );

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
		flagCalculateDistance = 1;
		penaltyScalar = 1 - invLambda*ptrMySmpcConfig->getPenaltyState()/distanceXcst;
		additionVectorOffset<<<nodes, nx>>>(devVecDualXi, devVecDiffXi, penaltyScalar, 2*nx, 0, nx*nodes);
	}
	// calculating the cost of g(xBox)
	if(flagCalculateDistance){
		_CUDA( cudaMemcpy( devVecDiffXi, devVecDualXi, 2*nx*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		projectionBox<<<nodes, nx>>>(devVecDiffXi, ptrMyEngine->getSysXmin(), ptrMyEngine->getSysXmax(), 2*nx, 0, nx*nodes);
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &negAlpha, devVecDualXi, 1, devVecDiffXi, 1) );
		_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecDiffXi, 1, &valueFunGxBox));
		valueFunGxBox = ptrMySmpcConfig->getPenaltyState()*valueFunGxBox;
		flagCalculateDistance = 0;
	}else
		valueFunGxBox = 0;

	//distance with Xsafe
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nx*nodes, &devSuffleVecXi[nx*nodes], 1, &distanceXs));
	if(distanceXs > invLambda*ptrMySmpcConfig->getPenaltySafety()){
		flagCalculateDistance = 1;
		penaltyScalar = 1-invLambda*ptrMySmpcConfig->getPenaltySafety()/distanceXs;
		additionVectorOffset<<<nodes, nx>>>(devVecDualXi, devVecDiffXi, penaltyScalar, 2*nx, nx, nx*nodes);
	}
	// calculating the cost of g(xSafe)
	if(flagCalculateDistance){
		_CUDA( cudaMemcpy( devVecDiffXi, devVecDualXi, 2*nx*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		projectionBox<<<nodes, nx>>>(devVecDiffXi, ptrMyEngine->getSysXs(), ptrMyEngine->getSysXsUpper(), 2*nx, nx, nx*nodes);
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &negAlpha, devVecDualXi, 1, devVecDiffXi, 1) );
		_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecDiffXi, 1, &valueFunGxSafe));
		valueFunGxSafe = ptrMySmpcConfig->getPenaltySafety()*valueFunGxSafe;
		flagCalculateDistance = 0;
	}else
		valueFunGxSafe = 0;

	projectionBox<<<nodes, nu>>>(devVecDualPsi, ptrMyEngine->getSysUmin(), ptrMyEngine->getSysUmax(), nu, 0, nu*nodes);
	valueFunGuBox = 0;


	_CUDA( cudaFree(devSuffleVecXi) );
	_CUDA( cudaFree(devVecDiffXi) );
	devSuffleVecXi = NULL;
	devVecDiffXi = NULL;
}



void SmpcController::computeFixedPointResidual(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nDualx = 2*ptrMyEngine->getDwnNetwork()->getNumTanks()*nodes;
	uint_t nDualu = ptrMyEngine->getDwnNetwork()->getNumControls()*nodes;
	real_t negAlpha = -1;

	//Hx - z
	_CUDA(cudaMemcpy(devVecFixedPointResidualXi, devVecPrimalXi, nDualx*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUDA(cudaMemcpy(devVecFixedPointResidualPsi, devVecPrimalPsi, nDualu*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualx, &negAlpha, devVecDualXi, 1, devVecFixedPointResidualXi, 1));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualu, &negAlpha, devVecDualPsi, 1, devVecFixedPointResidualPsi, 1));
}



void SmpcController::dualUpdate(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();

	if(ptrMyEngine->getGlobalFbeFlag()){
		// gradFbePrev = gradFbe
		_CUDA( cudaMemcpy(devVecPrevGradientFbeXi, devVecGradientFbeXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA( cudaMemcpy(devVecPrevGradientFbePsi, devVecGradientFbePsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		// yOld = y
		_CUDA( cudaMemcpy(devVecPrevXi, devVecXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA( cudaMemcpy(devVecPrevPsi, devVecPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		// y = w + \lambda(Hz - t)
		_CUDA( cudaMemcpy(devVecXi, devVecAcceleratedXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA( cudaMemcpy(devVecPsi, devVecAcceleratedPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &stepSize, devVecFixedPointResidualXi, 1, devVecXi, 1) );
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &stepSize, devVecFixedPointResidualPsi, 1, devVecPsi, 1) );

	}else{
		// y = w + \lambda(Hx - z)
		_CUDA( cudaMemcpy(devVecUpdateXi, devVecAcceleratedXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA( cudaMemcpy(devVecUpdatePsi, devVecAcceleratedPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &stepSize, devVecFixedPointResidualXi, 1, devVecUpdateXi, 1) );
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &stepSize, devVecFixedPointResidualPsi, 1, devVecUpdatePsi, 1) );
	}
}


void SmpcController::computeHessianOracalGlobalFbe(){
	real_t *devTempVecR, *devTempVecQ, *devLv;
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t N =  ptrMyEngine->getScenarioTree()->getPredHorizon();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t *nodesPerStage = ptrMyEngine->getScenarioTree()->getNodesPerStage();
	uint_t *nodesPerStageCumul = ptrMyEngine->getScenarioTree()->getNodesPerStageCumul();
	uint_t iStageCumulNodes, iStageNodes, prevStageNodes, prevStageCumulNodes;
	real_t scale[3] = {-0.5, 1, -1};
	real_t alpha = 1;
	real_t beta = 0;

	if(factorStepFlag == false){
		initialiseSmpcController();
		factorStepFlag = true;
	}

	_CUDA( cudaMalloc((void**)&devTempVecQ, ns*nx*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devTempVecR, ns*nv*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devLv, ns*nu*sizeof(real_t)) );
	// sigma = 0
	_CUDA( cudaMemset(ptrMyEngine->getMatSigma(), 0, nv*nodes*sizeof(real_t)) );

	//Backward substitution
	for(uint_t iStage = N-1;iStage > -1;iStage--){
		iStageCumulNodes = nodesPerStageCumul[iStage];
		iStageNodes = nodesPerStage[iStage];

		if(iStage < N-1){
			// sigma = r
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
				(const real_t**)&devPtrVecHessianOraclePsi[iStageCumulNodes], nu, &alpha, &devPtrVecV
				[iStageCumulNodes], nv, iStageNodes));

		// v=Phi*xi+v
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, 2*nx, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatPhi()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecHessianOracleXi[iStageCumulNodes], 2*nx, &alpha, &devPtrVecV[iStageCumulNodes],
				nv, iStageNodes));

		// r=sigma
		_CUDA(cudaMemcpy(devVecR, &ptrMyEngine->getMatSigma()[iStageCumulNodes*nv], nv*iStageNodes*sizeof(real_t),
				cudaMemcpyDeviceToDevice));

		// r=D*xi+r
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, 2*nx, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatD()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecHessianOracleXi[iStageCumulNodes], 2*nx, &alpha, devPtrVecR, nv, iStageNodes));

		// r=f*psi+r
		_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nu, &alpha,
				(const real_t**)&ptrMyEngine->getPtrMatF()[iStageCumulNodes], nv, (const real_t**)
				&devPtrVecHessianOraclePsi[iStageCumulNodes], nu, &alpha, devPtrVecR, nv, iStageNodes));

		if(iStage < N-1){
			// r=g*q+r
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nv, 1, nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrMatG()[0], nv, (const real_t**)devPtrVecQ,
					nx, &alpha, devPtrVecR, nv, iStageNodes));
		}

		if(iStage < N-1){
			// q=F'xi+q
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nx, 1, 2*nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrSysMatF()[iStageCumulNodes], 2*nx, (const real_t**)
					&devPtrVecHessianOracleXi[iStageCumulNodes], 2*nx, &alpha, devPtrVecQ, nx, iStageNodes));
		}else{
			// q=F'xi
			_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, nx, 1, 2*nx, &alpha,
					(const real_t**)&ptrMyEngine->getPtrSysMatF()[iStageCumulNodes], 2*nx, (const real_t**)
					&devPtrVecHessianOracleXi[iStageCumulNodes], 2*nx, &beta, devPtrVecQ, nx, iStageNodes));
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
	for(uint_t iStage = 0;iStage < N;iStage++){
		iStageNodes = nodesPerStage[iStage];
		iStageCumulNodes = nodesPerStageCumul[iStage];
		if(iStage == 0){
			// x = 0
			_CUDA( cudaMemset(devVecXdir, 0, nx*sizeof(real_t)) );
			// u = Lv
			_CUBLAS(cublasSgemv_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, nu, nv, &alpha,
					ptrMyEngine->getSysMatL(), nu, devVecV, 1, &beta, devVecUdir, 1) );
			// x = x + Bu
			_CUBLAS(cublasSgemv_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, nx, nu, &alpha,
					ptrMyEngine->getSysMatB(), nx, devVecUdir, 1, &alpha, devVecXdir, 1) );
		}else{
			prevStageCumulNodes = nodesPerStageCumul[iStage - 1];
			if((nodesPerStage[iStage] - nodesPerStage[iStage-1]) > 0){
				// u_k = Lv_k
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, iStageNodes, nv,
						&alpha, ptrMyEngine->getSysMatL(), nu, &devVecV[iStageCumulNodes*nv], nv, &beta,
						&devVecUdir[iStageCumulNodes*nu], nu));
				// prevLv = u_{k-1}
				_CUDA( cudaMemcpy(devLv, &devVecUdir[prevStageCumulNodes*nu], nu*nodesPerStage[iStage-1]*sizeof(real_t),
						cudaMemcpyDeviceToDevice));
				// u_{k} = u_{k} + prevLu
				solveChildNodesUpdate<<<iStageNodes, nu>>>(devLv, &devVecUdir[iStageCumulNodes*nu], ptrMyEngine->getTreeAncestor(),
						iStageCumulNodes, nu);
				// x = Bu
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nx, iStageNodes, nu, &alpha,
						ptrMyEngine->getSysMatB(), nx, &devVecUdir[iStageCumulNodes*nu], nu, &beta, &devVecXdir[iStageCumulNodes*nx], nx));
				// x_{k+1}=x_k
				solveChildNodesUpdate<<<iStageNodes, nx>>>(&devVecXdir[prevStageCumulNodes*nx], &devVecXdir[iStageCumulNodes*nx],
						ptrMyEngine->getTreeAncestor(), iStageCumulNodes, nx);
			}else{
				// u_k = Lv_k
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, iStageNodes, nv, &alpha,
						ptrMyEngine->getSysMatL(), nu, &devVecV[iStageCumulNodes*nv], nv, &beta, &devVecUdir[iStageCumulNodes*nu], nu));
				// u_k = u_{k} + u_{k-1}^{anc}
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*iStageNodes ,&alpha, &devVecUdir[prevStageCumulNodes*nu], 1,
						&devVecUdir[iStageCumulNodes*nu], 1));
				// x_{k+1} = x_{k}
				_CUDA(cudaMemcpy(&devVecXdir[iStageCumulNodes*nx], &devVecXdir[prevStageCumulNodes*nx], nx*iStageNodes*sizeof(real_t),
						cudaMemcpyDeviceToDevice));
				// x = x+Bu
				_CUBLAS(cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nx, iStageNodes, nu, &alpha,
						ptrMyEngine->getSysMatB(), nx, &devVecUdir[iStageCumulNodes*nu], nu, &alpha, &devVecXdir[iStageCumulNodes*nx], nx));
			}

		}
	}

	// primalX = HxDir
	_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, 2*nx, 1, nx, &alpha, (const real_t**)
			ptrMyEngine->getPtrSysMatF(), 2*nx, (const real_t**)devPtrVecXdir, nx, &beta, devPtrVecPrimalXiDir, 2*nx, nodes) );
	_CUBLAS(cublasSgemmBatched(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, 1, nu, &alpha, (const real_t**)
			ptrMyEngine->getPtrSysMatG(), nu, (const real_t**)devPtrVecUdir, nu, &beta, devPtrVecPrimalPsiDir, nu, nodes) );

	_CUDA(cudaFree(devTempVecQ));
	_CUDA(cudaFree(devTempVecR));
	_CUDA(cudaFree(devLv) );
	devTempVecQ = NULL;
	devTempVecR = NULL;
	devLv = NULL;
}

/**
 * compute the gradient of the FBE
 */
void SmpcController::computeGradientFbe(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t ns = ptrMyEngine->getScenarioTree()->getNumScenarios();
	uint_t N =  ptrMyEngine->getScenarioTree()->getPredHorizon();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	real_t negAlpha = -1;

	// gradFbe = -(Hz - t)
	_CUDA( cudaMemcpy(devVecGradientFbeXi, devVecFixedPointResidualXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUDA( cudaMemcpy(devVecGradientFbePsi, devVecFixedPointResidualPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUBLAS( cublasSscal_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &negAlpha, devVecGradientFbeXi, 1));
	_CUBLAS( cublasSscal_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &negAlpha, devVecGradientFbePsi, 1));

	// hessianDirection(gradFbe)
	computeHessianOracalGlobalFbe();

	// gradFbeXi = gradFbe + lambda primalXidir and gradFbePsi = gradFbe + lambda primalPsidir
	_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &stepSize, devVecPrimalXiDir, 1, devVecGradientFbeXi, 1));
	_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &stepSize, devVecPrimalPsiDir, 1, devVecGradientFbePsi, 1));

}


/*
 * update the lbfgs buffer
 */
void SmpcController::updateLbfgsBuffer(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nDualXi = 2*nx*nodes;
	uint_t nDualPsi = nu*nodes;
	uint_t indexVecStart;
	uint_t lbfgsBufferSize = ptrMySmpcConfig->getLbfgsBufferSize();
	real_t negAlpha = -1;
	real_t lbfgsScale, invRhoVar;
	real_t normMatYk, normGrad, normMatSk;
	real_t gammaH;
	real_t *devVecS;
	real_t *devVecY;

	_CUDA( cudaMalloc((void**)&devVecS, (nDualXi + nDualPsi)*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecY, (nDualXi + nDualPsi)*sizeof(real_t)) );
	// vecS = y - yOld
	_CUDA( cudaMemcpy(devVecS, devVecXi, nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUDA( cudaMemcpy(&devVecS[nDualXi], devVecPsi, nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualXi, &negAlpha, devVecPrevXi, 1, devVecS, 1));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &negAlpha, devVecPrevPsi, 1, &devVecS[nDualXi], 1));

	// vecY = grad - gradOld
	_CUDA( cudaMemcpy(devVecY, ptrLbfgsCurrentYvecXi[0], nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUDA( cudaMemcpy(&devVecY[nDualXi], ptrLbfgsCurrentYvecPsi[0], nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualXi, &negAlpha, ptrLbfgsPreviousYvecXi[0], 1, devVecY, 1));
	_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &negAlpha, ptrLbfgsPreviousYvecPsi[0], 1, &devVecY[nDualXi], 1));
	// normGrad = norm(fbeGradient)
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualXi, ptrLbfgsCurrentYvecXi[0], 1, &lbfgsScale) );
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualPsi, ptrLbfgsCurrentYvecPsi[0], 1, &normGrad) );
	normGrad = sqrt(lbfgsScale*lbfgsScale + normGrad*normGrad);
	/*
	if (ptrMyEngine->getGlobalFbeFlag()){
		// vecY = grad - gradOld
		_CUDA( cudaMemcpy(devVecY, devVecGradientFbeXi, nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		_CUDA( cudaMemcpy(&devVecY[nDualXi], devVecGradientFbePsi, nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualXi, &negAlpha, devVecPrevGradientFbeXi, 1, devVecY, 1));
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &negAlpha, devVecPrevGradientFbePsi, 1, &devVecY[nDualXi], 1));
		// normGrad = norm(fbeGradient)
		_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualXi, devVecGradientFbeXi, 1, &lbfgsScale) );
		_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualPsi, devVecGradientFbePsi, 1, &normGrad) );
		normGrad = sqrt(lbfgsScale*lbfgsScale + normGrad*normGrad);
	}

	if (ptrMyEngine->getNamaFlag()){
		// vecY = fixedPoint - fixedPointOld
		_CUDA( cudaMemcpy(devVecY, devVecFixedPointResidualXi, nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		_CUDA( cudaMemcpy(&devVecY[nDualXi], devVecFixedPointResidualPsi, nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice) );
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualXi, &negAlpha, devVecPrevFixedPointResidualXi, 1, devVecY, 1));
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &negAlpha, devVecPrevGradientFbePsi, 1, &devVecY[nDualXi], 1));
		// normGrad = norm(fixedPointResidual)
		_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualXi, devVecFixedPointResidualXi, 1, &lbfgsScale) );
		_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualPsi, devVecFixedPointResidualPsi, 1, &normGrad) );
		normGrad = sqrt(lbfgsScale*lbfgsScale + normGrad*normGrad);

	}
	*/
	// invRho = vecS'vecY
	_CUBLAS(cublasSdot_v2(ptrMyEngine->getCublasHandle(), nDualXi + nDualPsi, devVecS, 1, devVecY, 1, &invRhoVar) );
	// normMatYk = sqrt(vecY[:, lbfgsBufferCol]*vecY[:, lbfgsBufferCol])
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualXi + nDualPsi, devVecY, 1, &normMatYk) );
	// normMatSk = sqrt(matS[:, lbfgsBufferCol]*matS[:, lbfgsBufferCol])
	_CUBLAS(cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nDualXi + nDualPsi, devVecS, 1, &normMatSk) );


	if(normGrad < 1){
		normGrad = normGrad*normGrad*normGrad;
	}
	if( invRhoVar/(normMatSk * normMatSk) > 1e-6*normGrad){
		lbfgsBufferCol = 1 + (lbfgsBufferCol % lbfgsBufferSize);
		lbfgsBufferMemory = min(lbfgsBufferMemory + 1, lbfgsBufferSize);
		indexVecStart = lbfgsBufferCol*(nDualXi + nDualPsi);
		_CUDA( cudaMemcpy(&devLbfgsBufferMatY[indexVecStart], devVecY, (nDualXi + nDualPsi)*sizeof(real_t),
				cudaMemcpyDeviceToDevice) );
		_CUDA( cudaMemcpy(&devLbfgsBufferMatS[indexVecStart], devVecS, (nDualXi + nDualPsi)*sizeof(real_t),
				cudaMemcpyDeviceToDevice) );
		lbfgsBufferRho[lbfgsBufferCol] = 1/invRhoVar;
	}else{
		lbfgsSkipCount = lbfgsSkipCount + 1;
	}

	gammaH = invRhoVar/(normMatYk * normMatYk);
	if( gammaH < 0 || abs(gammaH - lbfgsBufferHessian) == 0){
		lbfgsBufferHessian = 1;
	}else{
		lbfgsBufferHessian = gammaH;
	}

	/*
	if( ptrMyEngine->getGlobalFbeFlag() ){
		// lbfgsDir = -gradFbe
		_CUDA( cudaMemcpy(devVecLbfgsDirXi, devVecGradientFbeXi, nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA( cudaMemcpy(devVecLbfgsDirPsi, devVecGradientFbePsi, nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice));
	}

	if( ptrMyEngine->getNamaFlag() ){
		// lbfgsDir = -fixedPointResidual
		_CUDA( cudaMemcpy(devVecLbfgsDirXi, devVecFixedPointResidualXi, nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice));
		_CUDA( cudaMemcpy(devVecLbfgsDirPsi, devVecFixedPointResidualPsi, nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice));
	}
	 */
	// lbfgsDir = -gradFbe or -fixedPointResidual
	_CUDA( cudaMemcpy(devVecLbfgsDirXi, ptrLbfgsCurrentYvecXi[0], nDualXi*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUDA( cudaMemcpy(devVecLbfgsDirPsi, ptrLbfgsCurrentYvecPsi[0], nDualPsi*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUBLAS( cublasSscal_v2(ptrMyEngine->getCublasHandle(), nDualXi, &negAlpha, devVecLbfgsDirXi, 1));
	_CUBLAS( cublasSscal_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &negAlpha, devVecLbfgsDirPsi, 1));

	_CUDA( cudaFree(devVecS));
	_CUDA( cudaFree(devVecY));
	devVecS = NULL;
	devVecY = NULL;
}

/*
 * two-loop recursion algorithm for the Lbfgs algorithm
 */
void SmpcController::twoLoopRecursionLbfgs(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nDualXi = 2*nx*nodes;
	uint_t nDualPsi = nu*nodes;
	uint_t lbfgsBufferSize = ptrMySmpcConfig->getLbfgsBufferSize();
	real_t lbfgsAlpha[lbfgsBufferSize];
	real_t lbfgsBeta, lbfgsScale;
	uint_t iCol, indexVecStart;

	// two-loop recursion
	for(uint_t iSize = 0; iSize < lbfgsBufferMemory; iSize++){
		iCol = lbfgsBufferCol - iSize;
		if (iCol < 0)
			iCol = lbfgsBufferMemory + iCol;
		indexVecStart = iCol*(nDualXi + nDualPsi);
		_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nDualXi, &devLbfgsBufferMatS[indexVecStart], 1,
				devVecLbfgsDirXi, 1, &lbfgsAlpha[iCol]));
		_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &devLbfgsBufferMatS[indexVecStart + nDualXi], 1,
				devVecLbfgsDirPsi, 1, &lbfgsBeta));
		lbfgsAlpha[iCol] = lbfgsBufferRho[iCol]*(lbfgsAlpha[iCol] + lbfgsBeta);
		lbfgsScale = -lbfgsAlpha[iCol];
		_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualXi, &lbfgsScale, &devLbfgsBufferMatY[indexVecStart], 1,
				devVecLbfgsDirXi, 1) );
		_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &lbfgsScale, &devLbfgsBufferMatY[indexVecStart + nDualXi], 1,
				devVecLbfgsDirPsi, 1) );
	}

	_CUBLAS( cublasSscal_v2(ptrMyEngine->getCublasHandle(), nDualXi, &lbfgsBufferHessian, devVecLbfgsDirXi, 1) );
	_CUBLAS( cublasSscal_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &lbfgsBufferHessian, devVecLbfgsDirPsi, 1) );

	for(uint_t iSize = lbfgsBufferMemory; iSize > 0; iSize--){
		iCol = lbfgsBufferCol - iSize + 1;
		if(iCol < 0)
			iCol  = lbfgsBufferMemory + iCol;
		uint_t indexStart = iCol*(nDualXi + nDualPsi);
		_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nDualXi, &devLbfgsBufferMatY[indexStart], 1,
				devVecLbfgsDirXi, 1, &lbfgsBeta));
		_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &devLbfgsBufferMatY[indexStart + nDualXi], 1,
				devVecLbfgsDirPsi, 1, &lbfgsScale));
		lbfgsBeta = lbfgsScale + lbfgsBeta;
		lbfgsBeta = lbfgsBufferRho[iCol]*lbfgsBeta;
		lbfgsScale = lbfgsAlpha[iCol] - lbfgsBeta;
		_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualXi, &lbfgsScale, &devLbfgsBufferMatS[indexStart], 1,
				devVecLbfgsDirXi, 1) );
		_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nDualPsi, &lbfgsScale, &devLbfgsBufferMatS[indexStart + nDualXi], 1,
				devVecLbfgsDirPsi, 1) );
	}

}


/*
 * compute lbfgs direction
 */

void SmpcController::computeLbfgsDirection(){

	updateLbfgsBuffer();

	twoLoopRecursionLbfgs();
}


/**
 * function that compute the line search lbfgs update
 */
real_t SmpcController::computeLineSearchLbfgsUpdate(real_t valueFbeYvar){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	real_t tau = 1;
	real_t TOLERANCE = 1e-4;

	real_t valueFbeWvar, valueDirection;
	uint_t maxLineSearchStep = 10;
	uint_t iStep = 0;

	// swap fbeGrad and lbfgsDir
	_CUBLAS( cublasSswap_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecGradientFbeXi, 1, devVecLbfgsDirXi, 1));
	_CUBLAS( cublasSswap_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecGradientFbePsi, 1, devVecLbfgsDirPsi, 1));
	computeHessianOracalGlobalFbe();
	// swap fbeGrad and lbfgsDir
	_CUBLAS( cublasSswap_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecGradientFbeXi, 1, devVecLbfgsDirXi, 1));
	_CUBLAS( cublasSswap_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecGradientFbePsi, 1, devVecLbfgsDirPsi, 1));
	// gradFbe'lbfgDir,
	_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecGradientFbeXi, 1, devVecLbfgsDirXi,
			1, &valueFbeWvar));
	_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecGradientFbePsi, 1, devVecLbfgsDirPsi,
			1, &valueDirection));
	valueDirection = valueDirection + valueFbeWvar;
	if( valueDirection > 0){
		cout<< "LBFGS direction is positive " << valueDirection << endl;
	}else{
		if(abs(valueDirection) < TOLERANCE){
			tau = 0;
		}else{
			while( iStep < maxLineSearchStep + 1){
				// x = x + tau*xDir, u = u +tau*uDir
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nx*nodes, &tau, devVecXdir, 1, devVecX, 1));
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &tau, devVecUdir, 1, devVecU, 1));
				// accXi = accxi(xi) + tau*lbfgsDirXi, accPsi = accpsi(psi) + tau*lbfgsDirPsi
				//_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &tau, devVecLbfgsDirXi, 1,
					//	devVecAcceleratedXi, 1));
				//_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &tau, devVecLbfgsDirPsi, 1,
					//	devVecAcceleratedPsi, 1));
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &tau, devVecLbfgsDirXi, 1,
						ptrProximalXi[0], 1));
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &tau, devVecLbfgsDirPsi, 1,
						ptrProximalPsi[0], 1));
				// primalXi and primalPsi update Hz + tau Hzdir
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, &tau, devVecPrimalXiDir, 1,
						devVecPrimalXi, 1));
				_CUBLAS( cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nu*nodes, &tau, devVecPrimalPsiDir, 1,
						devVecPrimalPsi, 1));

				proximalFunG();
				computeFixedPointResidual();

				valueFbeWvar = computeValueFbe();

				//cout << "iStep " << iStep << " " << tau << " " << valueFbeWvar - valueFbeYvar << endl;
				if( valueFbeWvar <= valueFbeYvar){
					iStep = iStep + 1;
					if (iStep < maxLineSearchStep){
						if (iStep == 1)
							tau = -1;
						tau = tau  + 1/pow(2, iStep);
					}
				}else{
					iStep = maxLineSearchStep + 1;
				}
			}
		}
	}

	return abs(tau);
}



real_t SmpcController::computeValueFbe(){
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	real_t costFbeUpdate, valueXi, valuePsi;
	real_t *devCostMatW;
	real_t *devVecDeltaU;
	real_t *devVecMatWDeltaU;
	real_t alpha = 1;
	real_t beta = 0;

	_CUDA( cudaMalloc((void**)&devCostMatW, nu*nu*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecDeltaU, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMalloc((void**)&devVecMatWDeltaU, nu*nodes*sizeof(real_t)) );
	_CUDA( cudaMemset(devVecDeltaU, 0, nu*nodes*sizeof(real_t)));
	_CUDA( cudaMemcpy(devCostMatW, ptrMySmpcConfig->getCostW(), nu*nu*sizeof(real_t), cudaMemcpyHostToDevice) );

	// w'(Hz-t), accXi'residualXi + accPi'residualPi
	//_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecAcceleratedXi, 1, devVecFixedPointResidualXi,
		//	1, &valueXi));
	//_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecAcceleratedPsi, 1, devVecFixedPointResidualPsi,
		//	1, &valuePsi));
	_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, ptrProximalXi[0], 1, devVecFixedPointResidualXi,
			1, &valueXi));
	_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nu*nodes, ptrProximalPsi[0], 1, devVecFixedPointResidualPsi,
			1, &valuePsi));
	costFbeUpdate = valueXi + valuePsi;

	// norm(residual)
	_CUBLAS( cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecFixedPointResidualXi, 1, &valueXi));
	_CUBLAS( cublasSnrm2_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecFixedPointResidualPsi, 1, &valuePsi));
	costFbeUpdate = costFbeUpdate + 0.5*stepSize*(valueXi*valueXi + valuePsi*valuePsi);

	// add the value g
	costFbeUpdate = costFbeUpdate + valueFunGuBox + valueFunGxBox + valueFunGxSafe;
	// cost of f
	calculateDiffUhat<<<nodes, nu>>>(devVecDeltaU, devVecU, ptrMyEngine->getVecPreviousControl(), ptrMyEngine->getTreeAncestor(),
			nu, nodes);
	_CUDA( cudaMemcpy(devVecMatWDeltaU, devVecDeltaU, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
	_CUBLAS( cublasSgemm_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, nu, nodes, nu, &alpha, devCostMatW, nu,
			devVecDeltaU, nu, &beta, devVecMatWDeltaU, nu));
	scaleVecProbalitity<<<nodes, nu>>>(devVecDeltaU, ptrMyEngine->getTreeProb(), nu, nodes);
	_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecDeltaU, 1, devVecMatWDeltaU, 1, &valueXi));
	costFbeUpdate = costFbeUpdate + valueXi;
	_CUDA(cudaMemcpy(devVecDeltaU, devVecU, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice ));
	scaleVecProbalitity<<<nodes, nu>>>(devVecDeltaU, ptrMyEngine->getTreeProb(), nu, nodes);
	_CUBLAS( cublasSdot_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecDeltaU, 1, ptrMyEngine->getPriceAlpha(),
			1, &valueXi));
	costFbeUpdate = costFbeUpdate + valueXi;

	_CUDA( cudaFree(devCostMatW));
	_CUDA( cudaFree(devVecDeltaU));
	_CUDA( cudaFree(devVecMatWDeltaU));

	devCostMatW = NULL;
	devVecDeltaU = NULL;
	devVecMatWDeltaU = NULL;

	return costFbeUpdate;
}



real_t SmpcController::updatePrimalInfeasibity(){
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	uint_t maxIndexXi, maxIndexPsi;
	real_t maxValueXi, maxValuePsi;

	_CUBLAS( cublasIsamax_v2(ptrMyEngine->getCublasHandle(), 2*nx*nodes, devVecFixedPointResidualXi,
			1, &maxIndexXi));
	_CUDA( cudaMemcpy(&maxValueXi, &devVecFixedPointResidualXi[maxIndexXi - 1], sizeof(real_t),
			cudaMemcpyDeviceToHost));
	_CUBLAS( cublasIsamax_v2(ptrMyEngine->getCublasHandle(), nu*nodes, devVecFixedPointResidualPsi,
			1, &maxIndexPsi));
	_CUDA( cudaMemcpy(&maxValuePsi, &devVecFixedPointResidualPsi[maxIndexPsi - 1], sizeof(real_t),
			cudaMemcpyDeviceToHost));
	return( max( maxValueXi, maxValuePsi) );
}



uint_t SmpcController::algorithmApg(){

	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();
	real_t theta[2] = {1, 1};
	real_t lambda;
	uint_t maxIndex;

	initialiseAlgorithm();
	initialiseAlgorithmSpecificData();

	for (uint_t iter = 0; iter < ptrMySmpcConfig->getMaxIterations(); iter++){
		lambda = theta[1]*(1/theta[0] - 1);
		dualExtrapolationStep(lambda);
		solveStep();
		proximalFunG();
		computeFixedPointResidual();
		dualUpdate();
		theta[0] = theta[1];
		theta[1] = 0.5*(sqrt(pow(theta[1], 4) + 4*pow(theta[1], 2)) - pow(theta[1], 2));
		vecPrimalInfs[iter] = updatePrimalInfeasibity();
	}

	return 1;
}



uint_t SmpcController::algorithmGlobalFbe(){

	uint_t maxIndex;
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	initialiseAlgorithm();
	initaliseLbfgBuffer();
	initialiseAlgorithmSpecificData();

	for (uint_t iter = 0; iter < ptrMySmpcConfig->getMaxIterations(); iter++){
		dualExtrapolationStep(0);
		solveStep();
		proximalFunG();
		computeFixedPointResidual();
		computeGradientFbe();
		if( iter > 0){
			vecValueFbe[iter - 1] = computeValueFbe();
			computeLbfgsDirection();
			vecTau[iter - 1] = computeLineSearchLbfgsUpdate( vecValueFbe[iter - 1] );
		}
		dualUpdate();
		vecPrimalInfs[iter] = updatePrimalInfeasibity();
	}

	return 1;
}



uint_t SmpcController::algorithmNama(){

	uint_t maxIndex;
	uint_t nx = ptrMyEngine->getDwnNetwork()->getNumTanks();
	uint_t nu = ptrMyEngine->getDwnNetwork()->getNumControls();
	uint_t nodes = ptrMyEngine->getScenarioTree()->getNumNodes();

	initialiseAlgorithm();
	initaliseLbfgBuffer();
	initialiseAlgorithmSpecificData();

	for (uint_t iter = 0; iter < ptrMySmpcConfig->getMaxIterations(); iter++){
		//_CUDA( cudaMemcpy(devVecAcceleratedXi, devVecXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		//_CUDA( cudaMemcpy(devVecAcceleratedPsi, devVecPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToDevice));
		dualExtrapolationStep(0);
		solveStep();
		proximalFunG();
		computeFixedPointResidual();
		computeGradientFbe();
		if( iter > 0){
			vecValueFbe[iter - 1] = computeValueFbe();
			computeLbfgsDirection();
			vecTau[iter - 1] = computeLineSearchLbfgsUpdate( vecValueFbe[iter - 1] );
		}
		dualUpdate();
		vecPrimalInfs[iter] = updatePrimalInfeasibity();
	}

	return 1;
}

/**
 * Invoke the SMPC controller on the current state of the network.
 * This method invokes #updateStateControl, eliminateInputDistubanceCoupling
 * and finally #algorithmApg.
 */
void SmpcController::controllerSmpc(){
	ptrMyEngine->updateStateControl(ptrMySmpcConfig->getCurrentX(), ptrMySmpcConfig->getPrevU(),
			ptrMySmpcConfig->getPrevDemand());
	ptrMyEngine->eliminateInputDistubanceCoupling(ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	algorithmApg();
}

/**
 * Computes a control action and returns a status code
 * which is an integer (1 = success).
 * @param u pointer to computed control action (CPU variable)
 * @return status code
 */
uint_t SmpcController::controlAction(real_t* u){
	uint_t status;
	size_t initialFreeBytes;
	size_t totalBytes;
	size_t finalFreeBytes;
	_CUDA( cudaMemGetInfo(&initialFreeBytes, &totalBytes) );
	ptrMyEngine->updateStateControl(ptrMySmpcConfig->getCurrentX(), ptrMySmpcConfig->getPrevU(),
			ptrMySmpcConfig->getPrevDemand());
	ptrMyEngine->eliminateInputDistubanceCoupling(ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	status = algorithmApg();
	_CUDA( cudaMemcpy(u, devVecU, ptrMySmpcConfig->getNU()*sizeof(real_t), cudaMemcpyDeviceToHost));
	_CUDA( cudaMemGetInfo(&finalFreeBytes, &totalBytes) );
	if( abs(finalFreeBytes - initialFreeBytes) > 0 ){
		cout << "RUNTIME ERROR: MEMORY LEAKS" << endl;
		return 0;
	}else
		return status;
}

/**
 * Compute the control action, stores in the json file
 * provided to it and returns a status code (1 = success).
 * @param   controlJson   file pointer to the output json file
 * @return  status        code
 */
uint_t SmpcController::controlAction(fstream& controlOutputJson){
	if( controlOutputJson.is_open()){
		uint_t status;
		uint_t nu = ptrMySmpcConfig->getNU();
		real_t *currentControl = new real_t[nu];
		size_t initialFreeBytes;
		size_t totalBytes;
		size_t finalFreeBytes;
		_CUDA( cudaMemGetInfo(&initialFreeBytes, &totalBytes) );
		ptrMyEngine->updateStateControl(ptrMySmpcConfig->getCurrentX(), ptrMySmpcConfig->getPrevU(),
				ptrMySmpcConfig->getPrevDemand());
		ptrMyEngine->eliminateInputDistubanceCoupling(ptrMyForecaster->getNominalDemand(),
				ptrMyForecaster->getNominalPrices());
		status = algorithmApg();
		_CUDA( cudaMemcpy(devControlAction, devVecU, nu*sizeof(real_t),
				cudaMemcpyDeviceToDevice) );
		projectionBox<<<1, nu>>>(devControlAction, ptrMyEngine->getSysUmin(), ptrMyEngine->getSysUmax(), nu, 0, nu);
		_CUDA( cudaMemcpy(currentControl, devControlAction, nu*sizeof(real_t), cudaMemcpyDeviceToHost));
		controlOutputJson << "\"control\" : [" ;
		for(uint_t iControl = 0; iControl < nu; iControl++ ){
			//cout << currentControl[iControl] << " " ;
			controlOutputJson << currentControl[iControl] << ", ";
		}
		//cout << endl;
		_CUDA( cudaMemGetInfo(&finalFreeBytes, &totalBytes) );
		controlOutputJson << "]" << endl;
		delete [] currentControl;
		if( abs(finalFreeBytes - initialFreeBytes) > 0 ){
			cout << "RUNTIME ERROR: MEMORY LEAKS" << endl;
			return 0;
		}else
			return status;
	}else
		return 0;
}

/*
 * During the closed-loop of the controller,
 * the controller moves to the next time instance. It checks
 * for the flag SIMULATOR_FLAG, 1 corresponds to an in-build
 * simulator call given by `updateSmpcConfiguration()` and
 * 0 corresponds to external simulator.
 *
 * Reads the smpcControlConfiguration file for currentState,
 * previousDemand and previousControl action.
 */
void SmpcController::moveForewardInTime(){
	if(simulatorFlag){
		//compute get the control from the devControl, apply the projection,
		// compute x+Bu+e to get updated state
		uint_t nx = this->ptrMyEngine->getDwnNetwork()->getNumTanks();
		uint_t nu = this->ptrMyEngine->getDwnNetwork()->getNumControls();
		uint_t nd = this->ptrMyEngine->getDwnNetwork()->getNumDemands();
		real_t *previousControl = new real_t[nu];
		real_t *stateUpdate = new real_t[nx];
		real_t *previousDemand;
		real_t alpha = 1;

		//x = p
		_CUDA( cudaMemcpy( devStateUpdate, ptrMyEngine->getVecCurrentState(), nx*sizeof(real_t),
				cudaMemcpyDeviceToDevice) );
		// x = x+w
		_CUBLAS(cublasSaxpy_v2(ptrMyEngine->getCublasHandle(), nx, &alpha, ptrMyEngine->getVecE(), 1, devVecX, 1));
		// x = x+Bu
		_CUBLAS(cublasSgemv_v2(ptrMyEngine->getCublasHandle(), CUBLAS_OP_N, nx, nu, &alpha,
				ptrMyEngine->getSysMatB(), nx, devControlAction, 1, &alpha, devStateUpdate, 1) );
		_CUDA( cudaMemcpy(stateUpdate, devStateUpdate, nx*sizeof(real_t), cudaMemcpyDeviceToHost) );
		_CUDA( cudaMemcpy(previousControl, devControlAction, nu*sizeof(real_t), cudaMemcpyDeviceToHost) );
		previousDemand = this->ptrMyForecaster->getNominalDemand();
		//updateSmpcConfiguration(stateUpdate, previousControl, previousDemand);
		updateKpi( stateUpdate, previousControl );
		this->ptrMySmpcConfig->setCurrentState( stateUpdate );
		this->ptrMySmpcConfig->setPreviousControl( previousControl );
		this->ptrMySmpcConfig->setpreviousdemand( previousDemand );
		delete [] previousControl;
		delete [] stateUpdate;
		previousControl = NULL;
		stateUpdate = NULL;
		//previousDemand = NULL;
	}else{
		this->ptrMySmpcConfig->setCurrentState();
		this->ptrMySmpcConfig->setPreviousControl();
		this->ptrMySmpcConfig->setPreviousDemand();
	}
}

/**
 * Update tje json file using the commands from rapidJson functions
 * When the SIMULATOR FLAG is set to 1, the previousControl,
 * currentState and previousDemand vectors in the smpc controller
 * configuration file are set.
 */
void SmpcController::updateSmpcConfiguration(real_t* updateState,
		real_t* control,
		real_t* demand){
	//const char* fileName = ptrMySmpcConfig->getPathToControllerConfig().c_str();
	string pathToFileString = "../systemData/testControl.json";
	const char* fileName = pathToFileString.c_str();
	rapidjson::Document jsonDocument;
	//rapidjson::Value valueJson;
	uint_t nx = ptrMySmpcConfig->getNX();
	uint_t nu = ptrMySmpcConfig->getNU();
	uint_t nd = ptrMySmpcConfig->getND();
	FILE* infile = fopen(fileName, "r");
	char* readBuffer = new char[65536];
	rapidjson::FileReadStream configurationJsonStream(infile, readBuffer, sizeof(readBuffer));
	jsonDocument.ParseStream(configurationJsonStream);
	//jsonDocument.RemoveMember(VARNAME_CURRENT_X);
	//jsonDocument.RemoveMember(VARNAME_PREV_U);
	//jsonDocument.RemoveMember(VARNAME_PREV_DEMAND);

	rapidjson::Value currentXjson(rapidjson::kArrayType);
	rapidjson::Value previousUjson(rapidjson::kArrayType);
	rapidjson::Value previousDemandJson(rapidjson::kArrayType);
	rapidjson::Document::AllocatorType& allocator = jsonDocument.GetAllocator();

	for(uint_t iSize = 0; iSize < nx; iSize++){
		currentXjson.PushBack(rapidjson::Value().SetFloat( updateState[iSize] ), allocator);
		cout << updateState[iSize] << " ";
	}
	cout << endl;
	for(uint_t iSize = 0; iSize < nu; iSize++){
		previousUjson.PushBack(rapidjson::Value().SetFloat( control[iSize] ), allocator);
	}
	for(uint_t iSize = 0; iSize < nd; iSize++){
		previousDemandJson.PushBack(rapidjson::Value().SetFloat( demand[iSize] ), allocator);
	}
	jsonDocument.AddMember(VARNAME_CURRENT_X, currentXjson, jsonDocument.GetAllocator());
	jsonDocument.AddMember(VARNAME_PREV_U, previousUjson, jsonDocument.GetAllocator());
	jsonDocument.AddMember(VARNAME_PREV_DEMAND, previousDemandJson, jsonDocument.GetAllocator());

	FILE* outfile = fopen(fileName, "w");
	char* writeBuffer = new char[65536];
	rapidjson::FileWriteStream os(outfile, writeBuffer, sizeof(writeBuffer));

	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
	jsonDocument.Accept(writer);
	fclose(outfile);
	delete [] readBuffer;
	delete [] writeBuffer;
}

/**
 * update the KPI at the current time instance
 */
void SmpcController::updateKpi(real_t* state, real_t* control){
	uint_t nx = this->ptrMySmpcConfig->getNX();
	uint_t nu = this->ptrMySmpcConfig->getNU();

	real_t *safeX = this->ptrMyEngine->getDwnNetwork()->getXsafe();
	real_t *constantPrice = this->ptrMyEngine->getDwnNetwork()->getAlpha();
	real_t *variablePrice = this->ptrMyForecaster->getNominalPrices();
	real_t *previousControl = this->ptrMySmpcConfig->getPrevU();
	real_t weightEconomic = ptrMySmpcConfig->getWeightEconomical();
	real_t *deltaU = new real_t[nu];
	real_t *waterLevel = new real_t[nx];

	real_t ecoKpi = 0;
	real_t smKpi = 0;
	real_t saKpi = 0;
	real_t netKpi = 0;
	real_t safeValue = 0;

	for(uint_t iSize = 0; iSize < nu; iSize++){
		ecoKpi = ecoKpi + weightEconomic*(constantPrice[iSize] + variablePrice[iSize])*abs(control[iSize]);
		deltaU[iSize] = previousControl[iSize] - control[iSize];
		smKpi = smKpi + deltaU[iSize]*deltaU[iSize];
	}
	for(uint_t iSize = 0; iSize < nx; iSize++){
		waterLevel[iSize] = state[iSize] - safeX[iSize];
		if( waterLevel[iSize] > 0 ){
			waterLevel[iSize] = 0;
		}
		safeValue = safeValue + abs( safeX[iSize] );
		saKpi = saKpi + abs( waterLevel[iSize] );
		netKpi = netKpi + abs( state[iSize] );
	}

	economicKpi = economicKpi + ecoKpi;
	smoothKpi = smoothKpi + smKpi;
	safeKpi = safeKpi + saKpi;
	networkKpi = networkKpi + netKpi;
	//cout << saKpi << " "<< netKpi << " " << safeValue << endl;
	delete [] deltaU;
	delete [] waterLevel;
}

/*
 * Get the economical KPI upto the simulation horizon
 * @param    simualtionTime  simulation horizon
 */
real_t SmpcController::getEconomicKpi( uint_t simulationTime){
	real_t economicValue = economicKpi/(3600);
	return economicValue/simulationTime;
}

/*
 * Get the smooth KPI upto the simulation horizon
 * @param    simulationTime   simulation horizon
 */
real_t SmpcController::getSmoothKpi( uint_t simulationTime){
	real_t smoothValue = smoothKpi/(3600);
	return smoothValue/simulationTime;
}

/*
 * Get the  network KPI upto the simulation horizon
 * @param   simulationTime    simulation horizon
 */
real_t SmpcController::getNetworkKpi( uint_t simulationTime){
	real_t networkKpiTime = networkKpi;
	real_t safeLevelNorm = 0;
	uint_t nx = this->ptrMySmpcConfig->getNX();
	for(uint_t iSize = 0; iSize < nx; iSize++){
		safeLevelNorm = safeLevelNorm + this->getDwnNetwork()->getXsafe()[iSize];
	}
	networkKpiTime = 100*simulationTime*safeLevelNorm/networkKpiTime;
	return networkKpiTime;
}

/*
 * Get the safety KPI upto the simulation horizon
 * @param   simulationTime    simulation horizon
 */
real_t SmpcController::getSafetyKpi( uint_t simulationTime){
	return safeKpi;
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

/**
 * get the value of the FBE
 * @return fbeValue
 */
real_t* SmpcController::getValueFbe(){
	return vecValueFbe;
}

/**
 * deallocate APG memory
 */
void SmpcController::deallocateApgAlgorithm(){
	_CUDA( cudaFree(devVecUpdateXi) );
	_CUDA( cudaFree(devVecUpdatePsi) );

	devVecUpdateXi = NULL;
	devVecUpdatePsi = NULL;
}

/**
 * deallocate LBFGS direction
 */
void SmpcController::deallocateLbfgsDirection(){

	_CUDA( cudaFree(devVecPrevXi) );
	_CUDA( cudaFree(devVecPrevPsi) );
	_CUDA( cudaFree(devVecLbfgsDirXi) );
	_CUDA( cudaFree(devVecLbfgsDirPsi) );
	_CUDA( cudaFree(devVecXdir) );
	_CUDA( cudaFree(devVecUdir) );
	_CUDA( cudaFree(devVecPrimalXiDir) );
	_CUDA( cudaFree(devVecPrimalPsiDir) );
	_CUDA( cudaFree(devPtrVecHessianOracleXi));
	_CUDA( cudaFree(devPtrVecHessianOraclePsi));

	_CUDA( cudaFree(devPtrVecXdir) );
	_CUDA( cudaFree(devPtrVecUdir) );
	_CUDA( cudaFree(devPtrVecPrimalXiDir) );
	_CUDA( cudaFree(devPtrVecPrimalPsiDir) );

	_CUDA( cudaFree(devLbfgsBufferMatS) );
	_CUDA( cudaFree(devLbfgsBufferMatY) );
	free(lbfgsBufferRho);

	free( ptrLbfgsCurrentYvecXi );
	free( ptrLbfgsCurrentYvecPsi );
	free( ptrLbfgsPreviousYvecXi );
	free( ptrLbfgsPreviousYvecPsi );


	devVecPrevXi = NULL;
	devVecPrevPsi = NULL;
	devVecLbfgsDirXi = NULL;
	devVecLbfgsDirPsi = NULL;
	devVecXdir = NULL;
	devVecUdir = NULL;
	devVecPrimalXiDir = NULL;
	devVecPrimalPsiDir = NULL;
	devPtrVecHessianOracleXi = NULL;
	devPtrVecHessianOraclePsi = NULL;

	devPtrVecXdir = NULL;
	devPtrVecUdir = NULL;
	devPtrVecPrimalXiDir = NULL;
	devPtrVecPrimalPsiDir = NULL;

	devLbfgsBufferMatS = NULL;
	devLbfgsBufferMatY = NULL;
	lbfgsBufferRho = NULL;

	ptrLbfgsCurrentYvecXi = NULL;
	ptrLbfgsCurrentYvecPsi = NULL;
	ptrLbfgsPreviousYvecXi = NULL;
	ptrLbfgsCurrentYvecPsi = NULL;
}

/**
 * deallocate global FBE algorithm
 */
void SmpcController::deallocateGlobalFbeAlgorithm(){

	deallocateLbfgsDirection();

	_CUDA( cudaFree(devVecGradientFbeXi) );
	_CUDA( cudaFree(devVecGradientFbePsi) );
	_CUDA( cudaFree(devVecPrevGradientFbeXi) );
	_CUDA( cudaFree(devVecPrevGradientFbePsi) );


	_CUDA( cudaFree(devPtrVecGradFbeXi) );
	_CUDA( cudaFree(devPtrVecGradFbePsi) );

	devVecGradientFbeXi = NULL;
	devVecGradientFbePsi = NULL;
	devVecPrevGradientFbeXi = NULL;
	devVecPrevGradientFbePsi = NULL;

	devPtrVecGradFbeXi = NULL;
	devPtrVecGradFbePsi = NULL;
}

/**
 * deallocate NAMA algorithm
 */
void SmpcController::deallocateNamaAlgorithm(){
	deallocateLbfgsDirection();

	_CUDA( cudaFree(devVecPrevFixedPointResidualXi) );
	_CUDA( cudaFree(devVecPrevFixedPointResidualPsi) );

	_CUDA( cudaFree(devPtrVecFixedPointResidualXi) );
	_CUDA( cudaFree(devPtrVecFixedPointResidualPsi) );

	devVecPrevFixedPointResidualXi = NULL;
	devVecPrevFixedPointResidualPsi = NULL;
	devPtrVecFixedPointResidualXi = NULL;
	devPtrVecFixedPointResidualPsi = NULL;
}

/**
 * deallocate SMPC
 */
void SmpcController::deallocateSmpcController(){
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
	_CUDA( cudaFree(devVecFixedPointResidualXi) );
	_CUDA( cudaFree(devVecFixedPointResidualPsi) );

	_CUDA( cudaFree(devVecQ) );
	_CUDA( cudaFree(devVecR) );
	_CUDA( cudaFree(devControlAction) );
	_CUDA( cudaFree(devStateUpdate) );

	_CUDA( cudaFree(devPtrVecX) );
	_CUDA( cudaFree(devPtrVecU) );
	_CUDA( cudaFree(devPtrVecV) );
	_CUDA( cudaFree(devPtrVecPrimalXi) );
	_CUDA( cudaFree(devPtrVecPrimalPsi) );
	_CUDA( cudaFree(devPtrVecQ));
	_CUDA( cudaFree(devPtrVecR));

	_CUDA( cudaFree(devPtrVecSolveStepXi) );
	_CUDA( cudaFree(devPtrVecSolveStepPsi) );
	delete [] ptrProximalXi;
	delete [] ptrProximalPsi;

	delete [] vecPrimalInfs;
	delete [] vecValueFbe;
	delete [] vecTau;


	devVecX = NULL;
	devVecU = NULL;
	devVecV = NULL;
	devVecXi = NULL;
	devVecPsi = NULL;
	devVecUpdateXi = NULL;
	devVecUpdatePsi = NULL;

	devVecPrimalXi = NULL;
	devVecPrimalPsi = NULL;
	devVecDualXi = NULL;
	devVecDualPsi = NULL;
	devVecFixedPointResidualXi = NULL;
	devVecFixedPointResidualPsi = NULL;

	devVecQ = NULL;
	devVecR = NULL;
	devControlAction = NULL;
	devStateUpdate = NULL;

	devPtrVecX = NULL;
	devPtrVecU = NULL;
	devPtrVecV = NULL;
	devPtrVecPrimalXi = NULL;
	devPtrVecPrimalPsi = NULL;
	devPtrVecQ = NULL;
	devPtrVecR = NULL;
	devPtrVecSolveStepXi = NULL;
	devPtrVecSolveStepPsi = NULL;
	ptrProximalXi = NULL;
	ptrProximalPsi = NULL;

	vecPrimalInfs = NULL;
	vecValueFbe = NULL;
	vecTau = NULL;

}


SmpcController::~SmpcController(){

	deallocateSmpcController();

	if( ptrMyEngine->getGlobalFbeFlag() ){
		deallocateGlobalFbeAlgorithm();
	}else if( ptrMyEngine->getNamaFlag() ){
		deallocateNamaAlgorithm();
	}else if( ptrMyEngine->getApgFlag() ){
		deallocateApgAlgorithm();
	}
}
