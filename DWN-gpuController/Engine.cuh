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

/*TODO Add proper copyright notice here and in every file */

#ifndef ENGINE_CUH_
#define ENGINE_CUH_

#include "Configuration.h"
#include "DwnNetwork.cuh"
#include "ScenarioTree.cuh"
#include "SmpcConfiguration.cuh"
#include "Forecaster.cuh"
#include "Utilities.cuh"
#include "cublas_v2.h"

/*TODO IF A METHOD IS NOT TO BE INVOKED FROM THE OUTSIDE OF THIS CLASS, MAKE IT PRIVATE */
/*TODO INTRODUCE GETTERS FOR THOSE FIELDS WHICH NEED TO BE ACCESSIBLE FROM THE OUTSIDE */

/**
 *TODO Document this class (what is its purpose)
 */
class Engine{
public:

	/**
	 *
	 * @param network
	 * @param forecaster
	 * @todo remove `SmpcConfig` from here
	 */
	Engine( DwnNetwork *network,
			ScenarioTree *scenarioTree,
			SmpcConfiguration *smpcConfig);

	/**
	 * Eliminate input-demand coupling equations
	 * @param   nominalDemand    demand predicted
	 * @param   nominalPrice     price prediction
	 */
	void eliminateInputDistubanceCoupling(
			real_t* nominalDemand,
			real_t *nominalPrices);

	/**
	 * Update the state and control in the device
	 * @param   currentX      current level of tanks in the system
	 * @param   prevU         previous control action
	 * @param   prevUhat      previous control action particular solution
	 * TODO replace the prevUhat parameter
	 */
	void updateStateControl(
			real_t* currentX,
			real_t* prevU,
			real_t* prevUhat);

	/**
	 * Implements the Factor step that calculates all the constant
	 * matrices of the APG algorithm - refer to Appendix B in the paper
	 */
	void factorStep();

	/**
	 *  pointer to the scenario tree
	 */
	ScenarioTree* getScenarioTree();
	/**
	 *  pointer to the DWN network
	 */
	DwnNetwork* getDwnNetwork();

	/* ---- GETTER'S OF THE SYSTEM ----*/
	/**
	 * System matrix B
	 */
	real_t* getSysMatB();
	/**
	 * constraints matrix F
	 */
	real_t* getSysMatF();
	/**
	 * constraints matrix G
	 */
	real_t* getSysMatG();
	/**
	 * matrix L
	 */
	real_t* getSysMatL();
	/**
	 * matrix Lhat
	 */
	real_t* getSysMatLhat();
	/**
	 * pointer to Matrix B
	 */
	real_t** getPtrSysMatB();
	/**
	 * pointer to matrix F
	 */
	real_t** getPtrSysMatF();
	/**
	 * pointer to matrix G
	 */
	real_t** getPtrSysMatG();
	/**
	 * pointer to matrix L
	 */
	real_t** getPtrSysMatL();
	/**
	 * pointer to matrix Lhat
	 */
	real_t** getPtrSysMatLhat();
	/**
	 * previous control
	 */
	real_t* getVecPreviousControl();
	/**
	 * current state
	 */
	real_t* getVecCurrentState();
	/**
	 * previous uhat
	 */
	real_t* getVecPreviousUhat();

	/** ----GETTER'S FOR FACTOR MATRICES----*/
	/**
	 *  matrix Phi
	 */
	real_t* getMatPhi();
	/**
	 * matrix Psi
	 */
	real_t* getMatPsi();
	/**
	 * matrix Theta
	 */
	real_t* getMatTheta();
	/**
	 * matrix Theta
	 */
	real_t* getMatOmega();
	/**
	 * matrix Sigma
	 */
	real_t* getMatSigma();
	/**
	 * matrix D
	 */
	real_t* getMatD();
	/**
	 * matrix F (Factor step)
	 */
	real_t* getMatF();
	/**
	 * matrix G (Facotr step)
	 */
	real_t* getMatG();
	/**
	 * pointer matrix Phi
	 */
	real_t** getPtrMatPhi();
	/**
	 * pointer matrix Psi
	 */
	real_t** getPtrMatPsi();
	/**
	 * pointer matrix Theta
	 */
	real_t** getPtrMatTheta();
	/**
	 * pointer matrix Omega
	 */
	real_t** getPtrMatOmega();
	/**
	 * pointer matrix Sigma
	 */
	real_t** getPtrMatSigma();
	/**
	 * pointer matrix D
	 */
	real_t** getPtrMatD();
	/**
	 * pointer matrix F (Factor step)
	 */
	real_t** getPtrMatF();
	/**
	 * pointer matrix G (Factor step)
	 */
	real_t** getPtrMatG();
	/**
	 * uhat
	 */
	real_t* getVecUhat();
	/**
	 * beta control-distribution elimination
	 */
	real_t* getVecBeta();
	/**
	 * e control-disturbance elimination
	 */
	real_t* getVecE();
	/**
	 * handle cublasHandle
	 */
	cublasHandle_t getCublasHandle();

	/* ----GETTER'S FOR THE SCENARIO TREE----*/
	/**
	 * Array of stages
	 */
	uint_t* getTreeStages();
	/**
	 * Array of nodes per stage
	 */
	uint_t* getTreeNodesPerStage();
	/**
	 * Array of past nodes
	 */
	uint_t* getTreeNodesPerStageCumul();
	/**
	 * Array of the leaves
	 */
	uint_t* getTreeLeaves();
	/**
	 * Array number of children
	 */
	uint_t* getTreeNumChildren();
	/**
	 * Array of ancestor
	 */
	uint_t* getTreeAncestor();
	/**
	 * Array of past cumulative children
	 */
	uint_t* getTreeNumChildrenCumul();
	/**
	 * Array of the probability
	 */
	real_t* getTreeProb();
	/**
	 * Array of the error in the demand
	 */
	real_t* getTreeErrorDemand();
	/**
	 * Array of the error in the prices
	 */
	real_t* getTreeErrorPrices();

	/* --- GETTER'S OF NETWORK CONSTRAINTS --- */
	/**
	 * state/volume minimum
	 */
	real_t* getSysXmin();
	/**
	 * state/volume maximum
	 */
	real_t* getSysXmax();
	/**
	 * state/volume safe level
	 */
	real_t* getSysXs();
	/**
	 * dummy state/volume safe level
	 */
	real_t* getSysXsUpper();
	/**
	 * actuator/control minimum
	 */
	real_t* getSysUmin();
	/**
	 * actuator/cotrol maximum
	 */
	real_t* getSysUmax();

	/**
	 * @todo remove Friendship
	 */
	friend class SmpcController;

	/**
	 * Destructor
	 */
	~Engine();

private:
	/**
	 *  Allocate the device memory for the scenario tree
	 */
	void allocateScenarioTreeDevice();

	/**
	 * Allocate the device memory for the system
	 */
	void allocateSystemDevice();

	/**
	 * Initialise the scenario tree in the device
	 */
	void initialiseScenarioTreeDevice();

	/**
	 * Initialise the system model in the device
	 */
	void initialiseSystemDevice();
	/**
	 *
	 * @param src
	 * @param dst
	 * @param n
	 * @param batchSize
	 */
	void inverseBatchMat(
			real_t** src,
			real_t** dst,
			uint_t n,
			uint_t batchSize);
	/**
	 * Deallocate the device memory of the scenario tree
	 */
	void deallocateScenarioTreeDevice();
	/**
	 * Deallocate the device memory of the system matrices
	 */
	void deallocateSystemDevice();
	/**
	 * Pointer to the network
	 */
	DwnNetwork *ptrMyNetwork;
	/**
	 * Pointer to the scenario tree
	 */
	ScenarioTree *ptrMyScenarioTree;
	/**
	 * Pointer to the Smpc configuration
	 */
	SmpcConfiguration *ptrMySmpcConfig;
	/* --- NETWORK --- */
	/**
	 * matrix B
	 */
	real_t *devSysMatB;
	/**
	 * constraints matrix F
	 */
	real_t *devSysMatF;
	/**
	 * constraints matrix G
	 */
	real_t *devSysMatG;
	/**
	 * matrix L
	 */
	real_t *devSysMatL;
	/**
	 * matrix Lhat
	 */
	real_t  *devSysMatLhat;
	/**
	 * pointer to Matrix B
	 */
	real_t **devPtrSysMatB;
	/**
	 * pointer to matrix F
	 */
	real_t  **devPtrSysMatF;
	/**
	 * pointer to matrix G
	 */
	real_t  **devPtrSysMatG;
	/**
	 * pointer to matrix L
	 */
	real_t  **devPtrSysMatL;
	/**
	 * pointer to matrix Lhat
	 */
	real_t  **devPtrSysMatLhat;
	/**
	 * previous control
	 */
	real_t *devVecPreviousControl;
	/**
	 * current state
	 */
	real_t  *devVecCurrentState;
	/**
	 * previous uhat
	 */
	real_t  *devVecPreviousUhat;




	/* --- NETWORK CONSTRAINTS --- */

	/**
	 * state/volume minimum
	 */
	real_t *devSysXmin;
	/**
	 * state/volume maximum
	 */
	real_t *devSysXmax;
	/**
	 * state/volume safe level
	 */
	real_t  *devSysXs;
	/**
	 * dummy state/volume safe level
	 */
	real_t  *devSysXsUpper;
	/**
	 * actuator/control minimum
	 */
	real_t  *devSysUmin;
	/**
	 * actuator/cotrol maximum
	 */
	real_t  *devSysUmax;




	/* --- COST FUNCTION --- */

	/**
	 * smooth operation cost W
	 */
	real_t *devSysCostW;
	/**
	 * pointer to smooth operation cost W
	 */
	real_t **devPtrSysCostW;


	/* --- SCENARIO TREE --- */

	/**
	 * Array of stages
	 */
	uint_t *devTreeStages;
	/**
	 * Array of nodes per stage
	 */
	uint_t *devTreeNodesPerStage;
	/**
	 * Array of past nodes
	 */
	uint_t *devTreeNodesPerStageCumul;
	/**
	 * Array of the leaves
	 */
	uint_t *devTreeLeaves;
	/**
	 * Array number of children
	 */
	uint_t *devTreeNumChildren;
	/**
	 * Array of ancestor
	 */
	uint_t *devTreeAncestor;
	/**
	 * Array of past cumulative children
	 */
	uint_t *devTreeNumChildrenCumul;
	/**
	 * Array of the probability
	 */
	real_t *devTreeProb;
	/**
	 * Array of the error in the demand
	 */
	real_t *devTreeErrorDemand;
	/**
	 * Array of the error in the prices
	 */
	real_t *devTreeErrorPrices;


	/* --- FACTOR MATRICES --- */

	/**
	 *  matrix Phi
	 */
	real_t *devMatPhi;
	/**
	 * matrix Psi
	 */
	real_t *devMatPsi;
	/**
	 * matrix Theta
	 */
	real_t *devMatTheta;
	/**
	 * matrix Theta
	 */
	real_t *devMatOmega;
	/**
	 * matrix Sigma
	 */
	real_t *devMatSigma;
	/**
	 * matrix D
	 */
	real_t *devMatD;
	/**
	 * matrix F (Factor step)
	 */
	real_t *devMatF;
	/**
	 * matrix G (Facotr step)
	 */
	real_t *devMatG;
	/**
	 * pointer matrix Phi
	 */
	real_t **devPtrMatPhi;
	/**
	 * pointer matrix Psi
	 */
	real_t **devPtrMatPsi;
	/**
	 * pointer matrix Theta
	 */
	real_t **devPtrMatTheta;
	/**
	 * pointer matrix Omega
	 */
	real_t **devPtrMatOmega;
	/**
	 * pointer matrix Sigma
	 */
	real_t **devPtrMatSigma;
	/**
	 * pointer matrix D
	 */
	real_t **devPtrMatD;
	/**
	 * pointer matrix F (Factor step)
	 */
	real_t **devPtrMatF;
	/**
	 * pointer matrix G (Factor step)
	 */
	real_t **devPtrMatG;
	/**
	 * uhat
	 */
	real_t *devVecUhat;
	/**
	 * beta control-distribution elimination
	 */
	real_t *devVecBeta;
	/**
	 * e control-disturbance elimination
	 */
	real_t *devVecE;
	/**
	 * Wv = W*L
	 */
	real_t *devMatWv;
	/**
	 * cublas handler
	 */
	cublasHandle_t handle;
};

#endif /* ENGINE_CUH_ */
