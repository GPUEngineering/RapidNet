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

#ifndef SMPCONTROLLERCLASS_CUH_
#define SMPCONTROLLERCLASS_CUH_

#include "Configuration.h"
#include "Engine.cuh"
//#include "cudaKernelHeader.cuh"

/**
 * GPU-based stochastic model predictive controller for management of the
 * drinking water water network. This method is used dual proximal gradient method
 * to solve the optimisation problem resulted from the stochastic mpc
 * controller. This algorithm is completely parallelisable takes full advantage of
 * parallel computational capability of the GPU devices.
 *
 * The object initiates the controller object. This contains
 *   - The forecaster object to predict the nominal water demand and nominal prices
 *   - The Engine object generates matrices that are used in the actual algorithm
 *   - The SmpcConfiguration object that have all the user defined parameters of the
 *     algorithm
 *
 */
class SmpcController {
public:
	/**
	 * Construct a new Controller with a given engine.
	 * @param  myForecaster   An instance of the Forecaster object
	 * @param  myEngine       An instance of Engine object
	 * @param  mySmpcConfig   An instance of the Smpc controller configuration object
	 */
	SmpcController( Forecaster *myForecaster,
			Engine *myEngine,
			SmpcConfiguration *mySmpcConfig );
	/**
	 * Construct a new Controller with a given engine.
	 * @param  pathToConfigFile   path to the controller configuration file
	 */
	SmpcController( string pathToConfigFile );

	/**
	 * Performs the initialise the smpc controller
	 *   - update the current state and previous controls in the device memory
	 *   - perform the factor step
	 */
	void initialiseSmpcController();

	/**
	 * Invoke the SMPC controller on the current state of the network.
	 * This method invokes #updateStateControl, eliminateInputDistubanceCoupling
	 * and finally #algorithmApg.
	 */
	void controllerSmpc();
	/**
	 *Computes a control action and returns a status code
	 *which is an integer (1 = success).
	 * @param u pointer to computed control action (CPU variable)
	 * @return status code
	 */
	int controlAction(real_t* u);
	/**
	 * Destructor. Frees allocated memory.
	 */
	~SmpcController();
protected:
	/**
	 * Performs the dual extrapolation step with given parameter.
	 * @param extrapolation parameter.
	 */
	void dualExtrapolationStep(real_t lambda);
	/**
	 * Computes the dual gradient.This is the main computational
	 * algorithm for the proximal gradient algorithm
	 */
	void solveStep();
	/**
	 * Computes the proximal operator of g at the current point and updates
	 * (primal psi, primal xi) - Hx, (dual psi, dual xi) - z.
	 */
	void proximalFunG();
	/**
	 * Performs the update of the dual vector.
	 */
	void dualUpdate();
	/**
	 * This method executes the APG algorithm and returns the primal infeasibility.
	 * @return primalInfeasibilty;
	 */
	uint_t algorithmApg();
	/**
	 *
	 */
//private:
	/**
	 * Pointer to an Engine object.
	 * The Engine is responsible for the factor step.
	 */
	Engine* ptrMyEngine;
	/**
	 * Pointer to the Forecaster object
	 * This contains the forecast of the demand and prices
	 */
	Forecaster* ptrMyForecaster;
	/**
	 * Pointer to the Smpc configuration object
	 * This contains the configurations for the controller
	 */
	SmpcConfiguration* ptrMySmpcConfig;
	/**
	 * Pointer to device state
	 */
	real_t *devVecX;
	/**
	 * Pointer to device control
	 */
	real_t *devVecU;
	/**
	 * Pointer to device reduced control
	 */
	real_t *devVecV;
	/**
	 * Pointer to device Xi
	 */
	real_t *devVecXi;
	/**
	 * Pointer to device Psi
	 */
	real_t *devVecPsi;
	/**
	 * Pointer to device accelerated Xi
	 */
	real_t *devVecAcceleratedXi;
	/**
	 * Pointer to device accelerated Psi
	 */
	real_t *devVecAcceleratedPsi;
	/**
	 * Pointer to device primal Xi
	 */
	real_t *devVecPrimalXi;
	/**
	 * Pointer to device primal Psi
	 */
	real_t *devVecPrimalPsi;
	/**
	 * Pointer to device dual Xi
	 */
	real_t *devVecDualXi;
	/**
	 * Pointer to device dual Psi
	 */
	real_t *devVecDualPsi;
	/**
	 * Pointer to device updated xi
	 */
	real_t *devVecUpdateXi;
	/**
	 * Pointer to device update psi
	 */
	real_t *devVecUpdatePsi;
	/**
	 * Pointer array for device pointers of state
	 */
	real_t **devPtrVecX;
	/**
	 * Pointer array for device pointers of control
	 */
	real_t **devPtrVecU;
	/**
	 * Pointer array for device pointers of reduced
	 * decision variable
	 */
	real_t **devPtrVecV;
	/**
	 * Pointer array for device pointers of accelerated
	 * psi
	 */
	real_t **devPtrVecAcceleratedPsi;
	/**
	 * Pointer array for device pointers of accelerated
	 * xi
	 */
	real_t **devPtrVecAcceleratedXi;
	/**
	 * Pointer array for device pointers of primal psi
	 */
	real_t **devPtrVecPrimalPsi;
	/**
	 * Pointer array for device pointer of primal xi
	 */
	real_t **devPtrVecPrimalXi;
	/**
	 * Pointer array for primal infeasibility Hx-z
	 */
	real_t *devVecPrimalInfeasibilty;
	/**
	 * Pointer for cost Q
	 */
	real_t *devVecQ;
	/**
	 * Pointer for cost R
	 */
	real_t *devVecR;
	/**
	 * Pointer array for device pointers of cost Q
	 */
	real_t **devPtrVecQ;
	/**
	 * Pointer array for device pointers of cost R
	 */
	real_t **devPtrVecR;
	/**
	 * step size
	 */
	real_t stepSize;
	/**
	 * Flag Factor step
	 */
	bool FlagFactorStep;
	/*
	 * primal Infeasibilty
	 */
	real_t *vecPrimalInfs;
};

#endif /* SMPCONTROLLERCLASS_CUH_ */
