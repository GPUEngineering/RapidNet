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

#ifndef SMPCCONFIGURATION_CUH_
#define SMPCCONFIGURATION_CUH_
#include "Configuration.h"
#define VARNAME_NX "nx"
#define VARNAME_NU "nu"
#define VARNAME_ND "nd"
#define VARNAME_NE "ne"
#define VARNAME_NV "nv"
#define VARNAME_N  "N"
#define VARNAME_L  "matL"
#define VARNAME_LHAT "matLhat"
#define VARNAME_COSTW "costW"
#define VARNAME_PENALITY_X "penaltyStateX"
#define VARNAME_PENALITY_XS "penaltySafetyX"
#define VARNAME_DIAG_PRCND "matDiagPrecnd"
#define VARNAME_CURRENT_X "currentX"
#define VARNAME_PREV_UHAT "prevUhat"
#define VARNAME_PREV_U "prevU"
#define VARNAME_PREV_V "prevV"
#define VARNAME_PREV_DEMAND "prevDemand"
#define VARNAME_STEP_SIZE "stepSize"
#define VARNAME_MAX_ITER "maxIterations"
class SmpcConfiguration{
public:

	/**
	 * Constructor of a SmpcConfiguration entity from a
	 * given JSON file.
	 *
	 * @param pathToFile filename of a JSON file containing
	 * 		     a representation of the controller
	 * 		     configuration.
	 */
	SmpcConfiguration(string pathToFile);
	/** ---GETTER'S FOR CONTROLLER CONFIGURATION---*/
	/**
	 * Dimension of the state
	 */
	uint_t getNX();
	/**
	 * Dimension of the control
	 */
	uint_t getNU();
	/**
	 * Dimension of the demand
	 */
	uint_t getND();
	/**
	 *  Reduced dimension of the control after eliminating the
	 *  control-disturbance equality
	 */
	uint_t getNV();
	/**
	 * Affine space representation for the control-disturbance equality
	 */
	real_t* getMatL();
	/**
	 * matrix to calculate the particular solution
	 */
	real_t* getMatLhat();
	/**
	 * Diagonal preconditioner used
	 */
	real_t* getMatPrcndDiag();
	/**
	 * Smooth operation cost matrix
	 */
	real_t* getCostW();
	/**
	 * Current state/level of water in the tanks
	 */
	real_t* getCurrentX();
	/**
	 * Previous control/actuators for the valves and pumps
	 */
	real_t* getPrevU();
	/**
	 * Previous demand
	 */
	real_t* getPrevDemand();
	/**
	 * Weight that penalise the constraints on the tank level
	 */
	real_t getPenaltyState();
	/**
	 * Weight that penalise the safety volume of the tank
	 */
	real_t getPenaltySafety();
	/**
	 * Get the maximum iterations number of iterations
	 */
	uint_t getMaxIterations();
	/**
	 * Get the step size
	 */
	real_t getStepSize();
	/**
	 * Default destructor to free the memory
	 */
	~SmpcConfiguration();
private:

	/**
	 * Number of tanks
	 */
	uint_t NX;
	/**
	 * Number of actuators
	 */
	uint_t NU;
	/**
	 * Number of demand nodes
	 */
	uint_t ND;
	/**
	 * Dimension of the control after eliminating the equality constraints
	 */
	uint_t NV;
	/**
	 * TODO should be calculated by the controller
	 * Affine subspace for the control-disturbance coupling
	 */
	real_t *matL;
	/**
	 * TODO should be calculated by the controller
	 * Matrix to calculate the particular solution
	 */
	real_t *matLhat;
	/**
	 * Smooth operation, penalise the change of control
	 */
	real_t *matCostW;
	/**
	 * Weight that penalise the constraints on the tank level
	 */
	real_t penaltyStateX;
	/**
	 * Weight that penalise the safety volume of the tank
	 */
	real_t penaltySafetyX;
	/**
	 * TODO should be calculated by the controller
	 * diagonal precondition matrix. Just have the diagonal elements of the
	 * precondition matrix of the single scenario
	 */
	real_t *matDiagPrecnd;
	/**
	 * Current water level of the tanks
	 */
	real_t *currentX;
	/**
	 * Previous control
	 */
	real_t *prevU;
	/**
	 * Previous demand
	 */
	real_t *prevDemand;
	/**
	 *  stepsize for the APG
	 */
	real_t stepSize;
	/**
	 * maximum number of iterations of the APG algorithm
	 */
	uint_t maxIteration;
};



#endif /* SMPCCONFIGURATION_CUH_ */
