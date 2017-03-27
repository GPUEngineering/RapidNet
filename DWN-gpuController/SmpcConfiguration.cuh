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

	uint_t getNV();

	real_t* getMatL();

	real_t* getMatLhat();

	real_t* getMatPrcndDiag();

	real_t* getCostW();

	real_t* getCurrentX();

	real_t* getPrevU();

	real_t* getPrevUhat();

	real_t* getPrevV();

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
	 * Number of mixing nodes
	 */
	uint_t NE;
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
	 * TODO should be replaced and calculated directly in Engine
	 * previous control particular solution
	 */
	real_t *prevUhat;
	/**
	 * TODO should be replaced and calculated directly in the Engine
	 * previous reduced control decision variable
	 */
	real_t *prevV;
	/**
	 * Previous control
	 */
	real_t *prevU;
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
