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

#ifndef NETWORKCLASS_CUH_
#define NETWORKCLASS_CUH_
#define VARNAME_NX "nx"
#define VARNAME_NU "nu"
#define VARNAME_ND "nd"
#define VARNAME_NE "ne"
#define VARNAME_NV "nv"
#define VARNAME_N  "N"
#define VARNAME_A  "matA"
#define VARNAME_B  "matB"
#define VARNAME_GD "matGd"
#define VARNAME_E  "matE"
#define VARNAME_ED "matEd"
#define VARNAME_L  "matL"
#define VARNAME_LHAT "matLhat"
#define VARNAME_XMIN "vecXmin"
#define VARNAME_XMAX "vecXmax"
#define VARNAME_XSAFE "vecXsafe"
#define VARNAME_UMIN "vecUmin"
#define VARNAME_UMAX "vecUmax"
#define VARNAME_COSTW "costW"
#define VARNAME_ALPHA1 "costAlpha1"
#define VARNAME_ALPHA2 "costAlpha2"
#define VARNAME_PENALITY_X "penaltyStateX"
#define VARNAME_PENALITY_XS "penaltySafetyX"
#define VARNAME_DIAG_PRCND "matDiagPrecnd"
#define VARNAME_CURRENT_X "currentX"
#define VARNAME_PREV_UHAT "prevUhat"
#define VARNAME_PREV_U "prevU"
#define VARNAME_PREV_V "prevV"
#include "Configuration.h"


/**
 * @todo Document this class
 * @todo Implement getters
 * @todo REMOVE Friendships
 */
class DwnNetwork{
public:

	/**
	 * Constructor of a DWN entity from a given JSON file.
	 *
 	 * @param pathToFile filename of a JSON file containing
	 * 		     a representation of the DWN.
	 */
	DwnNetwork(
		string pathToFile);
	/**
	 * Destructor of the DWN entity that removes it from the CPU
 	 */
	~DwnNetwork();

	friend class Engine; /*TODO REMOVE Friendship */

	friend class SmpcController; /*TODO REMOVE Friendship */

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
	 *
 	 */
	uint_t NV;
	/**
	 * Matrix A of the system dynamics.
	 * Typically this is equal to the identity.
 	 */
	real_t *matA;
	/**
	 * Matrix B of the system dynamics.
 	 */
	real_t *matB;
	/**
	 * Matrix Gd of the system dynamics.
 	 */
	real_t *matGd;
	/**
	 * Matrix E of the input-disturbance coupling.
 	 */
	real_t *matE;
	/**
	 * Matrix Ed of the input-disturbance coupling.
 	 */
	real_t *matEd;
	/**
	 * Minimum allowed volume of water in each tank.
 	 */
	real_t *vecXmin;
	/**
	 * Maximum allowed volume of water in each tank.
 	 */
	real_t *vecXmax;
	/**
	 * Vector of safety volumes of water in each tank.
 	 */
	real_t *vecXsafe;
	/**
	 * Lower bounds on actuator signals.
 	 */
	real_t *vecUmin;
	/**
	 * Upper bounds on actuator signals.
 	 */
	real_t *vecUmax;
	/**
	 * ?
 	 */
	real_t *matL;
	/**
	 * ?
 	 */
	real_t *matLhat;
	/**
	 *
 	 */
	real_t *matCostW;
	/**
	 *
 	 */
	real_t *vecCostAlpha1;
	/**
	 *
 	 */
	/* @todo alpha1 and alpha2 are not parameters of the DWN; they are costs. */
	real_t *vecCostAlpha2;
	/**
	 * ?
 	 */
	real_t penaltyStateX;
	/**
	 *
 	 */
	real_t penaltySafetyX;
	/**
	 * @todo REMOVE
 	 */
	real_t *matDiagPrecnd;
	/**
	 * @todo SHOULD THIS BE HERE?
 	 */
	real_t *currentX;
	/**
	 *
 	 */
	real_t *prevUhat;
	/**
	 *
 	 */
	real_t *prevV;
	/**
	 * @todo SHOULD THIS BE HERE?
 	 */
	real_t *prevU;


};
#endif /* NETWORKCLASS_CUH_ */
