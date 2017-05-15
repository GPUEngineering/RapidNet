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
#define VARNAME_A  "matA"
#define VARNAME_B  "matB"
#define VARNAME_GD "matGd"
#define VARNAME_E  "matE"
#define VARNAME_ED "matEd"
#define VARNAME_XMIN "vecXmin"
#define VARNAME_XMAX "vecXmax"
#define VARNAME_XSAFE "vecXsafe"
#define VARNAME_UMIN "vecUmin"
#define VARNAME_UMAX "vecUmax"
#define VARNAME_ALPHA1 "costAlpha1"
#include "Configuration.h"


/**
 * Drinking water network consists of tanks, valves, pumps and distribution points.
 * The network manager should transport the water from the storage tanks to the
 * demand/distribution points via series of pumps and valves. This dynamics of the
 * water network can be represented with a mass-conservation equation. All
 * physical components in the network should be operated in their physical limits given
 * as constraints on the system.
 * The DwnNetwork class represent the the drinking water model. This class includes
 *      - number of tanks
 *      - number of pumps and valves
 *      - number of demand points
 *      - number of junctions
 *      - matrix A, B, Gd, E and Ed that represent the topology of the network
 *      - physical limits on the tanks (minimum and maximum water level), control
 *        ( minimum and maximum flows)
 *      - constant production and treatment on this water.
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
	 * Return the number of tanks
	 */
	uint_t getNumTanks();

	/**
	 * Return the number of pumps and valves
	 */
	uint_t getNumControls();

	/**
	 * Return the number of demands
	 */
	uint_t getNumDemands();

	/**
	 * Return the number of mix nodes
	 */
	uint_t getNumMixNodes();

	/**
	 * Return the pointer to the matrix A
	 */
	real_t* getMatA();

	/**
	 * Return the pointer to the matrix B
	 */
	real_t* getMatB();

	/**
	 * Return the pointer to the matrix Gd
	 */
	real_t* getMatGd();

	/**
	 * Return the pointer to matrix E
	 */
	real_t* getMatE();

	/**
	 * Return the pointer to matrix Ed
	 */
	real_t* getMatEd();

	/**
	 * Return the pointer to minimum volume of the tanks
	 */
	real_t* getXmin();

	/**
	 * Return the pointer to the maximum volume of the tanks
	 */
	real_t* getXmax();

	/**
	 * Return the pointer to the safety level of the tanks
	 */
	real_t* getXsafe();

	/**
	 * Return the pointer to the minimum control level
	 */
	real_t* getUmin();

	/**
	 * Return the pointer to the maximum control level
	 */
	real_t* getUmax();

	/**
	 * Return the pointer to the production/treatment costs
	 */
	real_t* getAlpha();
	/**
	 * Destructor of the DWN entity that removes it from the CPU
 	 */
	~DwnNetwork();

private:

	/**
	 * Number of tanks
 	 */
	uint_t nTanks;
	/**
	 * Number of actuators
 	 */
	uint_t nControl;
	/**
	 * Number of demand nodes
 	 */
	uint_t nDemand;
	/**
	 * Number of mixing nodes
 	 */
	uint_t nMixNodes;
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
	 * production/ treatment cost
 	 */
	real_t *vecCostAlpha1;
};
#endif /* NETWORKCLASS_CUH_ */
