/*
 * networkClass.cuh
 *
 *  Created on: Feb 21, 2017
 *      Author: Ajay K. Sampathirao, P. Sopasakis
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
#include "DefinitionHeader.h"


/*TODO Document this class */
/*TODO Implement getters */

/**
 *
 */
class DWNnetwork{
public:

	/**
	 * Constructor of a DWN entity from a given JSON file.
	 *
 	 * @param pathToFile filename of a JSON file containing
	 * 		     a representation of the DWN.
	 */
	DWNnetwork(
		string pathToFile);
	/**
	 * Destructor of the DWN entity that removes it from the CPU
 	 */
	~DWNnetwork();

	/*TODO REMOVE Friendship */
	friend class Engine;

	/*TODO REMOVE Friendship */
	friend class SMPCController;

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
	/*TODO alpha1 and alpha2 are not parameters of the DWN; they are costs. */
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
	 *TODO REMOVE
 	 */
	real_t *matDiagPrecnd;
	/**
	 *TODO SHOULD THIS BE HERE?
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
	 *TODO SHOULD THIS BE HERE?
 	 */
	real_t *prevU;

	/*TODO the step size should not be here */

};
#endif /* NETWORKCLASS_CUH_ */
