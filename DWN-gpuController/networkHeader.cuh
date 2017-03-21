/*
 * networkClass.cuh
 *
 *  Created on: Feb 21, 2017
 *      Author: Ajay K. Sampathirao, P. Sopasakis
 */

#ifndef NETWORKCLASS_CUH_
#define NETWORKCLASS_CUH_

/*TODO Rename this file into DWNNetwork.cuh */
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
	 *
 	 */
	~DWNnetwork();
	
	/*TODO REMOVE Friendship */
	friend class Engine;

	/*TODO REMOVE Friendship */
	friend class SMPCController;

private:

	/**
	 *
 	 */
	uint_t NX;
	/**
	 *
 	 */
	uint_t NU;
	/**
	 *
 	 */
	uint_t ND;
	/**
	 *
 	 */
	uint_t NE;
	/**
	 *
 	 */
	uint_t NV;
	/**
	 *
 	 */
	real_t *matA;
	/**
	 *
 	 */
	real_t *matB;
	/**
	 *
 	 */
	real_t *matGd;
	/**
	 *
 	 */
	real_t *matE;
	/**
	 *
 	 */
	real_t *matEd;
	/**
	 *
 	 */
	real_t *vecXmin;
	/**
	 *
 	 */
	real_t *vecXmax;
	/**
	 *
 	 */
	real_t *vecXsafe;
	/**
	 *
 	 */
	real_t *vecUmin;
	/**
	 *
 	 */
	real_t *vecUmax;
	/**
	 *
 	 */
	real_t *matL;
	/**
	 *
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
	real_t *vecCostAlpha2;
	/**
	 *
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
	 *
 	 */
	real_t *prevU;

	/*TODO the step size should not be here */
	real_t STEP_SIZE;

};
#endif /* NETWORKCLASS_CUH_ */
