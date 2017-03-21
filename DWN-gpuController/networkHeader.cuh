/*
 * networkClass.cuh
 *
 *  Created on: Feb 21, 2017
 *      Author: control
 */

#ifndef NETWORKCLASS_CUH_
#define NETWORKCLASS_CUH_

class DWNnetwork{
public:
	DWNnetwork(string pathToFile);
	~DWNnetwork();
	friend class Engine;
	friend class SMPCController;
private:
	uint_t NX, NU, ND, NE, NV;
	real_t *matA, *matB, *matGd, *matE, *matEd, *vecXmin, *vecXmax, *vecXsafe, *vecUmin,
	*vecUmax, *matL, *matLhat, *matCostW, *vecCostAlpha1, *vecCostAlpha2;
	real_t penaltyStateX;
	real_t penaltySafetyX;
	real_t *matDiagPrecnd;
	real_t *currentX;
	real_t *prevUhat;
	real_t *prevV;
	real_t *prevU;
	real_t STEP_SIZE;

};
#endif /* NETWORKCLASS_CUH_ */
