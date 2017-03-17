/*
 * unitTestClass.cuh
 *
 *  Created on: Mar 8, 2017
 *      Author: control
 */

#ifndef UNITTESTCLASS_CUH_
#define UNITTESTCLASS_CUH_



class unitTest{
public:
	unitTest(string pathToFile);
	~unitTest();
	void checkObjectiveMatR(real_t *engineMatR);
	friend class Engine;
private:
	uint_t NX, NU, NV;
	real_t *matR;
};

#endif /* UNITTESTCLASS_CUH_ */
