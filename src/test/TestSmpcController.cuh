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


#ifndef TESTSMPCCONTROLLER_CUH_
#define TESTSMPCCONTROLLER_CUH_
#include "../SmpcController.cuh"
#include "../rapidjson/document.h"
#include "../rapidjson/rapidjson.h"
#include "../rapidjson/filereadstream.h"

#define VARNAME_TEST_EXTAPOLATION "theta"
#define VARNAME_TEST_ACCELE_XI "acceleXi"
#define VARNAME_TEST_ACCELE_PSI "accelePsi"
#define VARNAME_TEST_UPDATE_XI "updateXi"
#define VARNAME_TEST_UPDATE_PSI "updatePsi"
#define VARNAME_TEST_XI "xi"
#define VARNAME_TEST_PSI "psi"
#define VARNAME_TEST_X "X"
#define VARNAME_TEST_U "U"
#define VARNAME_TEST_DUALX "dualX"
#define VARNAME_TEST_DUALU "dualU"
#define VARNAME_TEST_PRIMALX "primalX"
#define VARNAME_TEST_PRIMALU "primalU"
#define VARNAME_TEST_FINAL_UPDATE_XI "finalUpdateXi"
#define VARNAME_TEST_FINAL_UPDATE_PSI "finalUpdatePsi"
#define VARNAME_TEST_FINAL_XI "finalXi"
#define VARNAME_TEST_FINAL_PSI "finalPsi"
#define VARNAME_TEST_PRIMAL_INFS_XI "primalInfsXi"
#define VARNAME_TEST_PRIMAL_INFS_PSI "primalInfsPsi"
#define VARNAME_TEST_FIXED_POINT_RESIDUAL_XI "fixedPointResidualXi"
#define VARNAME_TEST_FIXED_POINT_RESIDUAL_PSI "fixedPointResidualPsi"
#define VARNAME_TEST_TEMP_V  "tempV"
#define VARNAME_TEST_X_DIR "fbeHessianDirXdir"
#define VARNAME_TEST_U_DIR "fbeHessianDirUdir"
#define VARNAME_TEST_GRAD_FBE_XI "fbeGradXi"
#define VARNAME_TEST_GRAD_FBE_PSI "fbeGradPsi"
#define VARNAME_TEST_LBFGS_CURRENT_YVEC_XI "lbfgsCurrentYvecXi"
#define VARNAME_TEST_LBFGS_CURRENT_YVEC_PSI "lbfgsCurrentYvecPsi"
#define VARNAME_TEST_LBFGS_PREVIOUS_YVEC_XI "lbfgsPreviousYvecXi"
#define VARNAME_TEST_LBFGS_PREVIOUS_YVEC_PSI "lbfgsPreviousYvecPsi"
#define VARNAME_TEST_LBFGS_MAT_S "matS"
#define VARNAME_TEST_LBFGS_MAT_Y "matY"
#define VARNAME_TEST_LBFGS_COL "colLbfgs"
#define VARNAME_TEST_LBFGS_MEM "memLbfgs"
#define VARNAME_TEST_LBFGS_INV_RHO "vecInvRho"
#define VARNAME_TEST_LBFGS_H "H"
#define VARNAME_TEST_LBFGS_DIR_XI "lbfgsDirXi"
#define VARNAME_TEST_LBFGS_DIR_PSI "lbfgsDirPsi"
#define VARNAME_TEST_UPDATE_LBFGS_MAT_S "updateMatS"
#define VARNAME_TEST_UPDATE_LBFGS_MAT_Y "updateMatY"
#define VARNAME_TEST_UPDATE_LBFGS_COL "updateColLbfgs"
#define VARNAME_TEST_UPDATE_LBFGS_MEM "updateMemLbfgs"
#define VARNAME_TEST_UPDATE_LBFGS_INV_RHO "updateVecInvRho"
#define VARNAME_TEST_UPDATE_LBFGS_H "updateH"
#define VARNAME_TEST_FBE_COST "fbeObjDual"
#define VARNAME_TEST_UPDATE_RESIDUAL_XI "updateResidualXi"
#define VARNAME_TEST_UPDATE_RESIDUAL_PSI "updateResidualPsi"
#define VARNAME_TEST_TAU "tau"



class TestSmpcController : public SmpcController{
public :
	/**
	 * Constructor that create a TestSmpcController object which is derived from the SmpcController object
	 *
	 * @param  myForecaster  Forecaster object
	 * @param  myEngine      Engine object
	 * @param  mySmpcConfig  SmpcConfiguration object that contain the controller configuration
	 */
	TestSmpcController(Forecaster *myForecaster, Engine *myEngine, SmpcConfiguration *mySmpcConfig);

	/**
	 * Constructor that create a TestSmpcController object which is derived from the SmpcController object
	 *
	 * @param  pathToConfigFile  path to the SmpcController configuration file
	 */
	TestSmpcController(string pathToConfigFile);
	/**
	 * Function to test the dualExtrapolation function
	 */
	uint_t testExtrapolation();

	/**
	 * Function to test the solve function
	 */
	uint_t testSoveStep();

	/**
	 * Function to test the proximal function
	 */
	uint_t testProximalStep();

	/**
	 * Function to test the dual update function
	 */
	uint_t testDualUpdate();

	/**
	 * Function of test the fixed point residual
	 */
	uint_t testFixedPointResidual();

	/**
	 * Function to test the lbfgs direction
	 */
	uint_t testHessianOracalGlobalFbe();

	/**
	 * Function to test the fbe gradient
	 */

	uint_t testFbeGradient();

	/**
	 * function to test the lbfgs direction
	 */
	uint_t testLbfgsDirection();

	/**
	 * function to test the value FBE
	 */
	uint_t testValueFbe();
	/**
	 * function to test the line search update in
	 * the FBE
	 */
	uint_t testFbeLineSearch();
	/**
	 * function to test the dual update in the global FBE algorithm
	 */
	uint_t testFbeDualUpdate();
	/**
	 * function that update the current fixed point residual for calculating
	 * LBFSG direction
	 */
	uint_t testUpdateFixedPointResidualNamaAlgorithm();
	/**
	 * function to test the line search update in
	 * the AME
	 */
	uint_t testAmeLineSearch();
	/**
	 * Destructor
	 */
	~TestSmpcController();
private:

	/**
	 * Function to compare the deviceArray with the input from the json file
	 */
	template<typename T>
	uint_t compareDeviceArray(T* deviceArray);

	/**
	 * Function to compare the deviceArrayA and deviceArrayB
	 * @param     deviceArrayA    device array
	 * @param     deviceArrayB    host array
	 * @param     dim             dimension of the array
	 */
	template<typename T>
	uint_t compareDeviceArray(T* deviceArrayA, T* hostArrayA, uint_t dim);

	/**
	 * Function to set a deviceArray with the input from the json file
	 */
	template<typename T>
	void setDeviceArray(T* deviceArrary, uint_t dim);
	/*
	 * path to the json file which contain the test parameters
	 * smpc for the apg algorithm
	 */
	string pathToFileSmpc;
	/*
	 * path to the json file which contain the test parameters
	 * smpc for the globalFbe algorithm
	 */
	string pathToFileGlobalFbeSmpc;
	/*
	 * path to the json file which contain the test parameters
	 * smpc for the NAMA algorithm
	 */
	string pathToFileNamaSmpc;
	/**
	 * Object to retrive the information from the json file
	 */
	rapidjson::Value a;
};




#endif /* TESTSMPCCONTROLLER_CUH_ */
