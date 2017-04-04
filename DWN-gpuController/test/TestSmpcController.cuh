/*
 * testSmpcController.cuh
 *
 *  Created on: Apr 3, 2017
 *      Author: control
 */

#ifndef TESTSMPCCONTROLLER_CUH_
#define TESTSMPCCONTROLLER_CUH_
#include "../SmpcController.cuh"
//#include "../DwnNetwork.cuh"
//#include "../Forecaster.cuh"
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
#define VARNAME_TEST_DUAL_UPDATE_XI "dualUpdateXi"
#define VARNAME_TEST_DUAL_UPDATE_PSI "dualUpdatePsi"
#define VARNAME_TEST_PRIMAL_INFS_XI "primalInfsXi"
#define VARNAME_TEST_PRIMAL_INFS_PSI "primalInfsPsi"


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
	~TestSmpcController();
private:

	/**
	 * Function to compare the deviceArray with the input from the json file
	 */
	template<typename T>
	uint_t compareDeviceArray(T* deviceArray);

	/**
	 * Function to set a deviceArray with the input from the json file
	 */
	template<typename T>
	void setDeviceArray(T* deviceArrary, uint_t dim);
	/*
	 * path to the json file which contain the test parameters
	 */
	string PathToFileSmpcTest;
	/**
	 * Object to retrive the information from the json file
	 */
	rapidjson::Value a;
};




#endif /* TESTSMPCCONTROLLER_CUH_ */
