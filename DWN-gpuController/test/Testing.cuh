/*
 * Test.cuh
 *
 *  Created on: Mar 28, 2017
 *      Author: control
 */

#ifndef TEST_CUH_
#define TEST_CUH_
#include "../SmpcController.cuh"
#include "../rapidjson/document.h"
#include "../rapidjson/rapidjson.h"
#include "../rapidjson/filereadstream.h"
#define VARNAME_BETA "beta"
#define VARNAME_UHAT "uHat"
#define VARNAME_VEC_E "vecE"
#define VARNAME_SCE_NOD "scenarioNodes"
#define VARNAME_PHI "Phi"
#define VARNAME_PSI "Psi"
#define VARNAME_OMEGA "omega"
#define VARNAME_D "d"
#define VARNAME_F "f"
#define VARNAME_GBAR "Gbar"
#define VARNAME_THETA "Theta"
#define VARNAME_G "g"
#define VARNAME_BBAR "Bbar"
#define VARNAME_SYS_F "sysF"
#define VARNAME_SYS_G "sysG"
#define VARNAME_TEST_XMIN "xmin"
#define VARNAME_TEST_XMAX "xmax"
#define VARNAME_TEST_XS "xs"
#define VARNAME_TEST_UMIN "umin"
#define VARNAME_TEST_UMAX "umax"



class Testing{
public:
	Testing();
	int testNetwork();
	int testScenarioTree();
	int testForecaster();
	int testControllerConfig();
	int testEngineTesting();
	~Testing();
private:
	template<typename T> int compareArray(T* arrayA);
	template<typename T> int compareDeviceArray(T* arrayA);
	template<typename T> int compareDeviceScenarioArray(T* arrayA, uint_t *nodes, uint_t dim);
	//int compareArray(real_t* arrayA);
	string pathToFileNetwork;
	string pathToFileForecaster;
	string pathToFileScenarioTree;
	string pathToFileControllerConfig;
	string pathToFileEnigne;
	rapidjson::Value a;
};


#endif /* TEST_CUH_ */
