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

class Testing{
public:
	Testing();
	int testNetwork();
	int testScenarioTree();
	int testForecaster();
	int testControllerConfig();
	~Testing();
private:
	template<typename T> int compareArray(T* arrayA);
	//int compareArray(real_t* arrayA);
	string pathToFileNetwork;
	string pathToFileForecaster;
	string pathToFileScenarioTree;
	string pathToFileControllerConfig;
	string pathToFileEnigne;
	rapidjson::Value a;
};


#endif /* TEST_CUH_ */
