#include <cuda_device_runtime_api.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"

#include "Configuration.h"
#include "SmpcController.cuh"
#include "test/Testing.cuh"

int main(void){
	int runtimeVersion = -1 , driverVersion = -1;
	cudaRuntimeGetVersion(&runtimeVersion);
	_CUDA(cudaDriverGetVersion(&driverVersion));
	cout << runtimeVersion << " " << driverVersion << endl;
	Testing *myTesting = new Testing();
	_ASSERT( myTesting->testNetwork() );
	_ASSERT( myTesting->testScenarioTree() );
	_ASSERT( myTesting->testForecaster() );
	_ASSERT( myTesting->testControllerConfig() );
	_ASSERT( myTesting->testEngineTesting() );
	_ASSERT( myTesting->testSmpcController());
	//try{
		//cout << myTesting->testScenarioTree() << endl;
	//}catch (exception &e){
	//cout << e.what() << __LINE__ << endl;
	//}

	string pathToNetworkFile = "../dataFiles/network.json";
	string pathToScenarioTreeFile = "../dataFiles/scenarioTree.json";
	string pathToForecastFile = "../dataFiles/forecastor.json";
	string pathToTestFile = "../dataFiles/testVariables.json";
	string pathToSmpcConfigFile = "../dataFiles/controllerConfig.json";

	cout << "bye bye" << endl;
	return 0;
}
