#include <cuda_device_runtime_api.h>
#include "cublas_v2.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"

#include "Configuration.h"
#include "SmpcController.cuh"
#include "test/Testing.cuh"

int main(void){
	uint_t TESTING = 0;
	startTicToc();
	if (TESTING){
		Testing *myTesting = new Testing();
		_ASSERT( myTesting->testNetwork() );
		_ASSERT( myTesting->testScenarioTree() );
		_ASSERT( myTesting->testForecaster() );
		_ASSERT( myTesting->testControllerConfig() );
		_ASSERT( myTesting->testEngineTesting() );
		_ASSERT( myTesting->testSmpcController());
	}
	real_t time;
	string pathToControlOutput = "../systemData/controlOutput10844.json";
	string pathToControllerConfig = "../systemData/controllerConfig10844.json";
	//fstream controlOutputJson( pathToControlOutput.c_str(), ios::out);
	fstream controlOutputJson;
	controlOutputJson.open( pathToControlOutput.c_str(), fstream::out);
	/*real_t a[2] = {1, 2};
	controlOutputJson << " control " << a << endl;
	controlOutputJson << " next control " << 2 << endl;*/

	SmpcController *dwnController = new SmpcController( pathToControllerConfig );
	uint_t timeInstance = 0;

	while (timeInstance < 10){
		dwnController->getForecaster()->predictDemand( timeInstance );
		dwnController->getForecaster()->predictPrices( timeInstance );
		if( timeInstance == 0){
			dwnController->initialiseSmpcController();
		}
		tic();
		dwnController->controlAction( controlOutputJson );
		time = toc();
		cout << "time lapsed " << time << " milliseconds" << endl;
		controlOutputJson << "\"time" << timeInstance <<"\": [" << time << "]" << endl;
		timeInstance = timeInstance + 1;
		dwnController->moveForewardInTime();
		cout << timeInstance << " ";
	}
	cout << endl;

	cout << "economic kpi " << dwnController->getEconomicKpi(timeInstance) << endl;
	cout << "smooth kpi " << dwnController->getSmoothKpi(timeInstance) << endl;
	cout << "safety kpi " << dwnController->getSafetyKpi(timeInstance) << endl;
	cout << "network utility kpi " << dwnController->getNetworkKpi(timeInstance) << endl;

	uint_t nx = dwnController->getDwnNetwork()->getNumTanks();
	uint_t nu = dwnController->getDwnNetwork()->getNumControls();
	uint_t nd = dwnController->getDwnNetwork()->getNumDemands();
	real_t *currentState = new real_t[nx];
	real_t *prevControl = new real_t[nu];
	real_t *prevDemand = new real_t[nd];
	/*
	for(uint_t iSize = 0; iSize < nx; iSize++)
		currentState[iSize] = dwnController->getSmpcConfiguration()->getCurrentX()[iSize];
	for(uint_t iSize = 0; iSize < nu; iSize++)
		prevControl[iSize] = dwnController->getSmpcConfiguration()->getPrevU()[iSize];
	for(uint_t iSize = 0; iSize < nd; iSize++)
		prevDemand[iSize] = dwnController->getSmpcConfiguration()->getPrevDemand()[iSize];
	dwnController->moveForewardInTime();

	cout << "State "<< endl;
	for(uint_t iSize = 0; iSize < nx; iSize++)
		cout << currentState[iSize] - dwnController->getSmpcConfiguration()->getCurrentX()[iSize] << " ";
	cout << "Control" << endl;
	for(uint_t iSize = 0; iSize < nu; iSize++)
		cout<< prevControl[iSize] - dwnController->getSmpcConfiguration()->getPrevU()[iSize] << " ";
	cout << "Demand" << endl;
	for(uint_t iSize = 0; iSize < nd; iSize++)
		cout << iSize << " " << prevDemand[iSize] -
		dwnController->getSmpcConfiguration()->getPrevDemand()[iSize] << " ";
	cout << nx << " " << nu << " " << nd << endl;*/
	delete dwnController;
	controlOutputJson.close();
	delete [] currentState;
	delete [] prevControl;
	delete [] prevDemand;
	/**/
	//try{
		//cout << myTesting->testScenarioTree() << endl;
	//}catch (exception &e){
	//cout << e.what() << __LINE__ << endl;
	//}


	//string pathToScenarioTreeFile = "../dataFiles/scenarioTree.json";
	//string pathToForecastFile = "../dataFiles/forecastor.json";
	//string pathToTestFile = "../dataFiles/testVariables.json";
	//string pathToSmpcConfigFile = "../dataFiles/controllerConfig.json";

	cout << "bye bye" << endl;
	return 0;
}
