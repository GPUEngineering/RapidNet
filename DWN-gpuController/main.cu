#include <cuda_device_runtime_api.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"

#include "DefinitionHeader.h"
#include "SMPControllerHeader.cuh"
//#include "cudaKernalHeader.cuh"
//#include "EngineHeader.cuh"
//#include "cudaKernal.cu"
//#include "SMPControllerClass.cuh"

/*
__global__ void increment ( int* dev_a, int p, int N){
	int tid = threadIdx.x;
	if ( tid < N)
		dev_a [tid] = dev_a [tid] + p;
}
*/
int main(void){
	int runtimeVersion = -1 , driverVersion = -1;
	cudaRuntimeGetVersion(&runtimeVersion);
	_CUDA(cudaDriverGetVersion(&driverVersion));
	cout << runtimeVersion << " " << driverVersion << endl;
	string pathToNetworkFile = "../dataFiles/network.json";
	string pathToForecastFile = "../dataFiles/forecastor.json";
	string pathToTestfile = "../dataFiles/testVariables.json";
	DWNnetwork myNetwork( pathToNetworkFile );
	Forecaster myForecaster( pathToForecastFile );
	unitTest myTestor( pathToTestfile );
	Engine myEngine(&myNetwork, &myForecaster, &myTestor);
	SMPCController myController( &myEngine);
	myEngine.initialiseForecastDevice();
	myEngine.initialiseSystemDevice();
	myEngine.factorStep();
	//myController.solveStep();
	//myEngine.testStupidFunction();
	//myEngine.testPrecondtioningFunciton();

	//myEngine.testInverse();
	cout << "bye bye \n" << endl;
	return 0;
}
