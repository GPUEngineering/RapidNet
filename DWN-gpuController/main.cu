/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <cstdio>
#include <string>
#include <cuda_device_runtime_api.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"


//using namespace rapidjson;
using namespace std;

typedef int uint_t;
typedef float real_t;

#include "DefinitionHeader.h"
#include "networkClass.cuh"
#include "forecastClass.cuh"
#include "unitTestClass.cuh"
#include "EngineClass.cuh"
#include "SMPControllerClass.cuh"

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
	string pathToNetworkFile = "../network.json";
	string pathToForecastFile = "../forecastor.json";
	string pathToTestfile = "../testVariables.json";
	DWNnetwork myNetwork( pathToNetworkFile );
	Forecastor myForecastor( pathToForecastFile );
	unitTest myTestor( pathToTestfile );
	Engine myEngine(&myNetwork, &myForecastor, &myTestor);
	SMPCController myController( &myEngine);
	//myEngine.testStupidFunction();
	myEngine.initialiseForecastDevice();
	myEngine.initialiseSystemDevice();
	//myEngine.testPrecondtioningFunciton();
	myEngine.factorStep();
	//myEngine.testInverse();
	/*
	int *a, *b ;
	int *dev_a ;
	int N = 10 ;

	a = new int[N] ;
	b = new int[N] ;


	for (int i = 0; i < N ; i ++){
		a[i] = i;
	}


	cudaMalloc((void**)&dev_a, N * sizeof(int) );

	cudaMemcpy(dev_a, a , N*sizeof(int), cudaMemcpyHostToDevice);
	increment<<<1,N+3>>>(dev_a, 4, N);
	cudaMemcpy(b, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0 ; i < N ; i++){
		cout << a[i] << b[i]<<endl;
	}
	cout <<"Hello from the remote !!"<<endl;

	cudaFree(dev_a);
	delete [] a;
	delete [] b;
	*/
	cout << "bye bye \n" << endl;
	return 0;
}
