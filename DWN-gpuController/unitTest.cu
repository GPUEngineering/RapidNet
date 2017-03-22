/*
 * unitTest.cu
 *
 *  Created on: Mar 15, 2017
 *      Author: control
 */
#include <iostream>
#include <cstdio>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"

using namespace std;

typedef int uint_t;
typedef float real_t;
#include "Configuration.h"
#include "unitTestHeader.cuh"

unitTest::unitTest(string pathToFile){
	cout << "allocating memory for the test matrices \n";
	const char* fileName = pathToFile.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		a = jsonDocument["nx"];
		assert(a.IsArray());
		NX = (uint_t) a[0].GetDouble();
		a = jsonDocument["nu"];
		assert(a.IsArray());
		NU = (uint_t) a[0].GetDouble();
		a = jsonDocument["nv"];
		assert(a.IsArray());
		NV = (uint_t) a[0].GetDouble();
		matR = new real_t[NV * NV];
		a = jsonDocument["matR"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matR[i] = a[i].GetDouble();
		delete [] readBuffer;
	}
	fclose(infile);
}

void unitTest::checkObjectiveMatR(real_t *engineMatR){
	real_t *hostEngineMatR = new real_t[NV*NV];
	_CUDA( cudaMemcpy( hostEngineMatR, engineMatR, NV*NV*sizeof(real_t), cudaMemcpyDeviceToHost) );
	/*
	for (int iRow = 0; iRow < NV; iRow++){
		for(int iCol = 0; iCol < NV; iCol++){
			cout<< matR[iRow*NV + iCol] - hostEngineR[iRow*NV + iCol] << " ";
		}
		cout << "\n";
	}*/
	for( int i = 0; i < NV*NV; i++){
		if(abs(matR[i] - hostEngineMatR[i]) > 1e-3)
			cout<< matR[i] << " " << hostEngineMatR[i] << " " << i << " ";
	}
	delete hostEngineMatR;
}

unitTest::~unitTest(){
	delete [] matR;
}

