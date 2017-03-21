/*
 * Network.cu
 *
 *  Created on: Mar 14, 2017
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

#include "networkHeader.cuh"

DWNnetwork::DWNnetwork(string pathToFile){
	cout << "allocating memory for the network \n";
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
		a = jsonDocument["nd"];
		assert(a.IsArray());
		ND = (uint_t) a[0].GetDouble();
		a = jsonDocument["ne"];
		assert(a.IsArray());
		NE = (uint_t) a[0].GetDouble();
		a = jsonDocument["nv"];
		assert(a.IsArray());
		NV = (uint_t) a[0].GetDouble();
		a = jsonDocument["N"];
		assert(a.IsArray());
		uint_t N = (uint_t) a[0].GetDouble();
		matA = new real_t[NX * NX];
		a = jsonDocument["matA"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matA[i] = a[i].GetDouble();
		matB = new real_t[NX * NU];
		a = jsonDocument["matB"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matB[i] = a[i].GetDouble();
		matGd = new real_t[NX * ND];
		a = jsonDocument["matGd"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matGd[i] = a[i].GetDouble();
		matE = new real_t[NE * NU];
		a = jsonDocument["matE"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matE[i] = a[i].GetDouble();
		matEd = new real_t[NE * ND];
		a = jsonDocument["matEd"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matEd[i] = a[i].GetDouble();
		matL = new real_t[NU * NV];
		a = jsonDocument["matL"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matL[i] = a[i].GetDouble();
		matLhat = new real_t[NU * ND];
		a = jsonDocument["matLhat"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matLhat[i] = a[i].GetDouble();
		vecXmin = new real_t[NX];
		a = jsonDocument["vecXmin"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXmin[i] = a[i].GetDouble();
		vecXmax = new real_t[NX];
		a = jsonDocument["vecXmax"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXmax[i] = a[i].GetDouble();
		vecXsafe = new real_t[NX];
		a = jsonDocument["vecXsafe"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXsafe[i] = a[i].GetDouble();
		vecUmin = new real_t[NU];
		a = jsonDocument["vecUmin"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecUmin[i] = a[i].GetDouble();
		vecUmax = new real_t[NU];
		a = jsonDocument["vecUmax"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecUmax[i] = a[i].GetDouble();
		matCostW = new real_t[NU * NU];
		a = jsonDocument["costW"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matCostW[i] = a[i].GetDouble();
		vecCostAlpha1 = new real_t[NU];
		a = jsonDocument["costAlpha1"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecCostAlpha1[i] = a[i].GetDouble();
		vecCostAlpha2 = new real_t[NU*N];
		a = jsonDocument["costAlpha2"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecCostAlpha2[i] = a[i].GetDouble();
		a = jsonDocument["penaltyStateX"];
		penaltyStateX = a[0].GetDouble();
		a = jsonDocument["penaltySafetyX"];
		penaltySafetyX = a[0].GetDouble();
		matDiagPrecnd = new real_t[(NU + 2*NX) * N];
		//cout << "allocating memory matDiagPrecnd " << N <<"\n";
		a = jsonDocument["matDiagPrecnd"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matDiagPrecnd[i] = a[i].GetDouble();
		currentX = new real_t[NX];
		a = jsonDocument["currentX"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			currentX[i] = a[i].GetDouble();
		prevUhat = new real_t[NU];
		a = jsonDocument["prevUhat"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevUhat[i] = a[i].GetDouble();
		prevU = new real_t[NU];
		a = jsonDocument["prevU"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevU[i] = a[i].GetDouble();
		prevV = new real_t[NV];
		a = jsonDocument["prevV"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevV[i] = a[i].GetDouble();
		a = jsonDocument["stepSize"];
		assert(a.IsArray());
		STEP_SIZE = a[0].GetDouble();
		delete [] readBuffer;
	}
	fclose(infile);
}

DWNnetwork::~DWNnetwork(){
	delete [] matA;
	delete [] matB;
	delete [] matGd;
	delete [] matE;
	delete [] matEd;
	delete [] matL;
	delete [] matLhat;
	delete [] vecXmin;
	delete [] vecXmax;
	delete [] vecXsafe;
	delete [] vecUmin;
	delete [] vecUmax;
	delete [] matCostW;
	delete [] vecCostAlpha1;
	delete [] vecCostAlpha2;
	delete [] matDiagPrecnd;
	cout << "freeing the memory of the network \n";
}
