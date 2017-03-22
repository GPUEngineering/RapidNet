/*
 * Network.cu
 *
 *  Created on: Mar 14, 2017
 *      Author: Ajay K. Sampathirao, P. Sopasakis
 */
#include <iostream>
#include <cstdio>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"


#include "DWNnetwork.cuh"

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
		a = jsonDocument[VARNAME_NX];
		_ASSERT(a.IsArray());
		NX = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NU];
		_ASSERT(a.IsArray());
		NU = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_ND];
		_ASSERT(a.IsArray());
		ND = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NE];
		_ASSERT(a.IsArray());
		NE = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NV];
		_ASSERT(a.IsArray());
		NV = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_N];
		_ASSERT(a.IsArray());
		uint_t N = (uint_t) a[0].GetDouble();
		matA = new real_t[NX * NX];
		a = jsonDocument[VARNAME_A];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matA[i] = a[i].GetDouble();
		matB = new real_t[NX * NU];
		a = jsonDocument[VARNAME_B];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matB[i] = a[i].GetDouble();
		matGd = new real_t[NX * ND];
		a = jsonDocument[VARNAME_GD];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matGd[i] = a[i].GetDouble();
		matE = new real_t[NE * NU];
		a = jsonDocument[VARNAME_E];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matE[i] = a[i].GetDouble();
		matEd = new real_t[NE * ND];
		a = jsonDocument[VARNAME_ED];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matEd[i] = a[i].GetDouble();
		matL = new real_t[NU * NV];
		a = jsonDocument[VARNAME_L];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matL[i] = a[i].GetDouble();
		matLhat = new real_t[NU * ND];
		a = jsonDocument[VARNAME_LHAT];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matLhat[i] = a[i].GetDouble();
		vecXmin = new real_t[NX];
		a = jsonDocument[VARNAME_XMIN];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXmin[i] = a[i].GetDouble();
		vecXmax = new real_t[NX];
		a = jsonDocument[VARNAME_XMAX];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXmax[i] = a[i].GetDouble();
		vecXsafe = new real_t[NX];
		a = jsonDocument[VARNAME_XSAFE];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXsafe[i] = a[i].GetDouble();
		vecUmin = new real_t[NU];
		a = jsonDocument[VARNAME_UMIN];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecUmin[i] = a[i].GetDouble();
		vecUmax = new real_t[NU];
		a = jsonDocument[VARNAME_UMAX];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecUmax[i] = a[i].GetDouble();
		matCostW = new real_t[NU * NU];
		a = jsonDocument[VARNAME_COSTW];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matCostW[i] = a[i].GetDouble();
		vecCostAlpha1 = new real_t[NU];
		a = jsonDocument["costAlpha1"];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecCostAlpha1[i] = a[i].GetDouble();
		vecCostAlpha2 = new real_t[NU*N];
		a = jsonDocument[VARNAME_ALPHA2];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecCostAlpha2[i] = a[i].GetDouble();
		a = jsonDocument[VARNAME_PENALITY_X];
		penaltyStateX = a[0].GetDouble();
		a = jsonDocument["penaltySafetyX"];
		penaltySafetyX = a[0].GetDouble();
		matDiagPrecnd = new real_t[(NU + 2*NX) * N];
		a = jsonDocument[VARNAME_DIAG_PRCND];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matDiagPrecnd[i] = a[i].GetDouble();
		currentX = new real_t[NX];
		a = jsonDocument[VARNAME_CURRENT_X];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			currentX[i] = a[i].GetDouble();
		prevUhat = new real_t[NU];
		a = jsonDocument[VARNAME_PREV_UHAT];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevUhat[i] = a[i].GetDouble();
		prevU = new real_t[NU];
		a = jsonDocument[VARNAME_PREV_U];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevU[i] = a[i].GetDouble();
		prevV = new real_t[NV];
		a = jsonDocument[VARNAME_PREV_V];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevV[i] = a[i].GetDouble();
		delete [] readBuffer;
		readBuffer = NULL;
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
	matA = NULL;
	matB = NULL;
	matGd = NULL;
	matE = NULL;
	matEd = NULL;
	matL = NULL;
	matLhat = NULL;
	vecXmin = NULL;
	vecXmax = NULL;
	vecXsafe = NULL;
	vecUmin = NULL;
	vecUmax = NULL;
	matCostW = NULL;
	vecCostAlpha1 = NULL;
	vecCostAlpha2 = NULL;
	matDiagPrecnd = NULL;
	cout << "freeing the memory of the network \n";
}
