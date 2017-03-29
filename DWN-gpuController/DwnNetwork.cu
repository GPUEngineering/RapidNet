/*
 *    GPU-accelerated scenario-based stochastic MPC for the operational
 *    management of drinking water networks.
 *    Copyright (C) 2017 Ajay. K. Sampathirao and P. Sopasakis
 *
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */
#include <iostream>
#include <cstdio>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"


#include "DwnNetwork.cuh"

DwnNetwork::DwnNetwork(string pathToFile){
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
		nTanks = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NU];
		_ASSERT(a.IsArray());
		nControl = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_ND];
		_ASSERT(a.IsArray());
		nDemand = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NE];
		_ASSERT(a.IsArray());
		nMixNodes = (uint_t) a[0].GetDouble();
		/*a = jsonDocument[VARNAME_N];
		_ASSERT(a.IsArray());
		uint_t N = (uint_t) a[0].GetDouble();*/
		matA = new real_t[nTanks * nTanks];
		a = jsonDocument[VARNAME_A];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matA[i] = a[i].GetDouble();
		matB = new real_t[nTanks * nControl];
		a = jsonDocument[VARNAME_B];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matB[i] = a[i].GetDouble();
		matGd = new real_t[nTanks * nDemand];
		a = jsonDocument[VARNAME_GD];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matGd[i] = a[i].GetDouble();
		matE = new real_t[nMixNodes *nControl];
		a = jsonDocument[VARNAME_E];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matE[i] = a[i].GetDouble();
		matEd = new real_t[nMixNodes *nDemand];
		a = jsonDocument[VARNAME_ED];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matEd[i] = a[i].GetDouble();
		vecXmin = new real_t[nTanks];
		a = jsonDocument[VARNAME_XMIN];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXmin[i] = a[i].GetDouble();
		vecXmax = new real_t[nTanks];
		a = jsonDocument[VARNAME_XMAX];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXmax[i] = a[i].GetDouble();
		vecXsafe = new real_t[nTanks];
		a = jsonDocument[VARNAME_XSAFE];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecXsafe[i] = a[i].GetDouble();
		vecUmin = new real_t[nControl];
		a = jsonDocument[VARNAME_UMIN];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecUmin[i] = a[i].GetDouble();
		vecUmax = new real_t[nControl];
		a = jsonDocument[VARNAME_UMAX];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecUmax[i] = a[i].GetDouble();
		vecCostAlpha1 = new real_t[nControl];
		a = jsonDocument[VARNAME_ALPHA1];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			vecCostAlpha1[i] = a[i].GetDouble();
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
}

uint_t DwnNetwork::getNumTanks(){
	return nTanks;
}

uint_t DwnNetwork::getNumControls(){
	return nControl;
}

uint_t DwnNetwork::getNumDemands(){
	return nDemand;
}

uint_t DwnNetwork::getNumMixNodes(){
	return nMixNodes;
}

real_t* DwnNetwork::getMatA(){
	return matA;
}

real_t* DwnNetwork::getMatB(){
	return matB;
}

real_t* DwnNetwork::getMatGd(){
	return matGd;
}

real_t* DwnNetwork::getMatE(){
	return matE;
}

real_t* DwnNetwork::getMatEd(){
	return matEd;
}

real_t* DwnNetwork::getXmin(){
	return vecXmin;
}

real_t* DwnNetwork::getXmax(){
	return vecXmax;
}

real_t* DwnNetwork::getXsafe(){
	return vecXsafe;
}

real_t* DwnNetwork::getUmin(){
	return vecUmin;
}

real_t* DwnNetwork::getUmax(){
	return vecUmax;
}

real_t* DwnNetwork::getAlpha(){
	return vecCostAlpha1;
}

DwnNetwork::~DwnNetwork(){
	delete [] matA;
	delete [] matB;
	delete [] matGd;
	delete [] matE;
	delete [] matEd;
	delete [] vecXmin;
	delete [] vecXmax;
	delete [] vecXsafe;
	delete [] vecUmin;
	delete [] vecUmax;
	delete [] vecCostAlpha1;
	matA = NULL;
	matB = NULL;
	matGd = NULL;
	matE = NULL;
	matEd = NULL;
	vecXmin = NULL;
	vecXmax = NULL;
	vecXsafe = NULL;
	vecUmin = NULL;
	vecUmax = NULL;
	vecCostAlpha1 = NULL;
}
