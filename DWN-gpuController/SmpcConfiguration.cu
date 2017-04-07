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
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"
#include "SmpcConfiguration.cuh"


SmpcConfiguration::SmpcConfiguration(string pathToFile){
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
		NX = (uint_t) a[0].GetFloat();
		a = jsonDocument[VARNAME_NU];
		_ASSERT(a.IsArray());
		NU = (uint_t) a[0].GetFloat();
		a = jsonDocument[VARNAME_ND];
		_ASSERT(a.IsArray());
		ND = (uint_t) a[0].GetFloat();
		a = jsonDocument[VARNAME_NV];
		_ASSERT(a.IsArray());
		NV = (uint_t) a[0].GetFloat();
		a = jsonDocument[VARNAME_N];
		_ASSERT(a.IsArray());
		uint_t N = (uint_t) a[0].GetFloat();
		matL = new real_t[NU * NV];
		a = jsonDocument[VARNAME_L];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matL[i] = a[i].GetFloat();
		matLhat = new real_t[NU * ND];
		a = jsonDocument[VARNAME_LHAT];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matLhat[i] = a[i].GetFloat();
		matCostW = new real_t[NU * NU];
		a = jsonDocument[VARNAME_COSTW];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matCostW[i] = a[i].GetFloat();
		a = jsonDocument[VARNAME_PENALITY_X];
		penaltyStateX = a[0].GetFloat();
		a = jsonDocument[VARNAME_PENALITY_XS];
		penaltySafetyX = a[0].GetFloat();
		matDiagPrecnd = new real_t[(NU + 2*NX) * N];
		a = jsonDocument[VARNAME_DIAG_PRCND];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matDiagPrecnd[i] = a[i].GetFloat();
		currentX = new real_t[NX];
		a = jsonDocument[VARNAME_CURRENT_X];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			currentX[i] = a[i].GetFloat();
		/*prevUhat = new real_t[NU];
		a = jsonDocument[VARNAME_PREV_UHAT];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevUhat[i] = a[i].GetFloat();*/
		prevU = new real_t[NU];
		a = jsonDocument[VARNAME_PREV_U];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevU[i] = a[i].GetFloat();
		prevDemand = new real_t[ND];
		a = jsonDocument[VARNAME_PREV_DEMAND];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			prevDemand[i] = (real_t) a[i].GetFloat();
		a = jsonDocument[VARNAME_STEP_SIZE];
		_ASSERT(a.IsArray());
		stepSize = (real_t) a[0].GetFloat();
		a = jsonDocument[VARNAME_MAX_ITER];
		_ASSERT(a.IsArray());
		maxIteration = (uint_t) a[0].GetFloat();
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
}

uint_t SmpcConfiguration::getNX(){
	return NX;
}
uint_t SmpcConfiguration::getNU(){
	return NU;
}
uint_t SmpcConfiguration::getND(){
	return ND;
}
uint_t SmpcConfiguration::getNV(){
	return NV;
}
real_t* SmpcConfiguration::getMatL(){
	return matL;
}

real_t* SmpcConfiguration::getMatLhat(){
	return matLhat;
}

real_t* SmpcConfiguration::getMatPrcndDiag(){
	return matDiagPrecnd;
}
real_t* SmpcConfiguration::getCostW(){
	return matCostW;
}

real_t* SmpcConfiguration::getCurrentX(){
	return currentX;
}

real_t* SmpcConfiguration::getPrevU(){
	return prevU;
}

real_t* SmpcConfiguration::getPrevDemand(){
	return prevDemand;
}

real_t SmpcConfiguration::getPenaltyState(){
	return penaltyStateX;
}

real_t SmpcConfiguration::getPenaltySafety(){
	return penaltySafetyX;
}

uint_t SmpcConfiguration::getMaxIterations(){
	return maxIteration;
}

real_t SmpcConfiguration::getStepSize(){
	return stepSize;
}
SmpcConfiguration::~SmpcConfiguration(){
	delete [] matL;
	delete [] matLhat;
	delete [] matCostW;
	delete [] matDiagPrecnd;
	delete [] currentX;
	delete [] prevU;
	delete [] prevDemand;

	matL = NULL;
	matLhat = NULL;
	matCostW = NULL;
	matDiagPrecnd = NULL;
	currentX = NULL;
	prevDemand = NULL;
	prevU = NULL;
}


