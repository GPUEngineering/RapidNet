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

#include "SmpcConfiguration.cuh"


SmpcConfiguration::SmpcConfiguration(string pathToFile){
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
		matCostW = new real_t[NU * NU];
		a = jsonDocument[VARNAME_COSTW];
		_ASSERT(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matCostW[i] = a[i].GetDouble();
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
		stepSize = 1e-4;
		maxIteration = 500;
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
}

real_t* SmpcConfiguration::getMatL(){
	return matL;
}

real_t* SmpcConfiguration::getMatLhat(){
	return matLhat;
}

real_t* SmpcConfiguration::getCostW(){
	return matCostW;
}

real_t* SmpcConfiguration::getCurrentX(){
	return currentX;
}


real_t* SmpcConfiguration::getPrevUhat(){
	return prevUhat;
}

real_t* SmpcConfiguration::getPrevV(){
	return prevV;
}

real_t* SmpcConfiguration::getPrevU(){
	return prevU;
}

SmpcConfiguration::~SmpcConfiguration(){
	delete [] matL;
	delete [] matLhat;
	delete [] matCostW;
	delete [] matDiagPrecnd;
	delete [] currentX;
	delete [] prevU;
	delete [] prevUhat;
	delete [] prevV;
	matL = NULL;
	matLhat = NULL;
	matCostW = NULL;
	matDiagPrecnd = NULL;
	currentX = NULL;
	prevUhat = NULL;
	prevU = NULL;
	prevV = NULL;
	cout << "freeing the memory of the network \n";
}


