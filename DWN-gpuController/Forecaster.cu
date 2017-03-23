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
#include "Forecaster.cuh"


Forecaster::Forecaster(string pathToFile){
	cout << "allocating memory for the forecaster \n"; /*TODO Remove prints */
	const char* fileName = pathToFile.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		exit(100); /*TODO never use `exit`; throw an exception instead */
	}else{
		char* readBuffer = new char[65536]; /*TODO Make sure this is a good practice */
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument[VARNAME_N];
		_ASSERT( a.IsArray() );
		nPredHorizon = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_DIM_DEMAND];
		_ASSERT( a.IsArray() );
		dimDemand = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_DIM_PRICE];
		_ASSERT( a.IsArray() );
		dimPrices = (uint_t) a[0].GetDouble();
		dHat = new real_t[dimDemand * nPredHorizon];
		a = jsonDocument[VARNAME_DHAT];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			dHat[i] = a[i].GetDouble();
		alphaHat = new real_t[dimPrices * nPredHorizon];
		a = jsonDocument[VARNAME_DHAT];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			alphaHat[i] = a[i].GetDouble();
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
}

uint_t Forecaster::getPredHorizon(){
	return nPredHorizon;
}

uint_t Forecaster::getDimDemand(){
	return dimDemand;
}

uint_t Forecaster::getDimPrice(){
	return dimPrices;
}

real_t* Forecaster::getNominalDemand(){
	return dHat;
}

real_t* Forecaster::getNomialPrices(){
	return alphaHat;
}

Forecaster::~Forecaster(){
	delete [] dHat;
	delete [] alphaHat;
	dHat = NULL;
	alphaHat = NULL;
	cout << "freeing the memory of the forecaster \n"; /*TODO Remove prints */
}
