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
#include "Forecaster.cuh"


Forecaster::Forecaster(string pathToFile){
	const char* fileName = pathToFile.c_str();
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536]; /*TODO Make sure this is a good practice */
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		jsonValue = jsonDocument[VARNAME_N];
		_ASSERT( jsonValue.IsArray() );
		nPredHorizon = (uint_t) jsonValue[0].GetFloat();
		jsonValue = jsonDocument[VARNAME_SIM_HORIZON];
		_ASSERT( jsonValue.IsArray() );
		simHorizon = (uint_t) jsonValue[0].GetFloat();
		jsonValue = jsonDocument[VARNAME_DIM_DEMAND];
		_ASSERT( jsonValue.IsArray() );
		dimDemand = (uint_t) jsonValue[0].GetFloat();
		jsonValue = jsonDocument[VARNAME_DIM_PRICES];
		_ASSERT( jsonValue.IsArray() );
		dimPrices = (uint_t) jsonValue[0].GetFloat();
		nominalDemand = new real_t[dimDemand * nPredHorizon];
		/*jsonValue = jsonDocument[VARNAME_DHAT];
		_ASSERT( jsonValue.IsArray() );
		for (rapidjson::SizeType i = 0; i < jsonValue.Size(); i++)
			nominalDemand[i] = jsonValue[i].GetFloat();*/
		nominalPrice = new real_t[dimPrices * nPredHorizon];
		/*jsonValue = jsonDocument[VARNAME_ALPHAHAT];
		_ASSERT( jsonValue.IsArray() );
		for (rapidjson::SizeType i = 0; i < jsonValue.Size(); i++)
			nominalPrice[i] = jsonValue[i].GetFloat();*/
		itrNominalDemand = jsonDocument.MemberBegin();
		itrNominalPrices = jsonDocument.MemberBegin();
		delete [] readBuffer;
		readBuffer = NULL;
		fclose(infile);
		infile = NULL;
	}
}

uint_t Forecaster::getPredHorizon(){
	return nPredHorizon;
}

uint_t Forecaster::getSimHorizon(){
	return simHorizon;
}

uint_t Forecaster::getDimDemand(){
	return dimDemand;
}

uint_t Forecaster::getDimPrice(){
	return dimPrices;
}

real_t* Forecaster::getNominalDemand(){
	return nominalDemand;
}

real_t* Forecaster::getNominalPrices(){
	return nominalPrice;
}

uint_t Forecaster::predictDemand(uint_t simTime){
	itrNominalDemand = jsonDocument.MemberBegin() + 4 + 2*simTime;
	if(itrNominalDemand != jsonDocument.MemberEnd() ){
		jsonValue = jsonDocument[itrNominalDemand->name.GetString()];
		//cout<< " size of the nominal demand "<<jsonValue.Size() << " " << itrNominalDemand->name.GetString() << endl;
		for(uint_t iCount = 0; iCount < jsonValue.Size(); iCount++){
			nominalDemand[iCount]= jsonValue[iCount].GetFloat();
		}
		return 1;
	}else{
		return 0;
	}
}

uint_t Forecaster::predictPrices(uint_t simTime){
	itrNominalPrices = jsonDocument.MemberBegin() + 5 + 2*simTime;
	if(itrNominalDemand != jsonDocument.MemberEnd() ){
		jsonValue = jsonDocument[itrNominalPrices->name.GetString()];
		//cout<< " size of the nominal prices "<< jsonValue.Size()<< " " << itrNominalPrices->name.GetString() << endl;
		for(uint_t iCount = 0; iCount < jsonValue.Size(); iCount++){
			nominalPrice[iCount]= jsonValue[iCount].GetFloat();
		}
		return 1;
	}else{
		return 0;
	}
}

Forecaster::~Forecaster(){
	delete [] nominalDemand;
	delete [] nominalPrice;
	nominalDemand = NULL;
	nominalPrice = NULL;
}
