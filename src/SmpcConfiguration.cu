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
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
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
		weightPrice = 1;
		weightSmooth = 1;
		weightSafety = 1;
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream configurationJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(configurationJsonStream);
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
		for (rapidjson::SizeType i = 0; i < a.Size(); i++){
			matCostW[i] = weightSmooth*a[i].GetFloat();
		}
		a = jsonDocument[VARNAME_PENALITY_X];
		penaltyStateX = a[0].GetFloat();
		a = jsonDocument[VARNAME_PENALITY_XS];
		penaltySafetyX = weightSafety*a[0].GetFloat();
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
		a = jsonDocument[PATH_NETWORK_FILE];
		_ASSERT(a.IsString());
		pathToNetwork = a.GetString();
		a = jsonDocument[PATH_SCENARIO_TREE_FILE];
		_ASSERT(a.IsString());
		pathToScenarioTree = a.GetString();
		a = jsonDocument[PATH_FORECASTER_FILE];
		_ASSERT(a.IsString());
		pathToForecaster = a.GetString();
		a = jsonDocument[ALGORITHM_CONTROL];
		_ASSERT(a.IsString());
		algorithmName = a.GetString();
		a = jsonDocument[STYLE_HESSIAN_ORACAL];
		_ASSERT(a.IsString());
		styleHessianOracle = a.GetString();
		a = jsonDocument[VARNAME_LBFGS_BUFFER_SIZE];
		_ASSERT(a.IsArray());
		lbfgsBufferSize = (uint_t) a[0].GetFloat();
		pathToConfiguration = pathToFile;
		delete [] readBuffer;
		readBuffer = NULL;
		fclose(infile);
		infile = NULL;
	}
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
uint_t SmpcConfiguration::getLbfgsBufferSize(){
	return lbfgsBufferSize;
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

string SmpcConfiguration::getPathToNetwork(){
	return pathToNetwork;
}

string SmpcConfiguration::getPathToScenarioTree(){
	return pathToScenarioTree;
}

string SmpcConfiguration::getPathToForecaster(){
	return pathToForecaster;
}

string SmpcConfiguration::getOptimisationAlgorithm(){
	return algorithmName;
}

string SmpcConfiguration::getHessianOracleNamaAlgorithm(){
	return styleHessianOracle;
}

real_t SmpcConfiguration::getWeightEconomical(){
	return weightPrice;
}
/*
 * Get the path to the controller configuration file
 */
string SmpcConfiguration::getPathToControllerConfig(){
	return pathToConfiguration;
}

/**
 * update the level in the tanks
 * @param   state    updated state
 */
void SmpcConfiguration::setCurrentState(real_t* state){
	for(uint_t iSize = 0; iSize < NX ; iSize++)
		currentX[iSize] = state[iSize];
}
/**
 * update the previous control actions
 * @param    control  previous control action
 */
void SmpcConfiguration::setPreviousControl(real_t* control){
	for(uint_t iSize = 0; iSize < NU ; iSize++)
		prevU[iSize] = control[iSize];
}
/**
 * update the previous demand
 * @param    demand    previous demand
 */
void SmpcConfiguration::setpreviousdemand(real_t* demand){
	for(uint_t iSize = 0; iSize < ND ; iSize++)
		prevDemand[iSize] = demand[iSize];
}
/*
 * update the current state from the controller
 * configuration file
 */
void SmpcConfiguration::setCurrentState(){
	const char* fileName = pathToConfiguration.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value currentStateJson;
	FILE* configurationFile = fopen(fileName, "r");
	char* readBuffer = new char[65536];
	rapidjson::FileReadStream configurationJsonStream(configurationFile, readBuffer, sizeof(readBuffer));
	jsonDocument.ParseStream(configurationJsonStream);
	currentStateJson = jsonDocument[VARNAME_CURRENT_X];
	_ASSERT(currentStateJson.IsArray());
	for (rapidjson::SizeType i = 0; i < currentStateJson.Size(); i++)
		currentX[i] = currentStateJson[i].GetFloat();
	delete [] readBuffer;
	readBuffer = NULL;
}
/*
 * update the previous control from the controller
 * configuration file
 */
void SmpcConfiguration::setPreviousControl(){
	const char* fileName = pathToConfiguration.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value previousControlJson;
	FILE* configurationFile = fopen(fileName, "r");
	char* readBuffer = new char[65536];
	rapidjson::FileReadStream configurationJsonStream(configurationFile, readBuffer, sizeof(readBuffer));
	jsonDocument.ParseStream(configurationJsonStream);
	previousControlJson = jsonDocument[VARNAME_PREV_U];
	_ASSERT(previousControlJson.IsArray());
	for (rapidjson::SizeType i = 0; i < previousControlJson.Size(); i++)
		prevU[i] = previousControlJson[i].GetFloat();
	delete [] readBuffer;
	readBuffer = NULL;
}

/*
 * update the previous demand from the controller
 * configuration file
 */
void SmpcConfiguration::setPreviousDemand(){
	const char* fileName = pathToConfiguration.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value previousDemandJson;
	FILE* configurationFile = fopen(fileName, "r");
	char* readBuffer = new char[65536];
	rapidjson::FileReadStream configurationJsonStream(configurationFile, readBuffer, sizeof(readBuffer));
	jsonDocument.ParseStream(configurationJsonStream);
	previousDemandJson = jsonDocument[VARNAME_PREV_DEMAND];
	_ASSERT(previousDemandJson.IsArray());
	for (rapidjson::SizeType i = 0; i < previousDemandJson.Size(); i++)
		prevU[i] = previousDemandJson[i].GetFloat();
	delete [] readBuffer;
	readBuffer = NULL;
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


