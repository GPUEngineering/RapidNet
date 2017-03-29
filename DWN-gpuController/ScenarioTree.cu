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
#include "ScenarioTree.cuh"


ScenarioTree::ScenarioTree( string pathToFile ){
	const char* fileName = pathToFile.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		throw std::logic_error("Error in opening the file");
		//exit(100); /*TODO never use `exit`; throw an exception instead */
	}else{
		char* readBuffer = new char[65536]; /*TODO Make sure this is a good practice */
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument[VARNAME_N];
		_ASSERT( a.IsArray() );
		nPredHorizon = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_K];
		_ASSERT( a.IsArray() );
		nScenario = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NODES];
		_ASSERT( a.IsArray() );
		nNodes = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NUM_NONLEAF];
		_ASSERT(a.IsArray());
		nNonleafNodes = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NUM_CHILD_TOT];
		_ASSERT( a.IsArray() );
		nChildrenTot = (uint_t) a[0].GetDouble();
		stageArray = new uint_t[nNodes];
		a = jsonDocument[VARNAME_STAGES];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			stageArray[i] = (uint_t) a[i].GetDouble();
		nodesPerStage = new uint_t[nPredHorizon];
		a = jsonDocument[VARNAME_NODES_PER_STAGE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nodesPerStage[i] = (uint_t) a[i].GetDouble();
		nodesPerStageCumul = new uint_t[nPredHorizon + 1];
		a = jsonDocument[VARNAME_NODES_PER_STAGE_CUMUL];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nodesPerStageCumul[i] = (uint_t) a[i].GetDouble();
		leaveArray = new uint_t[nScenario];
		a = jsonDocument[VARNAME_LEAVES];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			leaveArray[i] = (uint_t) a[i].GetDouble();
		childArray = new uint_t[nChildrenTot];
		a = jsonDocument[VARNAME_CHILDREN];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			childArray[i] = (uint_t) a[i].GetDouble();
		ancestorArray = new uint_t[nNodes];
		a = jsonDocument[VARNAME_ANCESTOR];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			ancestorArray[i] = (uint_t) a[i].GetDouble();
		nChildArray = new uint_t[nNonleafNodes];
		a = jsonDocument[VARNAME_NUM_CHILDREN];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nChildArray[i] = (uint_t) a[i].GetDouble();
		nChildCumulArray = new uint_t[nNodes];
		a = jsonDocument[VARNAME_NUM_CHILD_CUMUL];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nChildCumulArray[i] = (uint_t) a[i].GetDouble();
		probNodeArray = new real_t[nNodes];
		a = jsonDocument[VARNAME_PROB_NODE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			probNodeArray[i] = a[i].GetDouble();
		a = jsonDocument[VARNAME_DIM_DEMAND];
		_ASSERT( a.IsArray() );
		dimDemand = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_DIM_PRICE];
		_ASSERT( a.IsArray() );
		dimPrice = (uint_t) a[0].GetDouble();
		errorDemandArray = new real_t[nNodes * dimDemand];
		a = jsonDocument[VARNAME_DEMAND_NODE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			errorDemandArray[i] = a[i].GetDouble();
		errorPriceArray = new real_t[nNodes * dimPrice];
		a = jsonDocument[VARNAME_PRICE_NODE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			errorPriceArray[i] = a[i].GetDouble();
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
}

uint_t ScenarioTree::getPredHorizon(){
	return nPredHorizon;
}

uint_t ScenarioTree::getNumScenarios(){
	return nScenario;
}

uint_t ScenarioTree::getNumNodes(){
	return nNodes;
}

uint_t ScenarioTree::getNumChildrenTot(){
	return nChildrenTot;
}

uint_t ScenarioTree::getNumNonleafNodes(){
	return nNonleafNodes;
}

uint_t* ScenarioTree::getStageNodes(){
	return stageArray;
}

uint_t* ScenarioTree::getNodesPerStage(){
	return nodesPerStage;
}

uint_t* ScenarioTree::getNodesPerStageCumul(){
	return nodesPerStageCumul;
}

uint_t* ScenarioTree::getLeaveArray(){
	return leaveArray;
}

uint_t* ScenarioTree::getChildArray(){
	return childArray;
}

uint_t* ScenarioTree::getAncestorArray(){
	return ancestorArray;
}

uint_t* ScenarioTree::getNumChildren(){
	return nChildArray;
}

uint_t* ScenarioTree::getNumChildrenCumul(){
	return nChildCumulArray;
}

real_t* ScenarioTree::getProbArray(){
	return probNodeArray;
}

real_t* ScenarioTree::getErrorDemandArray(){
	return errorDemandArray;
}

real_t* ScenarioTree::getErrorPriceArray(){
	return errorPriceArray;
}

ScenarioTree::~ScenarioTree(){
	delete [] stageArray;
	delete [] nodesPerStage;
	delete [] nodesPerStageCumul;
	delete [] leaveArray;
	delete [] childArray;
	delete [] ancestorArray;
	delete [] nChildArray;
	delete [] nChildCumulArray;
	delete [] probNodeArray;
	delete [] errorDemandArray;
	delete [] errorPriceArray;
	stageArray = NULL;
	nodesPerStage = NULL;
	nodesPerStageCumul = NULL;
	leaveArray = NULL;
	childArray = NULL;
	ancestorArray = NULL;
	nChildArray = NULL;
	nChildCumulArray = NULL;
	probNodeArray = NULL;
	errorDemandArray = NULL;
	errorPriceArray = NULL;
}


