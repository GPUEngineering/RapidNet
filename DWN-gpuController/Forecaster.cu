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
		/*TODO Do not hard-code variable names */
		a = jsonDocument[VARNAME_N];
		/*TODO Do you check whether there is such a node in the JSON file? */
		/*TODO Do not use `assert` - throw EXCEPTIONS instead */
		_ASSERT( a.IsArray() );
		N = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_K];
		_ASSERT( a.IsArray() );
		K = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NODES];
		_ASSERT( a.IsArray() );
		nNodes = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NUM_NONLEAF];
		_ASSERT(a.IsArray());
		nNonleafNodes = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_NUM_CHILD_TOT];
		_ASSERT( a.IsArray() );
		nChildrenTot = (uint_t) a[0].GetDouble();
		a = jsonDocument[VARNAME_DIM_NODE];
		_ASSERT( a.IsArray() );
		dimDemand = (uint_t) a[0].GetDouble();
		stages = new uint_t[nNodes];
		a = jsonDocument[VARNAME_STAGES];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			stages[i] = (uint_t) a[i].GetDouble();
		nodesPerStage = new uint_t[N];
		a = jsonDocument[VARNAME_NODES_PER_STAGE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nodesPerStage[i] = (uint_t) a[i].GetDouble();
		nodesPerStageCumul = new uint_t[N+1];
		a = jsonDocument[VARNAME_NODES_PER_STAGE_CUMUL];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nodesPerStageCumul[i] = (uint_t) a[i].GetDouble();
		leaves = new uint_t[K];
		a = jsonDocument[VARNAME_LEAVES];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			leaves[i] = (uint_t) a[i].GetDouble();
		children = new uint_t[nChildrenTot];
		a = jsonDocument[VARNAME_CHILDREN];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			children[i] = (uint_t) a[i].GetDouble();
		ancestor = new uint_t[nNodes];
		a = jsonDocument[VARNAME_ANCESTOR];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			ancestor[i] = (uint_t) a[i].GetDouble();
		nChildren = new uint_t[nNonleafNodes];
		a = jsonDocument[VARNAME_NUM_CHILDREN];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nChildren[i] = (uint_t) a[i].GetDouble();
		nChildrenCumul = new uint_t[nNodes];
		a = jsonDocument[VARNAME_NUM_CHILD_CUMUL];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nChildrenCumul[i] = (uint_t) a[i].GetDouble();
		probNode = new real_t[nNodes];
		a = jsonDocument[VARNAME_PROB_NODE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			probNode[i] = a[i].GetDouble();
		valueNode = new real_t[nNodes * dimDemand];
		a = jsonDocument[VARNAME_VALUE_NODE];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			valueNode[i] = a[i].GetDouble();
		dHat = new real_t[dimDemand * N];
		a = jsonDocument[VARNAME_DHAT];
		_ASSERT( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			dHat[i] = a[i].GetDouble();
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
}

Forecaster::~Forecaster(){
	delete [] stages;
	delete [] nodesPerStage;
	delete [] nodesPerStageCumul;
	delete [] leaves;
	delete [] children;
	delete [] ancestor;
	delete [] nChildren;/**/
	delete [] nChildrenCumul;
	delete [] probNode;
	delete [] valueNode;
	delete [] dHat;
	stages = NULL;
	nodesPerStage = NULL;
	nodesPerStageCumul = NULL;
	leaves = NULL;
	children = NULL;
	ancestor = NULL;
	nChildren = NULL;
	nChildrenCumul = NULL;
	probNode = NULL;
	valueNode = NULL;
	dHat = NULL;
	cout << "freeing the memory of the forecaster \n"; /*TODO Remove prints */
}
