/*
 * forecast.cu
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


#include "Forecaster.cuh"

Forecaster::Forecaster(string pathToFile){
	cout << "allocating memory for the forecaster \n";
	const char* fileName = pathToFile.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument["N"];
		assert( a.IsArray() );
		N = (uint_t) a[0].GetDouble();
		a = jsonDocument["K"];
		assert( a.IsArray() );
		K = (uint_t) a[0].GetDouble();
		a = jsonDocument["nodes"];
		assert( a.IsArray() );
		N_NODES = (uint_t) a[0].GetDouble();
		a = jsonDocument["nNonLeafNodes"];
		assert(a.IsArray());
		N_NONLEAF_NODES = (uint_t) a[0].GetDouble();
		a = jsonDocument["nChildrenTot"];
		assert( a.IsArray() );
		N_CHILDREN_TOT = (uint_t) a[0].GetDouble();
		a = jsonDocument["dimNode"];
		assert( a.IsArray() );
		DIM_NODE = (uint_t) a[0].GetDouble();
		//rapidjson::Value& a = jsonDocument["matA"];
		stages = new uint_t[N_NODES];
		a = jsonDocument["stages"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			stages[i] = a[i].GetDouble();
		nodesPerStage = new uint_t[N];
		a = jsonDocument["nodesPerStage"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nodesPerStage[i] = a[i].GetDouble();
		nodesPerStageCumul = new uint_t[N+1];
		a = jsonDocument["nodesPerStageCumul"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nodesPerStageCumul[i] = a[i].GetDouble();
		leaves = new uint_t[K];
		a = jsonDocument["leaves"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			leaves[i] = a[i].GetDouble();
		children = new uint_t[N_CHILDREN_TOT];
		a = jsonDocument["children"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			children[i] = a[i].GetDouble();
		ancestor = new uint_t[N_NODES];
		a = jsonDocument["ancestor"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			ancestor[i] = a[i].GetDouble();
		nChildren = new uint_t[N_NONLEAF_NODES];
		a = jsonDocument["nChildren"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nChildren[i] = a[i].GetDouble();
		nChildrenCumul = new uint_t[N_NODES];
		a = jsonDocument["nChildrenCumul"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			nChildrenCumul[i] = a[i].GetDouble();
		probNode = new real_t[N_NODES];
		a = jsonDocument["probNode"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			probNode[i] = a[i].GetDouble();
		valueNode = new real_t[N_NODES * DIM_NODE];
		a = jsonDocument["valueNode"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			valueNode[i] = a[i].GetDouble();
		dHat = new real_t[DIM_NODE * N];
		a = jsonDocument["dHat"];
		assert( a.IsArray() );
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			dHat[i] = a[i].GetDouble();
		delete [] readBuffer;
	}
	fclose(infile);
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
	cout << "freeing the memory of the forecaster \n";
}
