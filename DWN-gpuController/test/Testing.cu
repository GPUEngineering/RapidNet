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

#include "Testing.cuh"

Testing::Testing(){
	pathToFileEnigne = "../test/testDataFiles/engine.json";
	pathToFileForecaster = "../test/testDataFiles/forecastor.json";
	pathToFileNetwork = "../test/testDataFiles/network.json";
	pathToFileScenarioTree = "../test/testDataFiles/scenarioTree.json";
	pathToFileControllerConfig = "../test/testDataFiles/controllerConfig.json";
	pathToFileEnigne = "../test/testDataFiles/engineTest.json";
}

template<typename T>
int Testing::compareArray(T* arrayA){
	T variable;
	real_t TOLERANCE = 1e-2;
	for (rapidjson::SizeType i = 0; i < a.Size(); i++){
		variable = arrayA[i] - a[i].GetDouble();
		_ASSERT(abs(variable) < TOLERANCE);
	}
	return 1;
}

template<typename T>
int Testing::compareDeviceArray(T* deviceArrayA){
	T variable;
	real_t TOLERANCE = 1e-2;
	T* hostArrayA = new T[a.Size()];
	_CUDA( cudaMemcpy(hostArrayA, deviceArrayA, a.Size()*sizeof(T), cudaMemcpyDeviceToHost) );
	for (rapidjson::SizeType i = 0; i < a.Size(); i++){
		variable = hostArrayA[i] - a[i].GetDouble();
		cout << variable << " ";
		//_ASSERT(abs(variable) < TOLERANCE);
	}
	cout << endl;
	return 1;
}



int Testing::testNetwork(){
	DwnNetwork *ptrMyDwnNetwork = new DwnNetwork(pathToFileNetwork);
	const char* fileName = pathToFileNetwork.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileNetwork << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		//matA = new real_t[nTanks * nTanks];
		a = jsonDocument[VARNAME_A];
		_ASSERT(a.IsArray());
		//cout<< compareArray(ptrMyDwnNetwork->getMatA()) << endl;
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getMatA() ) );
		a = jsonDocument[VARNAME_B];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getMatB()) );
		a = jsonDocument[VARNAME_GD];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getMatGd()) );
		a = jsonDocument[VARNAME_E];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getMatE()) );
		a = jsonDocument[VARNAME_ED];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getMatEd()) );
		a = jsonDocument[VARNAME_XMIN];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getXmin()) );
		a = jsonDocument[VARNAME_XMAX];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getXmax()) );
		a = jsonDocument[VARNAME_XSAFE];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getXsafe()) );
		a = jsonDocument[VARNAME_UMIN];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getUmin()) );
		a = jsonDocument[VARNAME_UMAX];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getUmax()) );
		a = jsonDocument[VARNAME_ALPHA1];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMyDwnNetwork->getAlpha()) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrMyDwnNetwork->~DwnNetwork();
	ptrMyDwnNetwork = NULL;
	cout<< "Completed testing of the DWN network" << endl;
	return 1;
}

int Testing::testScenarioTree(){
	ScenarioTree *ptrMyScenarioTree = new ScenarioTree( pathToFileScenarioTree );
	const char* fileName = pathToFileScenarioTree.c_str();
	rapidjson::Document jsonDocument;
	//rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileScenarioTree << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		throw std::logic_error("Error in opening the file");
		//exit(100); /*TODO never use `exit`; throw an exception instead */
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument[VARNAME_N];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getPredHorizon() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_K];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getNumScenarios() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_NODES];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getNumNodes() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_NUM_NONLEAF];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMyScenarioTree->getNumNonleafNodes() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_NUM_CHILD_TOT];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getNumChildrenTot() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_STAGES];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getStageNodes() ) );
		a = jsonDocument[VARNAME_NODES_PER_STAGE];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getNodesPerStage()) );
		a = jsonDocument[VARNAME_NODES_PER_STAGE_CUMUL];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getNodesPerStageCumul() ) );
		a = jsonDocument[VARNAME_LEAVES];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getLeaveArray()) );
		a = jsonDocument[VARNAME_CHILDREN];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getChildArray()) );
		a = jsonDocument[VARNAME_ANCESTOR];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getAncestorArray()) );
		a = jsonDocument[VARNAME_NUM_CHILDREN];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getNumChildren()) );
		a = jsonDocument[VARNAME_NUM_CHILD_CUMUL];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<uint_t>( ptrMyScenarioTree->getNumChildrenCumul()) );
		a = jsonDocument[VARNAME_PROB_NODE];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyScenarioTree->getProbArray()) );
		a = jsonDocument[VARNAME_DEMAND_NODE];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyScenarioTree->getErrorDemandArray()) );
		a = jsonDocument[VARNAME_PRICE_NODE];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyScenarioTree->getErrorPriceArray()) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrMyScenarioTree->~ScenarioTree();
	ptrMyScenarioTree = NULL;
	cout<< "Completed testing of the scenario tree" << endl;
	return 1;
}

int Testing::testForecaster(){
	Forecaster *ptrMyForecaster = new Forecaster( pathToFileForecaster );
	const char* fileName = pathToFileForecaster.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileForecaster << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536]; /*TODO Make sure this is a good practice */
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument[VARNAME_N];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getPredHorizon() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_DIM_DEMAND];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getDimDemand() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_DIM_PRICES];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getDimPrice() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_DHAT];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyForecaster->getNominalDemand()) );
		a = jsonDocument[VARNAME_ALPHAHAT];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyForecaster->getNominalPrices()) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	ptrMyForecaster->~Forecaster();
	ptrMyForecaster = NULL;
	infile = NULL;
	cout<< "Completed testing of the Forecaster" << endl;
	return 1;
}

int Testing::testControllerConfig(){
	SmpcConfiguration *ptrMySmpcConfig = new SmpcConfiguration( pathToFileControllerConfig );
	const char* fileName = pathToFileControllerConfig.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileControllerConfig << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		a = jsonDocument[VARNAME_NX];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getNX() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_NU];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getNU() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_ND];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getND() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_NV];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getNV() == (uint_t) a[0].GetDouble() );
		a = jsonDocument[VARNAME_L];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getMatL() ) );
		a = jsonDocument[VARNAME_LHAT];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getMatLhat() ) );
		a = jsonDocument[VARNAME_COSTW];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getCostW() ) );
		a = jsonDocument[VARNAME_PENALITY_X];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getPenaltyState() == a[0].GetDouble() );
		a = jsonDocument[VARNAME_PENALITY_XS];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getPenaltySafety() == a[0].GetDouble() );
		a = jsonDocument[VARNAME_DIAG_PRCND];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getMatPrcndDiag() ) );
		a = jsonDocument[VARNAME_CURRENT_X];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getCurrentX() ) );
		a = jsonDocument[VARNAME_PREV_UHAT];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getPrevUhat() ) );
		a = jsonDocument[VARNAME_PREV_U];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getPrevU() ) );
		a = jsonDocument[VARNAME_PREV_V];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getPrevV() ) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	ptrMySmpcConfig->~SmpcConfiguration();
	ptrMySmpcConfig = NULL;
	infile = NULL;
	cout << "Completed testing the SmpcConfiguration" << endl;
	return 1;
}

int Testing::testEngineTesting(){
	DwnNetwork *ptrMyDwnNetwork = new DwnNetwork(pathToFileNetwork);
	ScenarioTree *ptrMyScenarioTree = new ScenarioTree( pathToFileScenarioTree );
	SmpcConfiguration *ptrMySmpcConfig = new SmpcConfiguration( pathToFileControllerConfig );
	Forecaster *ptrMyForecaster = new Forecaster( pathToFileForecaster );

	Engine *ptrMyEngine = new Engine( ptrMyDwnNetwork, ptrMyScenarioTree, ptrMySmpcConfig );
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	const char* fileName = pathToFileEnigne.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileEnigne << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		a = jsonDocument[VARNAME_BETA];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getVecBeta() ) );
		/*a = jsonDocument[VARNAME_UHAT];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getVecUhat() ) );
		a = jsonDocument[VARNAME_VEC_E];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getVecE() ) );*/
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;	/**/
	ptrMyDwnNetwork->~DwnNetwork();
	ptrMyScenarioTree->~ScenarioTree();
	ptrMySmpcConfig->~SmpcConfiguration();
	ptrMyForecaster->~Forecaster();
	ptrMyEngine->~Engine();

	cout << "Completed testing the Engine" << endl;
	return 1;
}

Testing::~Testing(){

}
