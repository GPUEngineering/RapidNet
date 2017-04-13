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
uint_t Testing::compareArray(T* arrayA){
	T variable;
	real_t TOLERANCE = 1e-2;
	for (rapidjson::SizeType i = 0; i < a.Size(); i++){
		variable = arrayA[i] - (T) a[i].GetFloat();
		_ASSERT(abs(variable) < TOLERANCE);
	}
	return 1;
}

template<typename T>
uint_t Testing::compareDeviceArray(T* deviceArrayA){
	T variable;
	real_t TOLERANCE = 1e-2;
	T* hostArrayA = new T[a.Size()];
	_CUDA( cudaMemcpy(hostArrayA, deviceArrayA, a.Size()*sizeof(T), cudaMemcpyDeviceToHost) );
	for (uint_t i = 0; i < a.Size(); i++){
		variable = hostArrayA[i] - (T) a[i].GetFloat();
		//cout << variable << " ";
		_ASSERT(abs(variable) < TOLERANCE);
	}
	//cout << endl;
	return 1;
}

template<typename T>
uint_t Testing::compareDeviceScenarioArray(T* arrayA, uint_t *nodes, uint_t dim){
	T variable;
	real_t TOLERANCE = 1e-2;
	T* hostArrayA = new T[dim];
	uint_t numNodes = a.Size()/dim;
	for (uint_t i = 0; i < numNodes; i++){
		//cout << (nodes[i]-1)*dim << " " << numNodes << " " << dim << endl;
		_CUDA( cudaMemcpy(hostArrayA, &arrayA[(nodes[i]-1)*dim], dim*sizeof(T), cudaMemcpyDeviceToHost));
		for (uint_t j = 0; j < dim; j++ ){
			variable = hostArrayA[j] - a[i*dim + j].GetFloat();
			if (abs(variable) > TOLERANCE)
				cout << i*dim + j << " " << hostArrayA[j] << " " << a[i*dim + j].GetFloat() << " ";
			_ASSERT(abs(variable) < TOLERANCE);
		}
	}
	return 1;
}

uint_t Testing::testNetwork(){
	const char* fileName = pathToFileNetwork.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileNetwork << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		DwnNetwork *ptrMyDwnNetwork = new DwnNetwork(pathToFileNetwork);
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		a = jsonDocument[VARNAME_A];
		_ASSERT(a.IsArray());
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
		fclose(infile);
		infile = NULL;
		delete ptrMyDwnNetwork;
		ptrMyDwnNetwork = NULL;
	}
	cout<< "Completed testing of the DWN network" << endl;
	return 1;
}

uint_t Testing::testScenarioTree(){
	const char* fileName = pathToFileScenarioTree.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		try{
			throw std::logic_error("Error in opening the file ");
		}
		catch (exception &e){
			//cout << e.what() << __LINE__ << endl;
		}
		return 0;
		//exit(100); /*TODO never use `exit`; throw an exception instead */
	}else{
		ScenarioTree *ptrMyScenarioTree = new ScenarioTree( pathToFileScenarioTree );
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument[VARNAME_N];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getPredHorizon() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_K];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getNumScenarios() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_NODES];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getNumNodes() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_NUM_NONLEAF];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMyScenarioTree->getNumNonleafNodes() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_NUM_CHILD_TOT];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyScenarioTree->getNumChildrenTot() == (uint_t) a[0].GetFloat() );
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
		delete ptrMyScenarioTree;
		ptrMyScenarioTree = NULL;
		fclose(infile);
		infile = NULL;
		cout<< "Completed testing of the scenario tree" << endl;
		return 1;
	}
}

uint_t Testing::testForecaster(){
	const char* fileName = pathToFileForecaster.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileForecaster << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << fileName << endl;
		exit(100);
	}else{
		Forecaster *ptrMyForecaster = new Forecaster( pathToFileForecaster );
		char* readBuffer = new char[65536]; /*TODO Make sure this is a good practice */
		rapidjson::FileReadStream networkJsonStream( infile, readBuffer, sizeof(readBuffer) );
		jsonDocument.ParseStream( networkJsonStream );
		a = jsonDocument[VARNAME_N];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getPredHorizon() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_SIM_HORIZON];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getSimHorizon() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_DIM_DEMAND];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getDimDemand() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_DIM_PRICES];
		_ASSERT( a.IsArray() );
		_ASSERT( ptrMyForecaster->getDimPrice() == (uint_t) a[0].GetFloat() );
		uint_t timeInst = 1;
		ptrMyForecaster->predictDemand( timeInst );
		a = jsonDocument[VARNAME_DEMAND_SIM];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyForecaster->getNominalDemand()) );
		/*for(uint_t iCount = 0; iCount < a.Size(); iCount++){
			cout << ptrMyForecaster->getNominalDemand()[iCount] << " "<< a[iCount].GetFloat()<< " ";
		}
		cout << endl;*/
		ptrMyForecaster->predictPrices( timeInst );
		a = jsonDocument[VARNAME_PRICE_SIM];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyForecaster->getNominalPrices()) );
		/*for(uint_t iCount = 0; iCount < a.Size(); iCount++){
			cout << ptrMyForecaster->getNominalPrices()[iCount] << " "<< a[iCount].GetFloat()<< " ";
		}
		cout << endl;
		/*a = jsonDocument[VARNAME_DHAT];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyForecaster->getNominalDemand()) );
		a = jsonDocument[VARNAME_ALPHAHAT];
		_ASSERT( a.IsArray() );
		_ASSERT( compareArray<real_t>( ptrMyForecaster->getNominalPrices()) );*/
		delete [] readBuffer;
		readBuffer = NULL;
		fclose(infile);
		delete ptrMyForecaster;
		ptrMyForecaster = NULL;
		infile = NULL;
		cout<< "Completed testing of the Forecaster" << endl;
	}
	return 1;
}

uint_t Testing::testControllerConfig(){
	const char* fileName = pathToFileControllerConfig.c_str();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileControllerConfig << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		SmpcConfiguration *ptrMySmpcConfig = new SmpcConfiguration( pathToFileControllerConfig );
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		a = jsonDocument[VARNAME_NX];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getNX() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_NU];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getNU() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_ND];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getND() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_NV];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getNV() == (uint_t) a[0].GetFloat() );
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
		_ASSERT( ptrMySmpcConfig->getPenaltyState() == (real_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_PENALITY_XS];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getPenaltySafety() == (real_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_DIAG_PRCND];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getMatPrcndDiag() ) );
		a = jsonDocument[VARNAME_CURRENT_X];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getCurrentX() ) );
		a = jsonDocument[VARNAME_PREV_DEMAND];
		_ASSERT(a.IsArray());
		_ASSERT( compareArray<real_t>( ptrMySmpcConfig->getPrevDemand() ) );
		a = jsonDocument[VARNAME_STEP_SIZE];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getStepSize() == (real_t) a[0].GetFloat() );
		a = jsonDocument[VARNAME_MAX_ITER];
		_ASSERT(a.IsArray());
		_ASSERT( ptrMySmpcConfig->getMaxIterations() == (uint_t) a[0].GetFloat() );
		a = jsonDocument[PATH_NETWORK_FILE];
		_ASSERT(a.IsString());
		_ASSERT( !ptrMySmpcConfig->getPathToNetwork().compare(a.GetString()));
		a = jsonDocument[PATH_SCENARIO_TREE_FILE];
		_ASSERT(a.IsString());
		_ASSERT( !ptrMySmpcConfig->getPathToScenarioTree().compare(a.GetString()));
		a = jsonDocument[PATH_FORECASTER_FILE];
		_ASSERT(a.IsString());
		_ASSERT( !ptrMySmpcConfig->getPathToForecaster().compare(a.GetString()));
		delete [] readBuffer;
		readBuffer = NULL;
		fclose(infile);
		delete ptrMySmpcConfig;
		ptrMySmpcConfig = NULL;
		infile = NULL;
		cout << "Completed testing the SmpcConfiguration" << endl;
	}
	return 1;
}

uint_t Testing::testEngineTesting(){
	//DwnNetwork *ptrMyDwnNetwork = new DwnNetwork(pathToFileNetwork);
	//ScenarioTree *ptrMyScenarioTree = new ScenarioTree( pathToFileScenarioTree );
	SmpcConfiguration *ptrMySmpcConfig = new SmpcConfiguration( pathToFileControllerConfig );
	Forecaster *ptrMyForecaster = new Forecaster( pathToFileForecaster );

	//Engine *ptrMyEngine = new Engine( ptrMyDwnNetwork, ptrMyScenarioTree, ptrMySmpcConfig );
	Engine *ptrMyEngine = new Engine( ptrMySmpcConfig );
	DwnNetwork *ptrMyDwnNetwork = ptrMyEngine->getDwnNetwork();
	ScenarioTree *ptrMyScenarioTree = ptrMyEngine->getScenarioTree();

	uint_t *testNodeArray = new uint_t[ptrMyScenarioTree->getPredHorizon()];
	uint_t dim;
	uint_t nx  = ptrMyDwnNetwork->getNumTanks();
	uint_t nu  = ptrMyDwnNetwork->getNumControls();
	uint_t nd  = ptrMyDwnNetwork->getNumDemands();
	uint_t nv  = ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	real_t *y = new real_t[ptrMyScenarioTree->getNumNodes()*nu*nu];
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
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
		a = jsonDocument[VARNAME_UHAT];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getVecUhat() ) );
		a = jsonDocument[VARNAME_VEC_E];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getVecE() ) );
		a = jsonDocument[VARNAME_BETA];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getVecBeta() ) );
		a = jsonDocument[VARNAME_SCE_NOD];
		_ASSERT(a.IsArray());
		for (uint_t i = 0; i < ptrMyScenarioTree->getPredHorizon(); i++){
			testNodeArray[i] = (uint_t) a[i].GetFloat();
		}
		a = jsonDocument[VARNAME_L];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>( ptrMyEngine->getSysMatL() ) );
		a = jsonDocument[VARNAME_SYS_F];
		_ASSERT(a.IsArray());
		dim = 2*nx*nx;
		_ASSERT( compareDeviceScenarioArray<real_t>( ptrMyEngine->getSysMatF(), testNodeArray, dim));
		a = jsonDocument[VARNAME_SYS_G];
		_ASSERT(a.IsArray());
		dim = nu*nu;
		_ASSERT( compareDeviceScenarioArray<real_t>( ptrMyEngine->getSysMatG(), testNodeArray, dim));
		a = jsonDocument[VARNAME_TEST_XMIN];
		_ASSERT(a.IsArray());
		dim = nx;
		_ASSERT( compareDeviceScenarioArray<real_t>( ptrMyEngine->getSysXmin(), testNodeArray, dim));
		a = jsonDocument[VARNAME_TEST_XMAX];
		_ASSERT(a.IsArray());
		dim = nx;
		_ASSERT( compareDeviceScenarioArray<real_t>( ptrMyEngine->getSysXmax(), testNodeArray, dim));
		a = jsonDocument[VARNAME_TEST_XS];
		_ASSERT(a.IsArray());
		dim = nx;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getSysXs(), testNodeArray, dim));
		a = jsonDocument[VARNAME_TEST_UMIN];
		_ASSERT(a.IsArray());
		dim = nu;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getSysUmin(), testNodeArray, dim));
		a = jsonDocument[VARNAME_TEST_UMAX];
		_ASSERT(a.IsArray());
		dim = nu;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getSysUmax(), testNodeArray, dim));
		a = jsonDocument[VARNAME_OMEGA];
		_ASSERT(a.IsArray());
		dim = nv*nv;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatOmega(), testNodeArray, dim));
		a = jsonDocument[VARNAME_G];
		_ASSERT(a.IsArray());
		dim = nx*nv;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatG(), testNodeArray, dim));
		a = jsonDocument[VARNAME_D];
		_ASSERT(a.IsArray());
		dim = 2*nx*nv;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatD(), testNodeArray, dim));
		a = jsonDocument[VARNAME_THETA];
		_ASSERT(a.IsArray());
		dim = nx*nv;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatTheta(), testNodeArray, dim));
		a = jsonDocument[VARNAME_F];
		_ASSERT(a.IsArray());
		dim = nu*nv;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatF(), testNodeArray, dim));
		a = jsonDocument[VARNAME_PSI];
		_ASSERT(a.IsArray());
		dim = nv*nu;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatPsi(), testNodeArray, dim));
		a = jsonDocument[VARNAME_PHI];
		_ASSERT(a.IsArray());
		dim = 2*nv*nx;
		_ASSERT( compareDeviceScenarioArray( ptrMyEngine->getMatPhi(), testNodeArray, dim));
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	delete ptrMyDwnNetwork;
	delete ptrMyScenarioTree;
	delete ptrMySmpcConfig;
	delete ptrMyForecaster;
	delete ptrMyEngine;
	ptrMyDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	ptrMyScenarioTree = NULL;
	ptrMySmpcConfig = NULL;

	cout << "Completed testing the Engine" << endl;
	return 1;
}

uint_t Testing::testSmpcController(){
	//DwnNetwork *ptrMyDwnNetwork = new DwnNetwork(pathToFileNetwork);
	//ScenarioTree *ptrMyScenarioTree = new ScenarioTree( pathToFileScenarioTree );
	//SmpcConfiguration *ptrMySmpcConfig = new SmpcConfiguration( pathToFileControllerConfig );
	//Forecaster *ptrMyForecaster = new Forecaster( pathToFileForecaster );
	//Engine *ptrMyEngine = new Engine( ptrMyDwnNetwork, ptrMyScenarioTree, ptrMySmpcConfig );
	//TestSmpcController *ptrMyTestSmpc = new TestSmpcController(ptrMyForecaster, ptrMyEngine, ptrMySmpcConfig);
	TestSmpcController *ptrMyTestSmpc = new TestSmpcController( pathToFileControllerConfig );
	DwnNetwork *ptrMyDwnNetwork = ptrMyTestSmpc->getDwnNetwork();
	ScenarioTree *ptrMyScenarioTree = ptrMyTestSmpc->getScenarioTree();
	SmpcConfiguration *ptrMySmpcConfig = ptrMyTestSmpc->getSmpcConfiguration();
	Forecaster *ptrMyForecaster = ptrMyTestSmpc->getForecaster();
	Engine *ptrMyEngine = ptrMyTestSmpc->getEngine();

	uint_t nx  = ptrMyDwnNetwork->getNumTanks();
	uint_t nu  = ptrMyDwnNetwork->getNumControls();
	uint_t nv  = ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	real_t *y = new real_t[ptrMyScenarioTree->getNumNodes()*nu*nu];
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());
	_ASSERT( ptrMyTestSmpc->testExtrapolation() );
	_ASSERT( ptrMyTestSmpc->testSoveStep() );
	_ASSERT( ptrMyTestSmpc->testProximalStep());
	_ASSERT( ptrMyTestSmpc->testDualUpdate() );

	cout << "completed testing of the controller" << endl;
	delete ptrMyDwnNetwork;
	delete ptrMyScenarioTree;
	delete ptrMySmpcConfig;
	delete ptrMyForecaster;
	delete ptrMyEngine;
	delete ptrMyTestSmpc;
	ptrMyDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;

	return 1;
}
Testing::~Testing(){

}
