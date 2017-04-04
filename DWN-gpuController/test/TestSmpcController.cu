/*
 * TestSmpcController.cu
 *
 *  Created on: Apr 3, 2017
 *      Author: control
 */


#include "TestSmpcController.cuh"

/**
 * Function to compare the deviceArray with the input from the json file
 */
template<typename T>
uint_t TestSmpcController::compareDeviceArray(T* deviceArrayA){
	T variable;
	real_t TOLERANCE = 1e-2;
	T* hostArrayA = new T[a.Size()];
	_CUDA( cudaMemcpy(hostArrayA, deviceArrayA, a.Size()*sizeof(T), cudaMemcpyDeviceToHost) );
	for (uint_t i = 0; i < a.Size(); i++){
		variable = hostArrayA[i] - a[i].GetDouble();
		_ASSERT(abs(variable) < TOLERANCE);
	}
	return 1;
}

/**
 * Function to set a deviceArray with the input from the json file
 */
template<typename T>
void TestSmpcController::setDeviceArray(T* deviceArray, uint_t dim){
	T *hostArray = new T[dim];
	for (int i = 0; i < dim; i++)
		hostArray[i] = (T) a[i].GetDouble();
	_CUDA( cudaMemcpy(deviceArray, hostArray, dim*sizeof(T), cudaMemcpyDeviceToHost));
	//_CUDA( cudaMemcpy(deviceArray, &a[0].GetDouble(), dim*sizeof(T), cudaMemcpyDeviceToHost));
}

/**
 * Constructor that create a TestSmpcController object which is derived from the SmpcController object
 *
 * @param  myForecaster  Forecaster object
 * @param  myEngine      Engine object
 * @param  mySmpcConfig  SmpcConfiguration object that contain the controller configuration
 */
TestSmpcController::TestSmpcController(Forecaster *myForecaster, Engine *myEngine,
		SmpcConfiguration *mySmpcConfig): SmpcController( myForecaster, myEngine, mySmpcConfig){
	PathToSmpcTestFile = "../test/testDataFiles/smpcTest.json";
}

/**
 * Function to test the dualExtrapolation function
 */
uint_t TestSmpcController::testExtrapolation(){
	const char* fileName = PathToFileSmpcTest.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
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
		a = jsonDocument[VARNAME_TEST_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPSi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_EXTAPOLATION];
		_ASSERT(a.IsArray());
		real_t lambda = a[1].GetDouble()*(1/a[0].GetDouble() - 1);
		this->dualExtrapolationStep(lambda);
		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecAcceleratedXi ) );
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecAcceleratedPsi ) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

/**
 * Function to test the solve function
 */
uint_t TestSmpcController::testSoveStep(){
	const char* fileName = PathToFileSmpcTest.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	// implement the factor step, elimination of the control-disturbance constaints and updating the
	// nominal forecasts and prices
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevUhat = ptrMySmpcConfig->getPrevUhat();
	this->ptrMyEngine->factorStep();
	this->ptrMyEngine->updateStateControl(currentX, prevU, prevUhat);
	this->ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());

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

		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedPsi, nu*nodes);
		this->solveStep();
		a = jsonDocument[VARNAME_TEST_X];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecX ) );
		a = jsonDocument[VARNAME_TEST_U];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecU ) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

/**
 * Function to test the proximal function
 */
uint_t TestSmpcController::testProximalStep(){
	const char* fileName = PathToFileSmpcTest.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

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

		a = jsonDocument[VARNAME_TEST_X];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecX, nx*nodes);
		a = jsonDocument[VARNAME_TEST_U];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecU, nu*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedPsi, nu*nodes);
		this->proximalFunG();
		a = jsonDocument[VARNAME_TEST_DUALX];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecDualXi ) );
		a = jsonDocument[VARNAME_TEST_DUALU];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecDualPsi ) );
		a = jsonDocument[VARNAME_TEST_PRIMALX];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecPrimalXi ) );
		a = jsonDocument[VARNAME_TEST_PRIMALU];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecPrimalPsi ) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

/**
 * Function to test the dual update function
 */
uint_t TestSmpcController::testDualUpdate(){
	const char* fileName = PathToFileSmpcTest.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

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

		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMALX];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrimalXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMALU];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrimalPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_DUALX];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecDualXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_DUALU];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecDualPsi, nu*nodes);
		this->dualUpdate();
		a = jsonDocument[VARNAME_TEST_DUAL_UPDATE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUpdateXi ) );
		a = jsonDocument[VARNAME_TEST_PRIMALU];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUpdatePsi ) );
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

TestSmpcController::~TestSmpcController(){

}
