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



#include "TestSmpcController.cuh"

/**
 * Function to compare the deviceArray with the input from the json file
 */
template<typename T>
uint_t TestSmpcController::compareDeviceArray(T* deviceArrayA){
	T variable;
	real_t TOLERANCE = 1e-1;
	T* hostArrayA = new T[a.Size()];
	_CUDA( cudaMemcpy(hostArrayA, deviceArrayA, a.Size()*sizeof(T), cudaMemcpyDeviceToHost) );
	for (uint_t i = 0; i < a.Size(); i++){
		variable = hostArrayA[i] - (T) a[i].GetFloat();
		if(abs(variable) > TOLERANCE)
			cout<< i << " " << hostArrayA[i] << " " << a[i].GetFloat()<< " ";
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
	for (uint_t i = 0; i < dim; i++)
		hostArray[i] = (T) a[i].GetFloat();
	_CUDA( cudaMemcpy(deviceArray, hostArray, dim*sizeof(T), cudaMemcpyHostToDevice));
	free(hostArray);
	//_CUDA( cudaMemcpy(deviceArray, &a[0].GetFloat(), dim*sizeof(T), cudaMemcpyDeviceToHost));
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
	pathToFileSmpc = "../test/testDataFiles/smpcTest.json";
}

/**
 * Function to test the dualExtrapolation function
 */
uint_t TestSmpcController::testExtrapolation(){
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileSmpc << infile << endl;
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
		setDeviceArray<real_t>(devVecPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_UPDATE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecUpdateXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_UPDATE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecUpdatePsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_EXTAPOLATION];
		_ASSERT(a.IsArray());
		real_t lambda = a[1].GetFloat()*(1/a[0].GetFloat() - 1);
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
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	// implement the factor step, elimination of the control-disturbance constraints and updating the
	// nominal forecasts and prices
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevUhat = ptrMySmpcConfig->getPrevUhat();
	real_t *prevV = ptrMySmpcConfig->getPrevV();
	this->ptrMyEngine->factorStep();
	this->ptrMyEngine->updateStateControl(currentX, prevU, prevUhat, prevV);
	this->ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileSmpc << infile << endl;
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
		/*a = jsonDocument[VARNAME_TEST_TEMP_V];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecV ) );*/
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
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileSmpc << infile << endl;
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
		/*real_t *y = new real_t[nu*nu*nodes];
		real_t **ptrY = new real_t*[nodes];
		cout << "Now it is the primal X" << endl ;
		for ( uint_t i = 0; i < nodes; i++){
			cout << i << " ";
			_CUDA( cudaMemcpy( y, &ptrMyEngine->getSysXsUpper()[i*nx], nx*sizeof(real_t), cudaMemcpyDeviceToHost));
			for( uint_t j = 0; j < nx; j++){
				cout <<  y[j] <<" ";
			}
			cout << endl;
		}*/
		a = jsonDocument[VARNAME_TEST_PRIMALX];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecPrimalXi ) );
		a = jsonDocument[VARNAME_TEST_PRIMALU];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecPrimalPsi ) );
		a = jsonDocument[VARNAME_TEST_DUALX];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecDualXi ) );
		a = jsonDocument[VARNAME_TEST_DUALU];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecDualPsi ) );/**/

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
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->ptrMyEngine->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->ptrMyEngine->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileSmpc << infile << endl;
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
		a = jsonDocument[VARNAME_TEST_FINAL_UPDATE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUpdateXi ) );
		a = jsonDocument[VARNAME_TEST_FINAL_UPDATE_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUpdatePsi ) );
		a = jsonDocument[VARNAME_TEST_FINAL_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecXi ) );
		a = jsonDocument[VARNAME_TEST_FINAL_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecPsi ) );
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
