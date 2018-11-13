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
		if(abs(hostArrayA[i]) > 1e2){
			TOLERANCE = 1e-1;
			variable = variable/hostArrayA[i] *100;
		}
			//TOLERANCE = 10;
		if(abs(variable) > TOLERANCE)
			cout<< i << " " << variable/hostArrayA[i] *100 << " " << a[i].GetFloat() << " " << variable << " ";
		_ASSERT(abs(variable) < TOLERANCE);
	}
	delete [] hostArrayA;
	return 1;
}

/**
 * Function to compare the deviceArray with the input from the json file
 */
template<typename T>
uint_t TestSmpcController::compareDeviceArray(T* deviceArrayA, T* hostArrayB, uint_t dim){
	T variable;
	real_t TOLERANCE = 1e-2;
	T* hostArrayA = new T[dim];
	_CUDA( cudaMemcpy(hostArrayA, deviceArrayA, dim*sizeof(T), cudaMemcpyDeviceToHost) );
	for (uint_t i = 0; i < dim; i++){
		variable = hostArrayA[i] - hostArrayB[i];
		if(abs(hostArrayA[i]) > 1e2){
			TOLERANCE = 1e-1;
			variable = variable/hostArrayA[i] *100;
		}
		if(abs(variable) > TOLERANCE)
			cout<< i << " " << variable/hostArrayA[i] *100 << " " << hostArrayB[i] << " " << variable << " ";
		_ASSERT(abs(variable) < TOLERANCE);
	}
	delete [] hostArrayA;
	return 1;
}


/**
 * Function to set a deviceArray with the input from the json file
 */
template<typename T>
void TestSmpcController::setDeviceArray(T* deviceArray, uint_t dim){
	T *hostArray = new T[dim];
	for (uint_t i = 0; i < dim; i++){
		hostArray[i] = (T) a[i].GetFloat();
	}
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
	pathToFileGlobalFbeSmpc = "../test/testDataFiles/smpcFbeTest.json";
}

/**
 * Constructor that create a TestSmpcController object which is derived from the SmpcController object
 *
 * @param pathToConfigFile   path to the smpc configuration file
 */
TestSmpcController::TestSmpcController( string pathToConfigFile ): SmpcController( pathToConfigFile ){
	pathToFileSmpc = "../test/testDataFiles/smpcTest.json";
	pathToFileGlobalFbeSmpc = "../test/testDataFiles/smpcFbeTest.json";
}
/**
 * Function to test the dualExtrapolation function
 */
uint_t TestSmpcController::testExtrapolation(){
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
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

/**
 * Function to test the solve function
 */
uint_t TestSmpcController::testSoveStep(){
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nd = ptrDwnNetwork->getNumDemands();
	uint_t nv = ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	// implement the factor step, elimination of the control-disturbance constraints and updating the
	// nominal forecasts and prices
	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	/*uint_t timeInst = 1;
	ptrMyForecaster->predictDemand( timeInst );
	ptrMyForecaster->predictPrices( timeInst );

	this->ptrMyEngine->factorStep();
	this->ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	this->ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());*/

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
	currentX = NULL;
	prevU = NULL;
	prevDemand = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

/**
 * Function to test the proximal function
 */
uint_t TestSmpcController::testProximalStep(){
	const char* fileName = pathToFileSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
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
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
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
		this->computeFixedPointResidual();
		this->dualUpdate();
		a = jsonDocument[VARNAME_TEST_FINAL_UPDATE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUpdateXi ) );
		a = jsonDocument[VARNAME_TEST_FINAL_UPDATE_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUpdatePsi ) );
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecResidual ) );
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(&devVecResidual[2*nx*nodes] ) );
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
 * Function to test the fixed point residual
 */
uint_t TestSmpcController::testFbeFixedPointResidual(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

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

		this->computeFixedPointResidual();

		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecResidual ) );
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(&devVecResidual[2*nx*nodes]) );
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
 * function to test the FBE hessian oracal
 */
uint_t TestSmpcController::testHessianOracalGlobalFbe(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbeXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbePsi, nu*nodes);

		this->computeHessianOracalGlobalFbe();

		a = jsonDocument[VARNAME_TEST_U_DIR];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecUdir ) );
		a = jsonDocument[VARNAME_TEST_X_DIR];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecXdir ) );
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
 * function to test the gradient of the FBE
 */
uint_t TestSmpcController::testFbeGradient(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecResidual, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(&devVecResidual[2*nx*nodes], nu*nodes);

		computeGradientFbe();

		a = jsonDocument[VARNAME_TEST_GRAD_FBE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecGradientFbeXi ) );
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecGradientFbePsi ) );
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
 * function to test the lbfgs direction
 */
uint_t TestSmpcController::testLbfgsDirection(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

		uint_t bufferSize = ptrMySmpcConfig->getLbfgsBufferSize();
		real_t TOLERANCE = 1e-1;
		real_t variable;

		a = jsonDocument[VARNAME_TEST_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrevXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrevPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbeXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbePsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_PREV_GRAD_FBE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrevGradientFbeXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PREV_GRAD_FBE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrevGradientFbePsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_LBFGS_MAT_S];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devLbfgsBufferMatS, (2*nx + nu)*nodes*bufferSize);
		a = jsonDocument[VARNAME_TEST_LBFGS_MAT_Y];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devLbfgsBufferMatY, (2*nx + nu)*nodes*bufferSize);
		a = jsonDocument[VARNAME_TEST_LBFGS_INV_RHO];
		_ASSERT(a.IsArray());
		for(uint_t i = 0; i< bufferSize;i++ )
			if(a[i].GetFloat() != 0)
				lbfgsBufferRho[i] = 1/a[i].GetFloat();
			else
				lbfgsBufferRho[i] = 0;
		a = jsonDocument[VARNAME_TEST_LBFGS_H];
		_ASSERT(a.IsArray());
		lbfgsBufferHessian = a[0].GetFloat();
		a = jsonDocument[VARNAME_TEST_LBFGS_COL];
		_ASSERT(a.IsArray());
		lbfgsBufferCol = (uint_t) a[0].GetFloat();
		a = jsonDocument[VARNAME_TEST_LBFGS_MEM];
		_ASSERT(a.IsArray());
		lbfgsBufferMemory = (uint_t) a[0].GetFloat();

		computeLbfgsDirection();

		a = jsonDocument[VARNAME_TEST_UPDATE_LBFGS_H];
		_ASSERT(a.IsArray());
		variable = lbfgsBufferHessian - a[0].GetFloat();
		_ASSERT(abs(variable) < TOLERANCE);
		a = jsonDocument[VARNAME_TEST_UPDATE_LBFGS_COL];
		_ASSERT(a.IsArray());
		variable = lbfgsBufferCol - (uint_t) a[0].GetFloat();
		_ASSERT(abs(variable) < TOLERANCE);
		a = jsonDocument[VARNAME_TEST_UPDATE_LBFGS_INV_RHO];
		_ASSERT(a.IsArray());
		for (uint_t i = 0; i < a.Size(); i++){
			if(a[i].GetFloat() != 0)
				variable = lbfgsBufferRho[i] - (real_t) 1/a[i].GetFloat();
			else
				variable = lbfgsBufferRho[i] - (real_t) a[i].GetFloat();
			if(abs(variable) > TOLERANCE)
				cout<< i << " " << lbfgsBufferRho[i] << " " << a[i].GetFloat()<< endl;
			_ASSERT(abs(variable) < TOLERANCE);
		}
		a = jsonDocument[VARNAME_TEST_UPDATE_LBFGS_MAT_S];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devLbfgsBufferMatS ) );
		a = jsonDocument[VARNAME_TEST_UPDATE_LBFGS_MAT_Y];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devLbfgsBufferMatY ) );
		a = jsonDocument[VARNAME_TEST_LBFGS_DIR_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecLbfgsDirXi ) );
		a = jsonDocument[VARNAME_TEST_LBFGS_DIR_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecLbfgsDirPsi ) );

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
 * function to test the value FBE
 */
uint_t TestSmpcController::testValueFbe(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	real_t costFbeDualY, variable;
	real_t TOLERANCE = 1e-1;

	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecResidual, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(&devVecResidual[2*nx*nodes], nu*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_U];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecU, nu*nodes);

		costFbeDualY = this->computeValueFbe();

		a = jsonDocument[VARNAME_TEST_FBE_COST];
		_ASSERT(a.IsArray());
		variable = costFbeDualY - a[0].GetFloat();
		variable = variable/a[0].GetFloat()*100;
		//cout<< variable << " " << costFbeDualY << " " << a[0].GetFloat() << endl;
		_ASSERT(abs(variable) < TOLERANCE);
		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

uint_t TestSmpcController::testFbeLineSearch(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nv = this->ptrMySmpcConfig->getNV();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();
	real_t costFbeDualY, variable;
	real_t TOLERANCE = 1e-1;
	real_t linesearchTau;

	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecResidual, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMAL_INFS_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(&devVecResidual[2*nx*nodes], nu*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_X];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecX, nx*nodes);
		a = jsonDocument[VARNAME_TEST_U];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecU, nu*nodes);
		a = jsonDocument[VARNAME_TEST_LBFGS_DIR_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecLbfgsDirXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_LBFGS_DIR_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecLbfgsDirPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbeXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbePsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMALX];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrimalXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_PRIMALU];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPrimalPsi, nu*nodes);

		costFbeDualY = this->computeValueFbe();
		linesearchTau = computeLineSearchLbfgsUpdate( costFbeDualY );

		a = jsonDocument[VARNAME_TEST_FBE_COST];
		_ASSERT(a.IsArray());
		variable = costFbeDualY - a[0].GetFloat();
		variable = variable/a[0].GetFloat()*100;
		//cout<< variable << " " << costFbeDualY << " " << a[0].GetFloat() << endl;
		_ASSERT(abs(variable) < TOLERANCE);

		a = jsonDocument[VARNAME_TEST_UPDATE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecAcceleratedXi ) );
		a = jsonDocument[VARNAME_TEST_UPDATE_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecAcceleratedPsi ) );
		a = jsonDocument[VARNAME_TEST_TAU];
		_ASSERT(a.IsArray());
		variable = linesearchTau - a[0].GetFloat();
		_ASSERT(abs(variable) < TOLERANCE);
		a = jsonDocument[VARNAME_TEST_UPDATE_RESIDUAL_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecResidual ) );
		a = jsonDocument[VARNAME_TEST_UPDATE_RESIDUAL_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(&devVecResidual[2*nx*nodes] ) );

		delete [] readBuffer;
		readBuffer = NULL;
	}
	fclose(infile);
	infile = NULL;
	ptrDwnNetwork = NULL;
	ptrMyScenarioTree = NULL;
	return 1;
}

uint_t TestSmpcController::testFbeDualUpdate(){
	const char* fileName = pathToFileGlobalFbeSmpc.c_str();
	ScenarioTree *ptrMyScenarioTree = this->getScenarioTree();
	DwnNetwork *ptrDwnNetwork = this->getDwnNetwork();
	uint_t nx = ptrDwnNetwork->getNumTanks();
	uint_t nu = ptrDwnNetwork->getNumControls();
	uint_t nodes = ptrMyScenarioTree->getNumNodes();

	real_t *currentX = ptrMySmpcConfig->getCurrentX();
	real_t *prevU = ptrMySmpcConfig->getPrevU();
	real_t *prevDemand = ptrMySmpcConfig->getPrevDemand();
	real_t *hostPrevXi = new real_t[2*nx*nodes];
	real_t *hostPrevPsi = new real_t[nu*nodes];
	real_t *hostPrevGradFbeXi = new real_t[2*nx*nodes];
	real_t *hostPrevGradFbePsi = new real_t[nu*nodes];

	ptrMyEngine->factorStep();
	ptrMyEngine->updateStateControl(currentX, prevU, prevDemand);
	ptrMyEngine->eliminateInputDistubanceCoupling( ptrMyForecaster->getNominalDemand(),
			ptrMyForecaster->getNominalPrices());

	rapidjson::Document jsonDocument;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFileGlobalFbeSmpc << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream smpcJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(smpcJsonStream);

		a = jsonDocument[VARNAME_TEST_ACCELE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecXi, 2*nx*nodes);
		_CUDA( cudaMemcpy( hostPrevXi, devVecXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToHost) );
		a = jsonDocument[VARNAME_TEST_ACCELE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecPsi, nu*nodes);
		_CUDA( cudaMemcpy( hostPrevPsi, devVecPsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToHost) );
		a = jsonDocument[VARNAME_TEST_UPDATE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedXi, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_UPDATE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecAcceleratedPsi, nu*nodes);
		a = jsonDocument[VARNAME_TEST_UPDATE_RESIDUAL_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecResidual, 2*nx*nodes);
		a = jsonDocument[VARNAME_TEST_UPDATE_RESIDUAL_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(&devVecResidual[2*nx*nodes], nu*nodes);
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_XI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbeXi, 2*nx*nodes);
		_CUDA( cudaMemcpy( hostPrevGradFbeXi, devVecGradientFbeXi, 2*nx*nodes*sizeof(real_t), cudaMemcpyDeviceToHost) );
		a = jsonDocument[VARNAME_TEST_GRAD_FBE_PSI];
		_ASSERT(a.IsArray());
		setDeviceArray<real_t>(devVecGradientFbePsi, nu*nodes);
		_CUDA( cudaMemcpy( hostPrevGradFbePsi, devVecGradientFbePsi, nu*nodes*sizeof(real_t), cudaMemcpyDeviceToHost) );

		dualUpdate();

		a = jsonDocument[VARNAME_TEST_FINAL_UPDATE_XI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecXi ) );
		a = jsonDocument[VARNAME_TEST_FINAL_UPDATE_PSI];
		_ASSERT(a.IsArray());
		_ASSERT( compareDeviceArray<real_t>(devVecPsi ) );

		_ASSERT( compareDeviceArray<real_t>(devVecPrevGradientFbeXi, hostPrevGradFbeXi, 2*nx*nodes) );
		_ASSERT( compareDeviceArray<real_t>(devVecPrevGradientFbePsi, hostPrevGradFbePsi, nu*nodes) );

		_ASSERT( compareDeviceArray<real_t>(devVecPrevXi, hostPrevXi, 2*nx*nodes) );
		_ASSERT( compareDeviceArray<real_t>(devVecPrevPsi, hostPrevPsi, nu*nodes) );

		delete [] readBuffer;
		delete [] hostPrevXi;
		delete [] hostPrevPsi;
		delete [] hostPrevGradFbeXi;
		delete [] hostPrevGradFbePsi;
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
