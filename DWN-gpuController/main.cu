#include <cuda_device_runtime_api.h>
#include "cublas_v2.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"

#include "Configuration.h"
#include "SmpcController.cuh"
#include "test/Testing.cuh"

int main(void){
	uint_t TESTING = 1;
	startTicToc();
	tic();
	if (TESTING){
		Testing *myTesting = new Testing();
		_ASSERT( myTesting->testNetwork() );
		_ASSERT( myTesting->testScenarioTree() );
		_ASSERT( myTesting->testForecaster() );
		_ASSERT( myTesting->testControllerConfig() );
		_ASSERT( myTesting->testEngineTesting() );
		_ASSERT( myTesting->testSmpcController());
	}
	real_t time = toc();
	cout << "time lapsed " << time << "in milliseconds" << endl;
	/*
	string pathToNetworkFile = "../dataFiles/network.json";
	const char* fileName = pathToNetworkFile.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToNetworkFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		/*for (rapidjson::Value::ConstMemberIterator itr = jsonDocument.MemberBegin();
		    itr != jsonDocument.MemberEnd(); ++itr){
			cout << "Type of member  " << itr->name.GetString() << endl;
		}
		rapidjson::Value::ConstMemberIterator itr = jsonDocument.MemberBegin();
		cout << "Let's find the first object " << itr->name.GetString() << endl;
		a = jsonDocument[itr->name.GetString()];
		_ASSERT(a.IsArray());
		real_t nTanks = (uint_t) a[0].GetFloat();
		cout << "nTanks " << nTanks << endl;
		itr = itr+2;
		a = jsonDocument[itr->name.GetString()];
		_ASSERT(a.IsArray());
		real_t n = (uint_t) a[0].GetFloat();
		cout << "string " << itr->name.GetString() << n << endl;
		delete [] readBuffer;
		readBuffer = NULL;
		fclose(infile);
	}*/
	//try{
		//cout << myTesting->testScenarioTree() << endl;
	//}catch (exception &e){
	//cout << e.what() << __LINE__ << endl;
	//}


	//string pathToScenarioTreeFile = "../dataFiles/scenarioTree.json";
	//string pathToForecastFile = "../dataFiles/forecastor.json";
	//string pathToTestFile = "../dataFiles/testVariables.json";
	//string pathToSmpcConfigFile = "../dataFiles/controllerConfig.json";

	cout << "bye bye" << endl;
	return 0;
}
