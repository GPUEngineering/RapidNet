/*
 * unitTestClass.cuh
 *
 *  Created on: Mar 8, 2017
 *      Author: control
 */

#ifndef UNITTESTCLASS_CUH_
#define UNITTESTCLASS_CUH_



class unitTest{
public:
	unitTest(string pathToFile);
	~unitTest();
	template<typename T> void checkObjectiveMatR(T *engineR);
	friend class Engine;
private:
	uint_t NX, NU, NV;
	real_t *matR;
};


unitTest::unitTest(string pathToFile){
	cout << "allocating memory for the test matrices \n";
	const char* fileName = pathToFile.c_str();
	rapidjson::Document jsonDocument;
	rapidjson::Value a;
	FILE* infile = fopen(fileName, "r");
	if(infile == NULL){
		cout << pathToFile << infile << endl;
		cerr << "Error in opening the file " <<__LINE__ << endl;
		exit(100);
	}else{
		char* readBuffer = new char[65536];
		rapidjson::FileReadStream networkJsonStream(infile, readBuffer, sizeof(readBuffer));
		jsonDocument.ParseStream(networkJsonStream);
		a = jsonDocument["nx"];
		assert(a.IsArray());
		NX = (uint_t) a[0].GetDouble();
		a = jsonDocument["nu"];
		assert(a.IsArray());
		NU = (uint_t) a[0].GetDouble();
		a = jsonDocument["nv"];
		assert(a.IsArray());
		NV = (uint_t) a[0].GetDouble();
		matR = new real_t[NV * NV];
		a = jsonDocument["matR"];
		assert(a.IsArray());
		for (rapidjson::SizeType i = 0; i < a.Size(); i++)
			matR[i] = a[i].GetDouble();
		delete [] readBuffer;
	}
	fclose(infile);
}

template<typename T> void unitTest::checkObjectiveMatR(T *engineR){
	real_t *hostEngineR = new real_t[NV*NV];
	_CUDA( cudaMemcpy( hostEngineR, engineR, NV*NV*sizeof(real_t), cudaMemcpyDeviceToHost) );
	/*
	for (int iRow = 0; iRow < NV; iRow++){
		for(int iCol = 0; iCol < NV; iCol++){
			cout<< matR[iRow*NV + iCol] - hostEngineR[iRow*NV + iCol] << " ";
		}
		cout << "\n";
	}*/
	for( int i = 0; i < NV*NV; i++){
		if(abs(matR[i] - hostEngineR[i]) > 1e-3)
			cout<< matR[i] << " " << hostEngineR[i] << " " << i << " ";
	}
	delete hostEngineR;
}

unitTest::~unitTest(){
	delete [] matR;
}

#endif /* UNITTESTCLASS_CUH_ */
