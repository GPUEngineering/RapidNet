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


#ifndef TEST_CUH_
#define TEST_CUH_
#include "TestSmpcController.cuh"
//#include "../SmpcController.cuh"
//#include "test/TestSmpcController.cuh"
//#include "../rapidjson/document.h"
//#include "../rapidjson/rapidjson.h"
//#include "../rapidjson/filereadstream.h"
#define VARNAME_BETA "beta"
#define VARNAME_UHAT "uHat"
#define VARNAME_VEC_E "vecE"
#define VARNAME_SCE_NOD "scenarioNodes"
#define VARNAME_PHI "Phi"
#define VARNAME_PSI "Psi"
#define VARNAME_OMEGA "omega"
#define VARNAME_D "d"
#define VARNAME_F "f"
#define VARNAME_GBAR "Gbar"
#define VARNAME_THETA "Theta"
#define VARNAME_G "g"
#define VARNAME_BBAR "Bbar"
#define VARNAME_SYS_F "sysF"
#define VARNAME_SYS_G "sysG"
#define VARNAME_TEST_XMIN "xmin"
#define VARNAME_TEST_XMAX "xmax"
#define VARNAME_TEST_XS "xs"
#define VARNAME_TEST_UMIN "umin"
#define VARNAME_TEST_UMAX "umax"
#define VARNAME_TEST_ALPHA_PRICE "costAlpha"



class Testing{
public:
	Testing();
	uint_t testNetwork();
	uint_t testScenarioTree();
	uint_t testForecaster();
	uint_t testControllerConfig();
	uint_t testEngineTesting();
	uint_t testSmpcController();
	uint_t testNewEngineTesting();
	uint_t testSmpcFbeController();
	uint_t testSmpcNamaController();
	~Testing();
private:
	template<typename T> uint_t compareArray(T* arrayA);
	template<typename T> uint_t compareDeviceArray(T* arrayA);
	template<typename T> uint_t compareDeviceScenarioArray(T* arrayA, uint_t *nodes, uint_t dim, uint_t arraySize);
	//int compareArray(real_t* arrayA);
	string pathToFileNetwork;
	string pathToFileForecaster;
	string pathToFileScenarioTree;
	string pathToFileControllerConfig;
	string pathToFileEnigne;
	string pathToFileControllerFbeConfig;
	string pathToFileControllerNamaConfig;
	rapidjson::Value a;
};


#endif /* TEST_CUH_ */
