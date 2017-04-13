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

#ifndef FORECASTCLASS_CUH_
#define FORECASTCLASS_CUH_
#define VARNAME_N  "N"
#define VARNAME_SIM_HORIZON "simHorizon"
#define VARNAME_DIM_DEMAND "dimDemand"
#define VARNAME_DIM_PRICES "dimPrices"
#define VARNAME_DHAT "dHat"
#define VARNAME_ALPHAHAT "alphaHat"
#define VARNAME_DEMAND_SIM "timeIdDemand4876"
#define VARNAME_PRICE_SIM "timeIdPrice4876"
#include "Configuration.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filereadstream.h"

/**
 * The uncertainty in the network are water demand and electricity
 * prices. Using a time series models, a nominal future water demand and
 * nominal electricity prices is predicted.
 *
 * A forecaster class contains:
 *    - nominal water demand
 *    - nominal electricity prices
 * @todo sanity check (check that the given file is well formed)
 * @todo new char[65536]: is this good practice?
 */
class Forecaster{

public:

	/*
	 *  Constructor of a Forecaster entity from a given JSON file.
	 *
 	 * @param pathToFile filename of a JSON file containing the forecasts
 	 *        of the water demand and electricity price.
	 */
	Forecaster(
		string pathToFile);
	/*
	 * @return the prediction horizon
	 */
	uint_t getPredHorizon();
	/**
	 * @return simulation Horizon of the nominal forecaster
	 */
	uint_t getSimHorizon();
	/*
	 * @returns the dimension of the demand
	 */
	uint_t getDimDemand();
	/*
	 * @return the dimension of the prices
	 */
	uint_t getDimPrice();
	/*
	 *@returns the pointer of the array of nominal demands
	 */
	real_t* getNominalDemand();

	/*
	 * @returns the pointer of the array of nominal prices
	 */
	real_t* getNominalPrices();
	/**
	 * Predicts the water demand. The base class forecaster
	 * reads the nominal demand from a json file.
	 * @return status  1 success and 0 failure
	 */
	virtual uint_t predictDemand(uint_t simTime);
	/**
	 * Predicts the prices. The base class forecaster
	 * reads the nominal prices from a json file.
	 * @return status  1 success and 0 failure
	 */
	virtual uint_t predictPrices(uint_t simTime);
	/**
	 * Default destructor.
	 */
	virtual ~Forecaster();
private:
	/**
	 * simulation horizon used in the forecaster
	 */
	uint_t simHorizon;
	/**
	 * Prediction horizon
	 */
	uint_t nPredHorizon;
	/**
	 * Number of demands
	 */
	uint_t dimDemand;
	/**
	 * Dimension of the electricity prices
	 */
	uint_t dimPrices;
	/**
	 * Nominal demand predicted
	 */
	real_t *nominalDemand;
	/**
	 * Nominal electricity prices
	 */
	real_t *nominalPrice;
	/**
	 * string path that contain the nominal demands and nominal prices
	 */
	string pathToNominalDemand;
	/**
	 * Member iterator to read demand in the json file.
	 */
	rapidjson::Value::ConstMemberIterator itrNominalDemand;
	/**
	 * Member iterator to read demand in the json file.
	 */
	rapidjson::Value::ConstMemberIterator itrNominalPrices;
	/**
	 * Document (rapidjson) object to read the json file
	 */
	rapidjson::Document jsonDocument;
	/**
	 * Value (rapidjson) object to find the value of a member in the json file
	 */
	rapidjson::Value jsonValue;
};



#endif /* FORECASTCLASS_CUH_ */
