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
#define VARNAME_DIM_DEMAND "dimDemand"
#define VARNAME_DIM_PRICES "dimPrices"
#define VARNAME_DHAT "dHat"
#define VARNAME_ALPHAHAT "alphaHat"
#include "Configuration.h"

/**
 * The uncertainty in the network are water demand and electricity
 * prices. Using a time series models, a nominal future water demand and
 * nominal electricity prices is predicted.
 *
 * A forecaster class contains:
 *    - nominal water demand
 *    - nominal electricity prices
 * @todo remove print statements
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
	/**
	 * Default destructor.
	 */
	~Forecaster();

	/*
	 * returns the prediction horizon
	 */
	uint_t getPredHorizon();

	/*
	 * returns the dimension of the demand
	 */
	uint_t getDimDemand();

	/*
	 * return the dimension of the prices
	 */
	uint_t getDimPrice();
	/*
	 * returns the pointer of the array of nominal demands
	 */
	real_t* getNominalDemand();

	/*
	 * returns the pointer of the array of nominal prices
	 */
	real_t* getNominalPrices();
private:
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
};



#endif /* FORECASTCLASS_CUH_ */
