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

#ifndef DEFINITIONHEADER_H_
#define DEFINITIONHEADER_H_

#include <iostream>
#include <cstdio>
#include <string>
#include <stdexcept>

typedef int uint_t;
typedef float real_t;

using namespace std;

/**
 * Handler for CUDA calls.
 */
#define _CUDA(call) \
		do \
		{ \
			cudaError_t err = (call); \
			if(cudaSuccess != err) \
			{ \
				cerr << "CUDA Error: \nFile = " << __FILE__ << "\nLine = " << __LINE__<< " \nReason = " << cudaGetErrorString(err);\
				cudaDeviceReset(); \
				exit(EXIT_FAILURE); \
			} \
		} \
		while (0)

/**
 * Handler for CUBLAS calls.
 */
#define _CUBLAS(call) \
		do \
		{ \
			cublasStatus_t status = (call); \
			if(CUBLAS_STATUS_SUCCESS != status) \
			{ \
				cerr << "CUBLAS Error: \nFile = " << __FILE__ << "\nLine = " << __LINE__<< " \nReason = " << status;\
				cudaDeviceReset(); \
				exit(EXIT_FAILURE); \
			} \
			\
		} \
		while(0)

/**
 * Assert that a condition holds; throw an std:logic_error if the condition is
 * not satisfied.
 */
#define _ASSERT(cond)\
	do \
	{\
		if(!(cond))\
		{ \
			cerr << " Error: \nFile = " << __FILE__ << "\nLine = " <<__LINE__ << "\n"; \
			exit(EXIT_FAILURE);\
		}\
	}\
	while (0)


#endif /* DEFINITIONHEADER_H_ */
