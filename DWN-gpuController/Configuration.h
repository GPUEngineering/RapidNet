/*
 * DefinitionHeader.h
 *
 *  Created on: Mar 1, 2017
 *      Author: control
 */

#ifndef DEFINITIONHEADER_H_
#define DEFINITIONHEADER_H_


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
#include <iostream>
#include <cstdio>
#include <string>

typedef int uint_t;
typedef float real_t;

using namespace std;


#endif /* DEFINITIONHEADER_H_ */
