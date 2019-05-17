/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define NI 4096
#define NJ 4096
#define NUM NI*NJ
/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#define NUM_SM 4
#define OFFSET NUM_SM*DIM_THREAD_BLOCK_X
/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;



void conv2D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				+ c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
				+ c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}



void init(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			A[i*NJ + j] = (float)rand()/RAND_MAX;
        	}
    	}
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=1; i < (NI-1); i++) 
	{
		for (j=1; j < (NJ-1); j++) 
		{
			if (percentDiff(B[i*NJ + j], B_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
	
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( 0 );
}


__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
	{
		B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
			+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
			+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
	}
}


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	t_start = rtclock();
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
	Convolution2D_kernel<<<grid,block>>>(A_gpu,B_gpu);
	cudaThreadSynchronize();
	cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
	
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);//);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
}

__global__ void Simplekernel(DATA_TYPE *dst, DATA_TYPE *src)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i< NUM/2; i+=OFFSET){
	reinterpret_cast<double2*>(dst)[i] = reinterpret_cast<double2*>(src)[i];	
    }	
 /*
      int offset=0, num=NI*NJ;
	while (offset<num){
	    if (offset+idx< (NI*NJ))
	    dst[offset+idx] = src[offset+idx];
	    offset += blockDim.x * NUM_SM;
	   }
   */
/*
   for (int i = idx; i< NUM; i+= OFFSET ){
	dst[i] = src[i];
   }
*/


}

void bandwidthtest(DATA_TYPE* A,DATA_TYPE* B)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaEvent_t start[2],stop[2];
	float elapsedTimeInMs = 0.0f;
        float bandwidthInMBs = 0.0f;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaEventCreate(&start[0]);
	cudaEventCreate(&start[1]);
	cudaEventCreate(&stop[0]);
	cudaEventCreate(&stop[1]);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(NUM_SM), (size_t)1 );
	//cudamemcpyasync HtoD
	cudaEventRecord(start[0],0);
	cudaMemcpyAsync(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice,0);
	cudaEventRecord(stop[0]);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsedTimeInMs, start[0], stop[0]);	
	bandwidthInMBs = ((float)(1<<10) * NI*NJ*sizeof(DATA_TYPE) ) / (elapsedTimeInMs * (float)(1 << 20));	
	fprintf(stdout,"HtoD cudamemcpyasync Bandwidth = %.1f MB/s, Time= %.1f Ms \n", bandwidthInMBs, elapsedTimeInMs);
	//Simplekernel HtoD
	cudaEventRecord(start[1]);
	Simplekernel<<<grid,block,0,0>>>(B_gpu, A);
	cudaEventRecord(stop[1]);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsedTimeInMs, start[1], stop[1]);	
	bandwidthInMBs = ((float)(1<<10) * NI*NJ*sizeof(DATA_TYPE) ) / (elapsedTimeInMs * (float)(1 << 20));	
 	fprintf(stdout,"HtoD simplekernel Bandwidth = %.1f MB/s, Time= %.1f Ms \n", bandwidthInMBs, elapsedTimeInMs);	

	cudaMemcpy(B, B_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);

	compareResults(A, B);
	//cudamemcpyasync DtoH
	cudaEventRecord(start[0],0);
	cudaMemcpyAsync(A, A_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost,0);
	cudaEventRecord(stop[0]);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsedTimeInMs, start[0], stop[0]);	
	bandwidthInMBs = ((float)(1<<10) * NI*NJ*sizeof(DATA_TYPE) ) / (elapsedTimeInMs * (float)(1 << 20));	
	fprintf(stdout,"DtoH cudamemcpyasync Bandwidth = %.1f MB/s, Time= %.1f Ms \n", bandwidthInMBs, elapsedTimeInMs);
	//simplekernel DtoH
	cudaEventRecord(start[1]);
	Simplekernel<<<grid,block,0,0>>>(B, A_gpu);
	cudaEventRecord(stop[1]);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsedTimeInMs, start[1], stop[1]);	
	bandwidthInMBs = ((float)(1<<10) * NI*NJ*sizeof(DATA_TYPE) ) / (elapsedTimeInMs * (float)(1 << 20));	
 	fprintf(stdout,"DtoH simplekernel Bandwidth = %.1f MB/s, Time= %.1f Ms \n", bandwidthInMBs, elapsedTimeInMs);	

	compareResults(A, B);

	cudaEventDestroy(start[0]);
	cudaEventDestroy(start[1]);
	cudaEventDestroy(stop[0]);
	cudaEventDestroy(stop[1]);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* B_outputFromGpu;
	
	//A = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	//B = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	//B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	cudaHostAlloc((void **)&A, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	cudaHostAlloc((void **)&B, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	cudaHostAlloc((void **)&B_outputFromGpu, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	//initialize the arrays
	init(A);
	
	GPU_argv_init();
	
	bandwidthtest(A,B);

	//convolution2DCuda(A, B, B_outputFromGpu);
	/*
	t_start = rtclock();
	conv2D(A, B);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);//);
	
	compareResults(B, B_outputFromGpu);

  	*/
	cudaFree(A);
	cudaFree(B);
	cudaFree(B_outputFromGpu);
	
	return 0;
}
