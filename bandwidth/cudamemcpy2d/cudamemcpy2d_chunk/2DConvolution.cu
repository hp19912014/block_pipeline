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
#include <cuda.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define NI 2048
#define NJ 2048
#define NK 2048
#define NUM NI*NJ
/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#define NUM_CHUNK 1024
#define CHUNK_SIZE NI/NUM_CHUNK

#define NUM_SM 8
#define OFFSET NUM_SM*DIM_THREAD_BLOCK_X
/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;


void init(DATA_TYPE ** A, DATA_TYPE **B)
{
	int i, j;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
			B[i][j] = A[i][j];
        	}
    	}
}


void compareResults(DATA_TYPE **B, DATA_TYPE **B_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=1; i < (NI-1); i++) 
	{
		for (j=1; j < (NJ-1); j++) 
		{
			if (percentDiff(B[i][j], B_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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


void bandwidthtest(DATA_TYPE ** A,DATA_TYPE** B)
{

	DATA_TYPE *A_gpu;
	size_t pitch;

	cudaEvent_t start[2],stop[2];
	float elapsedTimeInMs = 0.0f;
        float bandwidthInMBs = 0.0f;

	
	cudaEventCreate(&start[0]);
	cudaEventCreate(&start[1]);
	cudaEventCreate(&stop[0]);
	cudaEventCreate(&stop[1]);

	cudaMallocPitch(&A_gpu, &pitch, NJ*sizeof(DATA_TYPE),NI);
	fprintf(stdout,"finish malloc pitch\n");	
	
	//cudamemcpyasync2d  HtoD
	cudaEventRecord(start[0],0);
	for (int i = 0 ;i <NUM_CHUNK;i++){
	cudaMemcpy2DAsync(A_gpu+CHUNK_SIZE*i,pitch,&A[0][CHUNK_SIZE*i], NJ*sizeof(DATA_TYPE),CHUNK_SIZE*sizeof(DATA_TYPE),NI,cudaMemcpyHostToDevice,0);
	}
	fprintf(stdout,"finish copy2d htod \n");
	
	cudaDeviceSynchronize();
	cudaEventRecord(stop[0]);
	cudaEventSynchronize(stop[0]);
	cudaEventElapsedTime(&elapsedTimeInMs, start[0], stop[0]);	
	bandwidthInMBs = ((float)(1<<10) * NI*NJ*sizeof(DATA_TYPE) ) / (elapsedTimeInMs * (float)(1 << 20));	
	fprintf(stdout,"HtoD cudamemcpy2d  Bandwidth = %.1f MB/s, Time= %.1f Ms \n", bandwidthInMBs, elapsedTimeInMs);
	
	//cudamemcpyasync2d DtoH
	cudaEventRecord(start[1],0);
	
	for  (int i =0 ;i <NUM_CHUNK; i++){
	cudaMemcpy2DAsync(&B[0][CHUNK_SIZE*i], NJ*sizeof(DATA_TYPE),A_gpu+CHUNK_SIZE*i,pitch,CHUNK_SIZE*sizeof(DATA_TYPE),NI,cudaMemcpyDeviceToHost,0);
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop[1]);
	cudaEventSynchronize(stop[1]);
	cudaEventElapsedTime(&elapsedTimeInMs, start[1], stop[1]);	
	bandwidthInMBs = ((float)(1<<10) * NI*NJ*sizeof(DATA_TYPE) ) / (elapsedTimeInMs * (float)(1 << 20));	
	fprintf(stdout,"DtoH cudamemcpy2d Bandwidth = %.1f MB/s, Time= %.1f Ms \n", bandwidthInMBs, elapsedTimeInMs);
	
	cudaEventDestroy(start[0]);
	cudaEventDestroy(start[1]);
	cudaEventDestroy(stop[0]);
	cudaEventDestroy(stop[1]);
	cudaFree(A_gpu);
}

int main(int argc, char *argv[])
{
    int t;
    DATA_TYPE  **C_outputFromGpu, **A, **B;
    DATA_TYPE *co,*aa,*bb;

    C_outputFromGpu=(DATA_TYPE **)malloc(sizeof(DATA_TYPE *)*NI);
    cudaHostAlloc((void **)&co, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
    for (t=0;t<NI;t++)
        C_outputFromGpu[t]=co+t*NJ;

    A=(DATA_TYPE **)malloc(sizeof(DATA_TYPE *)*NI);
    cudaHostAlloc((void **)&aa, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
    for (t=0;t<NI;t++)
        A[t]=aa+t*NK;

    B=(DATA_TYPE **)malloc(sizeof(DATA_TYPE *)*NK);
    cudaHostAlloc((void **)&bb, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
    for (t=0;t<NK;t++)
        B[t]=bb+t*NJ;


    fprintf(stdout,"finish allocation\n");
    init(A,B);
    fprintf(stdout,"finish initialization\n");
   
    GPU_argv_init();

    bandwidthtest(A,C_outputFromGpu);

    fprintf(stdout,"finish test\n");
    
    compareResults(C_outputFromGpu,B);
    
    fprintf(stdout,"finish verify\n");
    return 0;
}

