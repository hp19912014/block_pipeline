/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <cooperative_groups.h>
#include "polybenchUtilFuncts.h"
#include <cuda_runtime.h>

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 1024
#define NJ 1024
#define NK 1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

#define NUM_BLOCK 1024
#define DIM_BLOCK_VECTOR 256
#define NUM_BLOCK_COMPUTE 32768   //1024*32


#define NUM_SM 80
#define NUM_SM_COMPUTE 78
#define NUM_SM_HtoD_A 1
#define NUM_SM_HtoD_B 1

#define IN_CHUNK_SIZE 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


#define DUMMY_N 1000

__device__ void dummy_comp()
{
    double sum = 0.0;
    for (int i = 0; i < DUMMY_N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__device__ int flag_global_read(volatile int * flag, int rid)
{
        return(flag[rid]);
}

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
      		A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
	//printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( 0 );
}



__global__ void gemm_kernel_block_spin(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *A_host, DATA_TYPE *B_host, int *inflag_A, int *inflag_B)
{
	__shared__ DATA_TYPE A_tile[32][32];
	__shared__ DATA_TYPE B_tile[32][32];

	const int tidx = threadIdx.x;
        const int tidy = threadIdx.y;

	if (blockIdx.x < NUM_SM_HtoD_A ){
		//copy matrix A by 32 columns
		
		for (int bid = blockIdx.x; bid < DIM_BLOCK_VECTOR; bid += NUM_SM_HtoD_A * IN_CHUNK_SIZE){

		    for (int i = 0; i< IN_CHUNK_SIZE;i++){

			int bid_chunk = bid + NUM_SM_HtoD_A*i;
			int block_row = bid_chunk % 32;
			int block_offset = bid_chunk / 32;

			int offset = 8192*block_row+256*tidy+block_offset*32+tidx;
			reinterpret_cast<double2*>(A)[offset] = reinterpret_cast<double2*>(A_host)[offset]; 
			}

			__syncthreads();
			__threadfence();
			if ((tidx< IN_CHUNK_SIZE)&&(tidy==0)){
				int bb = bid + NUM_SM_HtoD_A*tidx;
				int br = bb % 32;
				int bo = bb / 32;
				atomicOr(&inflag_A[br*8+bo],1);	
			}

			
	
		}
		

	}else if (blockIdx.x < (NUM_SM_HtoD_A+NUM_SM_HtoD_B)){
		//copy matrix B by 32 rows
		
                for (int bid = (blockIdx.x-NUM_SM_HtoD_A); bid < DIM_BLOCK_VECTOR; bid += NUM_SM_HtoD_B * IN_CHUNK_SIZE){

			for (int i = 0 ; i < IN_CHUNK_SIZE;i ++) {
			
			int bid_chunk = bid + NUM_SM_HtoD_B*i;	
                        int block_row = bid_chunk / 8;
                        int block_offset = bid_chunk % 8;
                        int offset = 8192*block_row+256*tidy+block_offset*32+tidx;
                        reinterpret_cast<double2*>(B)[offset] = reinterpret_cast<double2*>(B_host)[offset];

			}

	                __syncthreads();

			__threadfence();
                	if ((tidx< IN_CHUNK_SIZE)&&(tidy==0)){
                        	atomicOr(&inflag_B[bid+NUM_SM_HtoD_B*tidx],1);
                	}

		}
		
		
	}else{
		
		//compute blocks
			
		for (int bid= (blockIdx.x-(NUM_SM_HtoD_A+NUM_SM_HtoD_B)); bid < NUM_BLOCK_COMPUTE; bid += NUM_SM_COMPUTE){
	

		int tid = bid/ 1024;
		int bb = bid % 1024;

		int bidx = bb % 32;
		int bidy = bb / 32;
		
		
		int blockA,blockB;
					
		blockA = bidy*8+tid/4;
		blockB = tid*8+bidx/4;
		
		
		if ((tidx==0)&&(tidy==0)){
			//while ( (atomicAnd(&inflag_A[blockA],1)==0) ){ 
			while (flag_global_read(inflag_A,blockA) != 1){
			//dummy_comp();
			}
		}
	
		if ((tidx==1)&&(tidy==0)){
			//while ( (atomicAnd(&inflag_B[blockB],1)==0) ){ 
			while (flag_global_read(inflag_B,blockB) != 1){
			//dummy_comp();
			}
	
		}

		__syncthreads();
		
		

		DATA_TYPE  accu= 0;
	 	int i,j;
		
		i = bidy*32+tidy;
		j = bidx*32+tidx;

                A_tile[threadIdx.y][threadIdx.x] = A[i*NJ+tid*32+threadIdx.x];
                B_tile[threadIdx.y][threadIdx.x] = B[(tid*32+threadIdx.y)*NJ+j];

		__syncthreads();
		
		for (int k = 0; k < 32; k++){
                        accu += ALPHA * A_tile[threadIdx.y][k]*B_tile[k][threadIdx.x];
                }
		
		__syncthreads();
		
		
			if (tid==0){
				C[i*NJ+j] *= BETA;
			}		
		
			C[i*NJ+j] += accu;
		

		}
	       
	    }
	
}

void gemmCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	
	int *inflag_A, *inflag_B;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

	cudaMalloc((void **)&inflag_A,  sizeof(int) * DIM_BLOCK_VECTOR);
	cudaMalloc((void **)&inflag_B,  sizeof(int) * DIM_BLOCK_VECTOR);

	cudaMemset(inflag_A, 0, sizeof(int) * DIM_BLOCK_VECTOR);
	cudaMemset(inflag_B, 0, sizeof(int) * DIM_BLOCK_VECTOR);

        cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(NUM_SM), (size_t)1 );
		
	void *kernelArgs[] = {
                (void *)&A_gpu,             (void *)&B_gpu,
                (void *)&C_gpu,         (void *)&A,
                (void *)&B,        (void *)&inflag_A, (void*)&inflag_B
        };

	cudaEventRecord(start);


	 
	//verify compute


	//cudaMemset(inflag_A, 0, sizeof(int) * DIM_BLOCK_VECTOR);
	//cudaMemset(inflag_B, 0, sizeof(int) * DIM_BLOCK_VECTOR);


	//cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	//cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	
	//gemm_kernel_block_spin<<< grid, block >>>(A_gpu, B_gpu, C_gpu, A,B,inflag_A, inflag_B);
	cudaLaunchCooperativeKernel((void*)gemm_kernel_block_spin, grid, block, kernelArgs,0, NULL);

	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    

	cudaEventRecord(stop);
        cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        fprintf(stdout,"GPU RunTime= %.2f Ms \n",  elapsedTimeInMs);
	
	
	//verify copy results
	/*	
	DATA_TYPE *AA, *BB;
	int *FA,*FB;
	cudaHostAlloc((void **)&AA, sizeof(DATA_TYPE) * NI * NK, cudaHostAllocPortable);
	cudaHostAlloc((void **)&BB, sizeof(DATA_TYPE) * NI * NK, cudaHostAllocPortable);
	
	cudaHostAlloc((void **)&FA, sizeof(int) * DIM_BLOCK_VECTOR, cudaHostAllocPortable);
	cudaHostAlloc((void **)&FB, sizeof(int) * DIM_BLOCK_VECTOR, cudaHostAllocPortable);	

	cudaMemcpy(AA, A_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
	cudaMemcpy(BB, B_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(FA, inflag_A, sizeof(int) * DIM_BLOCK_VECTOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(FB, inflag_B, sizeof(int) * DIM_BLOCK_VECTOR, cudaMemcpyDeviceToHost);

	compareResults(A, AA);
	compareResults(B, BB);

	for(int i = 0 ; i < 256; i++) {
		fprintf(stdout, "%d",FA[i]);
	}	


	for(int i = 0 ; i < 256; i++) {
		fprintf(stdout, "%d",FB[i]);
	}	

	*/

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;  
	DATA_TYPE* C_outputFromGpu; 

	//A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	//B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	//C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
	//C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

	cudaHostAlloc((void **)&A, sizeof(DATA_TYPE) * NI * NK, cudaHostAllocPortable);
	cudaHostAlloc((void **)&B, sizeof(DATA_TYPE) * NK * NJ, cudaHostAllocPortable);
	cudaHostAlloc((void **)&C, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	cudaHostAlloc((void **)&C_outputFromGpu, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);



	init(A, B, C);
	
	GPU_argv_init();
	
	gemmCuda(A, B, C, C_outputFromGpu);
	
		
	/*		
	t_start = rtclock();	
	gemm(A, B, C);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(C, C_outputFromGpu);
	*/
	
	cudaFree(A);
	cudaFree(B);  
	cudaFree(C);  
	cudaFree(C_outputFromGpu); 
	
    	return 0;
}

