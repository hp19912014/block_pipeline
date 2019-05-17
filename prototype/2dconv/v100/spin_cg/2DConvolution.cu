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
#include <cuda_runtime.h>
#include <cooperative_groups.h>
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define NI 4096
#define NJ 4096
#define NJ_VECTOR NJ/2

#define NUM NI*NJ
#define NUM_VECTOR NUM/2
/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#define DIM_ROW (NJ/DIM_THREAD_BLOCK_X)       //4096/1024=4
#define DIM_COLUMN (NI/DIM_THREAD_BLOCK_Y)    //4096/1=4096
#define DIM_BLOCK DIM_ROW*DIM_COLUMN		  //4*4096=16384
#define DIM_BLOCK_VECTOR DIM_BLOCK/2		//2*4096


// for nvlink
#define NUM_SM 80
#define NUM_SM_COMPUTE 72
#define NUM_SM_HtoD 4
#define OFFSET NUM_SM_HtoD * DIM_THREAD_BLOCK_X
#define NUM_SM_DtoH 4 

// for pcie
/*
#define NUM_SM 80
#define NUM_SM_COMPUTE 77
#define NUM_SM_HtoD 2
#define OFFSET NUM_SM_HtoD * DIM_THREAD_BLOCK_X
#define NUM_SM_DtoH 1 
*/

#define ROW_CHUNK_SIZE 32
#define ROW_CHUNK NJ/ROW_CHUNK_SIZE	//128
#define ROW_CHUNK_OFFSET OFFSET*ROW_CHUNK_SIZE*2	//1024*2*32

#define IN_CHUNK_SIZE 32
#define IN_CHUNK NJ*2/NUM_SM_HtoD/ROW_CHUNK_SIZE // 4096*2/4/32 = 64
#define IN_CHUNK_OFFSET OFFSET*IN_CHUNK_SIZE      //1024*4*32 = 131072

#define OUT_CHUNK_SIZE 32
#define OUT_CHUNK NJ/OUT_CHUNK_SIZE	//128
#define OUT_CHUNK_OFFSET OFFSET*OUT_CHUNK_SIZE*2	//1024*2*32


/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

#define DUMMY_N 1000

__device__ void dummy_comp()
{
    double sum = 0.0;
    for (int i = 0; i < DUMMY_N; i++) 
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

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

__device__ int flag_global_read(volatile int * flag, int rid)
{
	return(flag[rid]);
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
	printf("setting device %d \n",0);
	cudaSetDevice( 0 );
}

__global__ void Convolution2D_kernel(DATA_TYPE *A_host, DATA_TYPE *B_host, DATA_TYPE *A, DATA_TYPE *B, int *inflag, int* outflag)
{

	if (blockIdx.x < NUM_SM_HtoD ){  //copy kernel HtoD
		/*
		const int idx = threadIdx.x;
		const int bidx = blockIdx.x;
		for (int i = bidx*1024+idx; i< NUM_VECTOR; i+= 4096 ){
	      		reinterpret_cast<double2*>(A)[i] = reinterpret_cast<double2*>(A_host)[i];
			__syncthreads();
			if (idx == 0){
			atomicOr(&inflag[(i>>10)],1);	
			}
		}	
		*/
		// groupping chunks
		const int idx = threadIdx.x;
                const int bidx = blockIdx.x;
		int chunk_offset;
		for (int i = 0; i<IN_CHUNK;i++ ){
			chunk_offset=i*IN_CHUNK_OFFSET;

			for (int k = (chunk_offset+bidx*1024+idx);k < ( chunk_offset+IN_CHUNK_OFFSET ) ; k+= OFFSET ){
			reinterpret_cast<double2*>(A)[k] = reinterpret_cast<double2*>(A_host)[k];
			}
			
			__syncthreads();
			__threadfence();
			if ( idx < IN_CHUNK_SIZE ){
            		atomicOr(&inflag[ (IN_CHUNK_SIZE*4*i) + idx*4+ bidx],1);
     		   	}
		}
			
	
	}else if (blockIdx.x < (NUM_SM_COMPUTE+NUM_SM_HtoD )){     //compute 
			
			//int bid = blockIdx.x;
			
			DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

			c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
			c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
			c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
			
			//initialize output 
	 		/*	
			if((blockIdx.x==45)&&(threadIdx.x==0))
			{
				atomicAdd(&outflag[0],2);
				atomicAdd(&outflag[1],2);
				atomicAdd(&outflag[8190],2);
				atomicAdd(&outflag[8191],2);
			}
			*/
			 	
	
			//while( bid < (DIM_BLOCK -4) )
			for (int bid=blockIdx.x ; bid < (DIM_BLOCK-4); bid+=NUM_SM_COMPUTE)	{

				int i= bid >>2;
				int j_base = bid & 3;
				int j = j_base * DIM_THREAD_BLOCK_X + threadIdx.x;
				int fid = bid >>1 ;

				if(threadIdx.x==0)    //spin ....wait for data ready
					{
						while( ( atomicAnd(&inflag[fid-2],1) == 0 )) 
						//while (flag_global_read(inflag,i+1 )==0)
						{
						//dummy_comp();
						}
					}
				
				if(threadIdx.x==1)    //spin ....wait for data ready
					{
						while( ( atomicAnd(&inflag[fid],1) == 0 )) 
						//while (flag_global_read(inflag,i+1 )==0)
						{
						//dummy_comp();
						}
					}

				if(threadIdx.x==2)    //spin ....wait for data ready
					{
						while( ( atomicAnd(&inflag[fid+2],1) == 0 )) 
						//while (flag_global_read(inflag,i+1 )==0)
						{
						//dummy_comp();
						}
					}
		
				if(threadIdx.x==3)    //spin ....wait for data ready
					{
						while( ( atomicAnd(&inflag[fid+1],1) == 0 )) 
						//while (flag_global_read(inflag,i+1 )==0)
						{
						//dummy_comp();
						}
					}
				
				if(threadIdx.x==4)    //spin ....wait for data ready
					{
						while( ( atomicAnd(&inflag[fid-1],1) == 0 )) 
						//while (flag_global_read(inflag,i+1 )==0)
						{
						//dummy_comp();
						}
					}
	
	

				__syncthreads();

				if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
				{
					//compute for block bid
					B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
							+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
							+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
				}

				//make sure compute has been finished
				__syncthreads();
				__threadfence();
				if(threadIdx.x==0)
				{
				      atomicAdd(&outflag[fid],1);
				     //outflag[i]=1;
				    //atomicCAS(&outflag[i],4,1);
				    	
				}

				//bid += NUM_SM_COMPUTE;
    		}		
	
	}
	else{	//copy kernel DtoH
	   		   
		const int idx = threadIdx.x;
		const int bidx = blockIdx.x-76;
		int rid = 0;
                for (int i =  2048+bidx*1024+idx; i< NUM_VECTOR-2048; i+= 4096){
			rid = i>>10;
			  while(  flag_global_read(outflag,rid) != 2 )
			 // while ( atomicAnd(&outflag[rid],3) == 0 )
                          {
                             //dummy_comp();
                          }

                reinterpret_cast<double2*>(B_host)[i] = reinterpret_cast<double2*>(B)[i];
		}
	    	

	   }
		
}


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	int *inflag,*outflag;
	//alloc
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMalloc((void **)&inflag,  sizeof(int) * DIM_BLOCK_VECTOR);
	cudaMalloc((void **)&outflag, sizeof(int) * DIM_BLOCK_VECTOR);

	//initial
	cudaMemset(inflag, 0, sizeof(int) * DIM_BLOCK_VECTOR);
	cudaMemset(outflag, 0, sizeof(int) * DIM_BLOCK_VECTOR);

	//cudaevent

	cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(NUM_SM), (size_t)1 );
	
	void *kernelArgs[] = {
                (void *)&A,             (void *)&B_outputFromGpu,
                (void *)&A_gpu,         (void *)&B_gpu,
                (void *)&inflag,        (void *)&outflag
        };

	cudaEventRecord(start);
	//Convolution2D_kernel<<<grid,block>>>(A,B_outputFromGpu,A_gpu,B_gpu,inflag,outflag);
	cudaLaunchCooperativeKernel((void*)Convolution2D_kernel, grid, block, kernelArgs,0, NULL);
	//debug verify data
	//cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
	//cudaMemcpy(B_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);

      	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	fprintf(stdout,"GPU RunTime= %.2f Ms \n",  elapsedTimeInMs);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}




int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* B_outputFromGpu;
	

	cudaHostAlloc((void **)&A, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	cudaHostAlloc((void **)&B, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	cudaHostAlloc((void **)&B_outputFromGpu, sizeof(DATA_TYPE) * NI * NJ, cudaHostAllocPortable);
	//initialize the arrays
	init(A);
	
	GPU_argv_init();
	
	/*
	convolution2DCuda(A, B, B_outputFromGpu);


	t_start = rtclock();
	conv2D(A, B);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(B, B_outputFromGpu);
	//compareResults(A, B_outputFromGpu);
	*/
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(B_outputFromGpu);
	
	return 0;
}

