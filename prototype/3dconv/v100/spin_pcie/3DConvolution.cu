/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size */
#define NI 256
#define NJ 256
#define NK 256

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#define DIM_BLOCK NJ*NI/4  // 256*256/4 = 16384
#define DIM_BLOCK_VECTOR DIM_BLOCK/2 //8192

#define NUM NI*NJ*NK
#define NUM_VECTOR NUM/2

#define NUM_SM 80
#define NUM_SM_COMPUTE 77
#define NUM_SM_HtoD 2
#define OFFSET NUM_SM_HtoD * DIM_THREAD_BLOCK_X
#define NUM_SM_DtoH 1

#define IN_CHUNK_SIZE 32
#define IN_CHUNK NJ*NI/8/NUM_SM_HtoD/IN_CHUNK_SIZE // 256*256/8/4/32 = 64
#define IN_CHUNK_OFFSET OFFSET*IN_CHUNK_SIZE      //  1024*64= 65536

#define OUT_CHUNK_SIZE 32

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

void conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}

__device__ int flag_global_read(volatile int * flag, int rid)
{
	return(flag[rid]);
}

void init(DATA_TYPE* A)
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
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


__global__ void convolution3D_kernel(DATA_TYPE *A_host, DATA_TYPE *B_host, DATA_TYPE *A, DATA_TYPE *B, int *inflag, int* outflag)
{
	
	if (blockIdx.x < NUM_SM_HtoD ){  //copy kernel HtoD
	
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
            		atomicOr(&inflag[ (IN_CHUNK_SIZE*2*i) + idx*2+ bidx],1);
     		   	}
		}
			
	
	}else if (blockIdx.x < (NUM_SM_COMPUTE+NUM_SM_HtoD )){     //compute 

		DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

		c11 = +2;  c21 = +5;  c31 = -8;
		c12 = -3;  c22 = +6;  c32 = -9;
		c13 = +4;  c23 = +7;  c33 = +10;
	
		for (int bid= blockIdx.x+62 ; bid < (DIM_BLOCK-64); bid+=NUM_SM_COMPUTE){
		
			int i = bid /64 ;
			int j = (bid % 64) *4 + (threadIdx.x /256);
			int k = threadIdx.x %  256;
			int fid = bid >> 1;
			
			if(threadIdx.x==0)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}	
			
			if(threadIdx.x==1)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid+1],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}
			
			if(threadIdx.x==2)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid+31],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}

			if(threadIdx.x==3)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid+32],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}
			
			if(threadIdx.x==4)    //spin ....wait for data ready
			{
				
				if(fid < 8159 ){
					while( ( atomicAnd(&inflag[fid+33],1) == 0 )) 
					//while (flag_global_read(inflag,i+1 )==0)
					{
					//dummy_comp();
					}
				}
			}
			/*
			if(threadIdx.x==5)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid-1],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}
			
			if(threadIdx.x==6)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid-1],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}
	
			if(threadIdx.x==7)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid-1],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}
		
			if(threadIdx.x==8)    //spin ....wait for data ready
			{
				while( ( atomicAnd(&inflag[fid-1],1) == 0 )) 
				//while (flag_global_read(inflag,i+1 )==0)
				{
				//dummy_comp();
				}
			}

			*/

			__syncthreads();

			if ((i < (NI-1)) && (j < (NJ-1)) &&  (k < (NK-1)) && (i > 0) && (j > 0) && (k > 0))
			{
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
			
			__syncthreads();
			__threadfence();
			if(threadIdx.x==0)
			{
			atomicAdd(&outflag[fid],1);
			}	
		}

	}else{	//copy kernel DtoH
	   		   
		const int idx = threadIdx.x;
		const int bidx = blockIdx.x-(NUM_SM_COMPUTE+NUM_SM_HtoD);
		int rid = 0;
                for (int i =  32768+bidx*1024+idx; i< NUM_VECTOR-32768; i+= (1024*NUM_SM_DtoH) ){
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


void convolution3DCuda(DATA_TYPE* A, DATA_TYPE* B_outputFromGpu)
{
	//double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	int *inflag,*outflag;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	
	cudaMalloc((void **)&inflag,  sizeof(int) * DIM_BLOCK_VECTOR);
	cudaMalloc((void **)&outflag, sizeof(int) * DIM_BLOCK_VECTOR);
	//initial flags
	cudaMemset(inflag, 0, sizeof(int) * DIM_BLOCK_VECTOR);
	cudaMemset(outflag, 0, sizeof(int) * DIM_BLOCK_VECTOR);

	cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
       
	cudaEventRecord(start);

	//cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);

	//dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	//dim3 grid((size_t)(ceil( ((float)NK) / ((float)block.x) )), (size_t)(ceil( ((float)NJ) / ((float)block.y) )));
	//dim3 grid((size_t)(NUM_SM) , (size_t) (1) );

	convolution3D_kernel<<< NUM_SM, 1024 >>>(A,B_outputFromGpu,A_gpu, B_gpu,inflag,outflag);

	//cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        fprintf(stdout,"GPU RunTime= %.2f Ms \n",  elapsedTimeInMs);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
}


int main(int argc, char *argv[])
{
	//double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* B_outputFromGpu;

	//A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	//B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	//B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	cudaHostAlloc((void **)&A, sizeof(DATA_TYPE) * NI * NJ * NK, cudaHostAllocPortable);
        cudaHostAlloc((void **)&B, sizeof(DATA_TYPE) * NI * NJ * NK, cudaHostAllocPortable);
        cudaHostAlloc((void **)&B_outputFromGpu, sizeof(DATA_TYPE) * NI * NJ *NK, cudaHostAllocPortable);

	init(A);
	
	GPU_argv_init();

	convolution3DCuda(A, B_outputFromGpu);

	conv3D(A,B);
	compareResults(B, B_outputFromGpu);

	cudaFree(A);
	cudaFree(B);
	cudaFree(B_outputFromGpu);

    	return 0;
}

