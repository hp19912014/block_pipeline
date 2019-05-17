/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "polybenchUtilFuncts.h"
#include <cooperative_groups.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 1
#define NX 4096
#define NY 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

#define DIM_ROW 4       //4096/1024=4
#define DIM_COLUMN 4096    //4096/1=4096
#define DIM_BLOCK 16384		  //4*4096=16384
#define DIM_BLOCK_VECTOR DIM_BLOCK/2	

#define NUM_X NX*(NY+1)
#define NUM_Y (NY+1)*NX
#define NUM_Z NX*NY

#define NUM_SM 80

#define NUM_SM_X 1
#define NUM_SM_Y 1
#define NUM_SM_Z 1
#define OFFSET 1024

#define NUM_SM_OUT 4

#define NUM_SM_COMPUTE_X 24
#define NUM_SM_COMPUTE_Y 24
#define NUM_SM_COMPUTE_Z 25

#define SM1 NUM_SM_X              //1
#define SM2 NUM_SM_X+NUM_SM_Y     //2
#define SM3 SM2+NUM_SM_Z          //3
#define SM4 SM3+NUM_SM_COMPUTE_X  //27
#define SM5 SM4+NUM_SM_COMPUTE_Y  //51
#define SM6 SM5+NUM_SM_COMPUTE_Z  //76
#define SM7 SM6+NUM_SM_OUT       //80

#define IN_CHUNK_SIZE 32
#define NUM_CHUNK NX/IN_CHUNK_SIZE // 128
#define OUT_CHUNK_SIZE 32


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


#define DUMMY_N 10000

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

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
	for (t=0; t < tmax; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       		for (j = 0; j < NY; j++)
			{
       			ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}


void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
	cudaSetDevice( GPU_DEVICE );
}



__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		if (i == 0) 
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


__global__ void fdtd_spin_kernel(DATA_TYPE *ex, DATA_TYPE *ex_host, DATA_TYPE *ey, DATA_TYPE *ey_host, DATA_TYPE *hz, DATA_TYPE *hz_host,DATA_TYPE *hz_gpu_out, DATA_TYPE *hz_gpu_out_host, int * inflag_ex, int* inflag_ey,int* inflag_hz,int* flag_ex,int* flag_ey,int* outflag)
{
    if (blockIdx.x ==0 ){   //copy array ex

	const int idx = threadIdx.x;

	int chunk_offset=0;
	const int in_chunk_offset=(NY+1)*(IN_CHUNK_SIZE)/4;

	for (int i = 0 ; i < NUM_CHUNK; i++){
		chunk_offset= i * in_chunk_offset;
		for (int k = (chunk_offset+idx) ; k< (chunk_offset+in_chunk_offset); k+=OFFSET ){
			reinterpret_cast<double2*>(ex)[k] = reinterpret_cast<double2*>(ex_host)[k];
		}
		__syncthreads();
		__threadfence();
		if ( idx < IN_CHUNK_SIZE ){
            		atomicOr(&inflag_ex[i*IN_CHUNK_SIZE+idx],1);
     	}

	}
    	
    }else if (blockIdx.x ==1){  //copy array ey
 
	const int idx = threadIdx.x;

	int chunk_offset=0;
	const int in_chunk_offset=(NY)*(IN_CHUNK_SIZE)/4;

	for (int i = 0 ; i < NUM_CHUNK; i++){
		chunk_offset= i * in_chunk_offset;
		for (int k = (chunk_offset+idx) ; k< (chunk_offset+in_chunk_offset); k+=OFFSET ){
			reinterpret_cast<double2*>(ey)[k] = reinterpret_cast<double2*>(ey_host)[k];
		}
		__syncthreads();
		__threadfence();
		if ( idx < IN_CHUNK_SIZE ){
            		atomicOr(&inflag_ey[i*IN_CHUNK_SIZE+idx],1);
     	}

	}

	if (idx==0){
		atomicOr(&inflag_ey[NX],1);
	}

    }else if (blockIdx.x ==2 ){ //copy array hz

    const int idx = threadIdx.x;

	int chunk_offset=0;
	const int in_chunk_offset=(NY)*(IN_CHUNK_SIZE)/4;

	for (int i = 0 ; i < NUM_CHUNK; i++){
		chunk_offset= i * in_chunk_offset;
		for (int k = (chunk_offset+idx) ; k< (chunk_offset+in_chunk_offset); k+=OFFSET ){
			reinterpret_cast<double2*>(hz)[k] = reinterpret_cast<double2*>(hz_host)[k];
		}
		__syncthreads();
		__threadfence();
		if ( idx < IN_CHUNK_SIZE ){
            		atomicOr(&inflag_hz[i*IN_CHUNK_SIZE+idx],1);
     	}

	}
	
    }else if ((blockIdx.x>=3)&&(blockIdx.x<=26)){ //execute ex
    

    for (int bid = (blockIdx.x-3); bid < (DIM_BLOCK); bid+=24){
    	int  i = bid >>2;
    	int  j_base = bid & 3;
    	int  j = j_base * DIM_THREAD_BLOCK_X + threadIdx.x;
    	 

    	//spin
		if (threadIdx.x==0)    //spin ....wait for data ready
		{
			while( ( atomicAnd(&inflag_ex[i],1) == 0 )) 
			//while (flag_global_read(inflag,i+1 )==0)
			{
			//dummy_comp();
			}
		}

		if (threadIdx.x==1)    //spin ....wait for data ready
		{
			while( ( atomicAnd(&inflag_hz[i],1) == 0 )) 
			//while (flag_global_read(inflag,i+1 )==0)
			{
			//dummy_comp();
			}
		}

    	__syncthreads();

    	//compute ex
	    	if ((i < NX) && (j < NY) && (j > 0))
		{
			ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
		}

		//make sure compute has been finished
		__syncthreads();
		__threadfence();
		//label the flag_ex
		if(threadIdx.x==1023)
		{
		atomicAdd(&flag_ex[i],1);
		}

    }
  


    }else if ((blockIdx.x>=27)&&(blockIdx.x<=50)){ //execute ey

    
    const int idx = threadIdx.x;

    for (int bid = (blockIdx.x-27); bid < (DIM_BLOCK); bid+=24){
    	int  i = bid >>2;
    	int  j_base = bid & 3;
    	int  j = j_base * DIM_THREAD_BLOCK_X + threadIdx.x; 

    	//spin
		if(idx==0)    //spin ....wait for data ready
		{
			while( ( atomicAnd(&inflag_ey[i],1) == 0 )) 
			//while (flag_global_read(inflag,i+1 )==0)
			{
			//dummy_comp();
			}
		}

		if(idx==1)    //spin ....wait for data ready
		{
			while( ( atomicAnd(&inflag_hz[i],1) == 0 )) 
			//while (flag_global_read(inflag,i+1 )==0)
			{
			//dummy_comp();
			}
		}

    	__syncthreads();

    	//compute ey
		if ((i < NX) && (j < NY))
		{
			if (i == 0) 
			{
				ey[i * NY + j] = 0;
			}
			else
			{ 
				ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
			}
		}
		//make sure compute has been finished
		__syncthreads();
		__threadfence();
		//label the flag_ey
		if(idx==0)
		{
		atomicAdd(&flag_ey[i],1);
		}

    	}
	    

    }else if ((blockIdx.x>=51)&&(blockIdx.x<=75)){ //execute output_hz
    
    
    for (int bid = (blockIdx.x-51); bid < (DIM_BLOCK); bid+=25){
    	int  i = bid >> 2;
    	int  j_base = bid & 3;
    	int  j = j_base * DIM_THREAD_BLOCK_X + threadIdx.x;

    	//spin
		if(threadIdx.x==0)    //spin ....wait for data ready
		{
			while( ( atomicAnd(&inflag_hz[i],1) == 0 )) 
			//while (flag_global_read(inflag,i+1 )==0)
			{
			//dummy_comp();
			}
		}

		if(threadIdx.x==1)    //spin ....wait for data ready
		{
			//while( ( atomicAnd(&flag_ex[i],7) != 4 )) 
			while (flag_global_read(flag_ex,i )!=4)
			{
			//dummy_comp();
			}
		}

		if(threadIdx.x==2)    //spin ....wait for data ready
		{
			//while( (i< (NX-1) )&&( atomicAnd(&flag_ey[i+1],7) != 4 )) 
			while ( (i< (NX-1) )&&(flag_global_read(flag_ey,i+1 )!=4))
			{
			//dummy_comp();
			}
		}
    	__syncthreads();

    	//compute hz_output
		if ((i < NX) && (j < NY))
		{	
			hz_gpu_out[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
		}
		//make sure compute has been finished
		__syncthreads();
		__threadfence();
		//label the flag_ex
		if(threadIdx.x==0)
		{
		atomicAdd(&outflag[i],1);
		}

    	}
	
	
    }else if ((blockIdx.x>=76)&&(blockIdx.x<=79)){							//copy output hz_gpu_out to hz_outputFromGpu
  	 /*
    	const int idx = threadIdx.x;

	const int in_chunk_offset=(NY)*(IN_CHUNK_SIZE)/4;
	
	int chunk_offset=0;

	for (int i = 0 ; i < NUM_CHUNK; i++){

	//spin by check every 32 rows of hz
	if ( idx < IN_CHUNK_SIZE ){
		while(  flag_global_read(outflag,i * IN_CHUNK_SIZE+idx) != 4 )
		// while ( atomicAnd(&outflag[rid],3) == 0 )
        {
                             //dummy_comp();
        }
    }
	__syncthreads();

	chunk_offset= i * in_chunk_offset;

	for (int k = (chunk_offset+idx) ; k< (chunk_offset+in_chunk_offset); k+=OFFSET ){
			reinterpret_cast<double2*>(hz_gpu_out_host)[k] = reinterpret_cast<double2*>(hz_gpu_out)[k];
		}
		
	}
	*/

	const int idx = threadIdx.x;
	const int bidx = blockIdx.x-76;
	int rid;
	for (int i = bidx*1024+idx; i < (1024*4096); i+= 4096){
	rid= i >> 10;
		while (flag_global_read(outflag,rid)!=4){
		dummy_comp();
		}
		
	reinterpret_cast<double2*>(hz_gpu_out_host)[i] = reinterpret_cast<double2*>(hz_gpu_out)[i];
	
	}
	
    }

	

}

void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu)
{

	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;
	DATA_TYPE *hz_gpu_out;

	int *inflag_ex,*inflag_ey,*inflag_hz, *flag_ex, *flag_ey, *outflag;

	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&hz_gpu_out, sizeof(DATA_TYPE) * NX * NY);

	cudaMalloc((void **)&inflag_ex,  sizeof(int) * NX);
	cudaMalloc((void **)&inflag_ey,  sizeof(int) * (NX+1));
	cudaMalloc((void **)&inflag_hz,  sizeof(int) * NX);
	cudaMalloc((void **)&flag_ex,  sizeof(int) * NX);
	cudaMalloc((void **)&flag_ey,  sizeof(int) * (NX+1));	
	cudaMalloc((void **)&outflag,  sizeof(int) * NX);
	
	cudaMemset(inflag_ex, 0, sizeof(int) * NX);
	cudaMemset(inflag_ey, 0, sizeof(int) * (NX+1));
	cudaMemset(inflag_hz, 0, sizeof(int) * NX);	
	cudaMemset(flag_ex, 0, sizeof(int) * NX);	
	cudaMemset(flag_ey, 0, sizeof(int) * (NX+1));	
	cudaMemset(outflag, 0, sizeof(int) * NX);	

	cudaEvent_t start,stop;
    float elapsedTimeInMs = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(NUM_SM), (size_t)1 );
	
	void *kernelArgs[] = {
                (void *)&ex_gpu,             (void *)&ex,
                (void *)&ey_gpu,         (void *)&ey,
                (void *)&hz_gpu,             (void *)&hz,
                (void *)&hz_gpu_out,             (void *)&hz_outputFromGpu,
                (void *)&inflag_ex,             (void *)&inflag_ey,
                (void *)&inflag_hz,             (void *)&flag_ex,
                (void *)&flag_ey,        (void *)&outflag
        };	

	
	//fdtd_spin_kernel<<<grid,block>>>(ex_gpu,ex,ey_gpu,ey, hz_gpu,hz, hz_gpu_out,hz_outputFromGpu,inflag_ex,inflag_ey,inflag_hz,flag_ex,flag_ey,outflag);
	
	 cudaLaunchCooperativeKernel((void*)fdtd_spin_kernel, grid, block, kernelArgs,0, NULL);

	cudaThreadSynchronize();
	cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
    fprintf(stdout,"GPU RunTime= %.1f Ms \n",  elapsedTimeInMs);
	//copyout the outflag array and check
	/*
	int* flag_host;
	cudaHostAlloc((void **)&flag_host, sizeof(int)*NX, cudaHostAllocPortable);

	cudaMemcpy(flag_host,outflag,sizeof(int)*NY,cudaMemcpyDeviceToHost );
	for (int i = 0 ; i < NX ; i++){
	fprintf(stdout,"%d,",  flag_host[i]);
	}

	*/
//	cudaMemcpy(hz_outputFromGpu,hz_gpu_out,sizeof(DATA_TYPE)*NY*NX,cudaMemcpyDeviceToHost );
	//end checking	
	cudaFree(_fict_gpu);
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
	cudaFree(inflag_ex);
	cudaFree(inflag_ey);
	cudaFree(inflag_hz);
	cudaFree(flag_ex);
	cudaFree(flag_ey);
	cudaFree(outflag);

}


int main()
{
	double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;
	
	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));


	cudaHostAlloc((void **)&_fict_, sizeof(DATA_TYPE) * tmax, cudaHostAllocPortable);
	cudaHostAlloc((void **)&ex, sizeof(DATA_TYPE)*NX*(NY+1), cudaHostAllocPortable);
	cudaHostAlloc((void **)&ey, sizeof(DATA_TYPE)*NX*(NY+1), cudaHostAllocPortable);
	cudaHostAlloc((void **)&hz, sizeof(DATA_TYPE)*NX*NY, cudaHostAllocPortable);
	cudaHostAlloc((void **)&hz_outputFromGpu, sizeof(DATA_TYPE)*NX*NY, cudaHostAllocPortable);
	
	init_arrays(_fict_, ex, ey, hz);

	GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz, hz_outputFromGpu);
	
	/*
	t_start = rtclock();
	runFdtd(_fict_, ex, ey, hz);
	t_end = rtclock();
	
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(hz, hz_outputFromGpu);
	*/
	cudaFree(_fict_);
	cudaFree(ex);
	cudaFree(ey);
	cudaFree(hz);
	cudaFree(hz_outputFromGpu);

	return 0;
}

