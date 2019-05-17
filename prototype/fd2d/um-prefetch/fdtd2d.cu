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

#define NUM_STREAMS 4
#define NUM_CHUNK 16
#define CHUNK_SIZE NY/NUM_CHUNK //4096/4= 1024


/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



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
			ey[0*NY + j] = _fict_[0];
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



__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int i_base)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i_base+blockIdx.y ;

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
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int i_base)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i_base + blockIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int i_base,DATA_TYPE *hz_out)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = i_base+blockIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz_out[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu)
{
	/*
	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;
	DATA_TYPE *hz_gpu_out;
	
	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&hz_gpu_out, sizeof(DATA_TYPE) * NX * NY);
	*/

	cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
	
	cudaStream_t streams[NUM_STREAMS];
        for (int i=0; i< NUM_STREAMS; i++)
                cudaStreamCreate(&(streams[i]));


	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)(CHUNK_SIZE) );
	
	//cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice);
	//cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice);
	//cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);

	for(int t = 0; t < NUM_CHUNK; t++)
	{

	int i_base;
	i_base=CHUNK_SIZE*t;
	
	cudaMemPrefetchAsync(ex+i_base*(NY+1), sizeof(DATA_TYPE) * CHUNK_SIZE * (NY + 1),0,streams[t % NUM_STREAMS]);
	cudaMemPrefetchAsync(ey+i_base*NY, sizeof(DATA_TYPE) * CHUNK_SIZE * NY, 0,streams[t % NUM_STREAMS]);
	cudaMemPrefetchAsync(hz+i_base*NY, sizeof(DATA_TYPE) * CHUNK_SIZE * NY, 0,streams[t % NUM_STREAMS]);
	

	fdtd_step1_kernel<<<grid,block,0,streams[t % NUM_STREAMS]>>>(_fict_, ex, ey, hz, i_base);
	fdtd_step2_kernel<<<grid,block,0,streams[t % NUM_STREAMS]>>>(ex, ey, hz, i_base);
	
	if (t>0){
	fdtd_step3_kernel<<<grid,block,0,streams[t % NUM_STREAMS]>>>(ex, ey, hz, i_base-CHUNK_SIZE,hz_outputFromGpu);

	cudaMemPrefetchAsync(hz_outputFromGpu+(i_base-CHUNK_SIZE)*NY,  sizeof(DATA_TYPE) * CHUNK_SIZE * NY, cudaCpuDeviceId,streams[t % NUM_STREAMS]);	
	
	}else{
	
	cudaMemPrefetchAsync(hz_outputFromGpu+(i_base-CHUNK_SIZE)*NY, sizeof(DATA_TYPE)*(CHUNK_SIZE-1)*NY, cudaCpuDeviceId,streams[t % NUM_STREAMS]);	
	}
	
	}

	//fdtd_step3_kernel<<<grid,block,0,0>>>(ex_gpu, ey_gpu, hz_gpu, NX-CHUNK_SIZE,hz_gpu_out);

	fdtd_step3_kernel<<<grid,block,0,streams[(NUM_CHUNK-1) % NUM_STREAMS]>>>(ex, ey, hz, NX-CHUNK_SIZE,hz_outputFromGpu);
	cudaMemPrefetchAsync(hz_outputFromGpu+(NX-CHUNK_SIZE)*NY,  sizeof(DATA_TYPE) * CHUNK_SIZE * NY, cudaCpuDeviceId,streams[(NUM_CHUNK-1) % NUM_STREAMS]);	
	
	//cudaMemcpyAsync(hz_outputFromGpu+(NX-CHUNK_SIZE)*NY, hz_gpu_out+(NX-CHUNK_SIZE)*NY, sizeof(DATA_TYPE) * CHUNK_SIZE * NY, cudaMemcpyDeviceToHost,0);	
	//cudaMemcpy(hz_outputFromGpu, hz_gpu_out, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaThreadSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        fprintf(stdout,"GPU RunTime= %.1f Ms \n",  elapsedTimeInMs);
		/*
	cudaFree(_fict_gpu);
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
*/
}


int main()
{
	double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;
	/*
	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	*/

	cudaMallocManaged((void **)&_fict_, sizeof(DATA_TYPE) * tmax);
	cudaMallocManaged((void **)&ex, sizeof(DATA_TYPE)*NX*(NY+1));
	cudaMallocManaged((void **)&ey, sizeof(DATA_TYPE)*NX*(NY+1));
	cudaMallocManaged((void **)&hz, sizeof(DATA_TYPE)*NX*NY);
	cudaMallocManaged((void **)&hz_outputFromGpu, sizeof(DATA_TYPE)*NX*NY);
	
	init_arrays(_fict_, ex, ey, hz);

	GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz, hz_outputFromGpu);
	
		
	t_start = rtclock();
	runFdtd(_fict_, ex, ey, hz);
	t_end = rtclock();
	
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(hz, hz_outputFromGpu);
	
	cudaFree(_fict_);
	cudaFree(ex);
	cudaFree(ey);
	cudaFree(hz);
	cudaFree(hz_outputFromGpu);

	return 0;
}

