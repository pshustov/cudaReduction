#include "reduction.h"

namespace cg = cooperative_groups;

__device__ double getMax(double x, double y) {
	return (x > y) ? x : y;
}

__device__ double getSum(double x, double y) {
	return x + y;
}


__global__ void reduceKernelMax(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = getMax(sdata[tid], sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceKernelSum(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = getSum(sdata[tid], sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce(int type, int size, int threads, int blocks, double *d_idata, double *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = threads * sizeof(double);

	switch (type)
	{
	case MAXIMUM:
		reduceKernelMax<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
		break;
	case SUMMATION:
		reduceKernelSum<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
		break;

	default:
		break;
	}
}



