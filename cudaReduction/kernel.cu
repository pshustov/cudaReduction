#include "reduction.h"

namespace cg = cooperative_groups;

__device__ double getMax(double x, double y) {
	return (x > y) ? x : y;
}

__device__ double getSum(double x, double y) {
	return x + y;
}

__global__ void reduceKernelMax2(double *g_idata, double *g_odata, unsigned int n)
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
__global__ void reduceKernelMax3(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getMax(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = result = getMax(result, sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = result;
}

__global__ void reduceKernelSum2(double *g_idata, double *g_odata, unsigned int n)
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
__global__ void reduceKernelSum3(double *g_idata, double *g_odata, unsigned int n)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	double result = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		result = getSum(result, g_idata[i + blockDim.x]);

	sdata[tid] = result;
	cg::sync(cta);

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = result = getSum(result, sdata[tid + s]);
		}
		cg::sync(cta);
	}

	if (tid == 0) g_odata[blockIdx.x] = result;
}

void reduce(int wichKernel, int type, int size, int threads, int blocks, double *d_idata, double *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = threads * sizeof(double);

	switch (type)
	{
	case MAXIMUM:
		switch (wichKernel)
		{
		case 2:
			reduceKernelMax2 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		case 3:
			reduceKernelMax3 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		default:
			break;
		}
		break;
	case SUMMATION:
		switch (wichKernel)
		{
		case 2:
			reduceKernelSum2 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		case 3:
			reduceKernelSum3 << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
			break;
		default:
			break;
		}
		break;

	default:
		break;
	}
}



