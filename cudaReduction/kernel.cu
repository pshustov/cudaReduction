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



__device__ double reduce_sum(cg::thread_group g, double *temp, double val)
{
	int lane = g.thread_rank();

	// Each iteration halves the number of active threads
	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i = g.size() / 2; i > 0; i /= 2)
	{
		temp[lane] = val;
		g.sync(); // wait for all threads to store
		if (lane < i) val += temp[lane + i];
		g.sync(); // wait for all threads to load
	}
	return val; // note: only thread 0 will return full sum
}

__device__ double thread_sum(double *input, int n)
{
	double sum = 0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < n / 4;
		i += blockDim.x * gridDim.x)
	{
		int4 in = ((int4*)input)[i];
		sum += in.x + in.y + in.z + in.w;
	}
	return sum;
}

__global__ void sum_kernel_block(double *sum, double *input, int n)
{
	double my_sum = thread_sum(input, n);

	extern __shared__ double temp[];
	auto g = cg::this_thread_block();
	double block_sum = reduce_sum(g, temp, my_sum);

	if (g.thread_rank() == 0) atomicAdd(sum, block_sum);

}


void reduceStrange(int type, int size, int threads, int blocks, double *d_idata, double *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int smemSize = threads * sizeof(double);

	switch (type)
	{
	case MAXIMUM:
		sum_kernel_block<<<dimGrid, dimBlock, smemSize>>>(d_odata, d_idata, size);
		break;
	case SUMMATION:
		sum_kernel_block<<<dimGrid, dimBlock, smemSize>>>(d_odata, d_idata, size);
		break;

	default:
		break;
	}
}


