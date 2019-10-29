#include "reduction.h"

#define gpuErrchk(val) \
    cudaErrorCheck(val, __FILE__, __LINE__, true)
void cudaErrorCheck(cudaError_t err, char* file, int line, bool abort)
{
	if (err != cudaSuccess)
	{
		printf("%s %s %d\n", cudaGetErrorString(err), file, line);
		if (abort) exit(-1);
	}
}

namespace cg = cooperative_groups;
typedef double(*pfunc)(double x, double y);

__device__ double getMax(double x, double y) {
	return (x > y) ? x : y;
}

__device__ double getSum(double x, double y) {
	return x + y;
}

__device__ pfunc dev_getMax = getMax;
__device__ pfunc dev_getSum = getSum;


__global__ void reduceKernel(pfunc f, double *g_idata, double *g_odata, unsigned int n)
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
			sdata[tid] = f(sdata[tid], sdata[tid + s]);
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

	pfunc host_function_ptr;
	switch (type)
	{
	case MAXIMUM:
		gpuErrchk(cudaMemcpyFromSymbol(&host_function_ptr, dev_getMax, sizeof(pfunc)));
		reduceKernel<<<dimGrid, dimBlock, smemSize>>>(host_function_ptr, d_idata, d_odata, size);
		break;
	case SUMMATION:
		gpuErrchk(cudaMemcpyFromSymbol(&host_function_ptr, dev_getSum, sizeof(pfunc)));
		reduceKernel<<<dimGrid, dimBlock, smemSize>>>(host_function_ptr, d_idata, d_odata, size);
		break;

	default:
		break;
	}
}



