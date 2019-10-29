#include "reduction.h"

void reduce(int type, int size, int threads, int blocks, double *d_idata, double *d_odata);
double reductionMax(int size, double *inData, int maxThreads, int cpuFinalThreshold);
double reductionSum(int size, double *inData, int maxThreads, int cpuFinalThreshold);

int main()
{
	int N = 1 << 27;

	double *arrHost = new double[N];
	for (int i = 0; i < N; i++)
	{
		arrHost[i] = i;
	}

	double *arrDev;
	cudaMalloc(&arrDev, N * sizeof(double));

	cudaMemcpy(arrDev, arrHost, N * sizeof(double), cudaMemcpyHostToDevice);


	int maxThreads, cpuFinalThreshold;
	double max, sum;
	std::clock_t  startAll;
	double durationAll;
	startAll = std::clock();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float duration;


	cudaEventRecord(start);
	maxThreads = 128, cpuFinalThreshold = 32;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 128, cpuFinalThreshold = 64;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 128, cpuFinalThreshold = 128;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 128, cpuFinalThreshold = 256;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 256, cpuFinalThreshold = 32;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 256, cpuFinalThreshold = 64;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 256, cpuFinalThreshold = 128;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 256, cpuFinalThreshold = 256;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;

	cudaEventRecord(start);
	maxThreads = 512, cpuFinalThreshold = 32;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 512, cpuFinalThreshold = 64;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 512, cpuFinalThreshold = 128;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;


	cudaEventRecord(start);
	maxThreads = 512, cpuFinalThreshold = 256;
	max = reductionMax(N, arrDev, maxThreads, cpuFinalThreshold);
	sum = reductionSum(N, arrDev, maxThreads, cpuFinalThreshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&duration, start, stop);
	std::cout << maxThreads << "\t" << cpuFinalThreshold << "\t" << max << "\t" << sum << "\t" << duration << std::endl;

	durationAll = (std::clock() - startAll) / (double)CLOCKS_PER_SEC;
	std::cout << "Full duration:" << durationAll << std::endl;


	delete[] arrHost;
	cudaFree(arrDev);

	double a;
	std::cin >> a;

	return 0;
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

void getNumBlocksAndThreads(int n, int maxThreads, int &blocks, int &threads)
{
	threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
	blocks = (n + threads - 1) / threads;
}

double reductionMax(int size, double *inData, int maxThreads, int cpuFinalThreshold)
{
	//int cpuFinalThreshold = 256;
	//int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);

	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(MAXIMUM, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(MAXIMUM, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = outData_host[0];
	for (size_t i = 1; i < s; i++)
	{
		result = result > outData_host[i] ? result : outData_host[i];
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

double reductionSum(int size, double *inData, int maxThreads, int cpuFinalThreshold)
{

	//int cpuFinalThreshold = 256;
	//int maxThreads = 256;

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&inData_dev, blocks * sizeof(double));
	cudaMalloc((void **)&outData_dev, blocks * sizeof(double));

	reduce(SUMMATION, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();

	int s = blocks;
	while (s > cpuFinalThreshold)
	{
		cudaMemcpy(inData_dev, outData_dev, blocks * sizeof(double), cudaMemcpyDeviceToDevice);

		getNumBlocksAndThreads(s, maxThreads, blocks, threads);
		reduce(SUMMATION, s, threads, blocks, inData_dev, outData_dev);

		s = blocks;
	}

	double *outData_host;
	outData_host = (double*)malloc(s * sizeof(double));
	cudaMemcpy(outData_host, outData_dev, s * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0;
	for (size_t i = 0; i < s; i++)
	{
		result += outData_host[i];
	}

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}
