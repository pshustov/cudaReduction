#include "reduction.h"

void reduce(int type, int size, int threads, int blocks, double *d_idata, double *d_odata);
void reduceStrange(int type, int size, int threads, int blocks, double *d_idata, double *d_odata);
double reductionMax(int size, double *inData, int maxThreads, int s);
double reductionSum(int size, double *inData, int maxThreads, int s);

int main()
{
	int N = 1 << 29;

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


double reductionMax(int size, double *inData, int maxThreads, int s)
{
	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&outData_dev, sizeof(double));

	reduceStrange(SUMMATION, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();


	double *outData_host;
	outData_host = (double*)malloc(sizeof(double));
	cudaMemcpy(outData_host, outData_dev, sizeof(double), cudaMemcpyDeviceToHost);

	double result = outData_host[0];

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}

double reductionSum(int size, double *inData, int maxThreads, int s)
{

	if (!isPow2(size)) throw;

	int blocks = 0, threads = 0;
	getNumBlocksAndThreads(size, maxThreads, blocks, threads);


	double *inData_dev = NULL;
	double *outData_dev = NULL;

	cudaMalloc((void **)&outData_dev, sizeof(double));

	reduceStrange(SUMMATION, size, threads, blocks, inData, outData_dev);
	cudaDeviceSynchronize();


	double *outData_host;
	outData_host = (double*)malloc(sizeof(double));
	cudaMemcpy(outData_host, outData_dev, sizeof(double), cudaMemcpyDeviceToHost);

	double result = outData_host[0];

	cudaFree(inData_dev);
	cudaFree(outData_dev);
	free(outData_host);

	return result;
}
