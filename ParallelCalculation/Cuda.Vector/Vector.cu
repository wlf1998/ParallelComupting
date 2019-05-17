#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

#define BLOCKSIZE 16

__global__ void Add(const int* a, const int* b, int* c, int n) 
{
	int i = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
	if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) 
{
	if (2 != argc) return -1;
	int n = atoi(argv[1]);
	if (n <= 0) return -1;
	printf("�����ģ��%d\n", n);

	int* host_a = (int*)malloc(sizeof(int) * n);
	int* host_b = (int*)malloc(sizeof(int) * n);
	int* host_c = (int*)malloc(sizeof(int) * n);
	int* host_c2 = (int*)malloc(sizeof(int) * n);

	srand((unsigned)time(NULL));

	for (int i = 0; i < n; i++) 
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	// GPU ���㲿��

	int* device_a, * device_b, * device_c;
	cudaMalloc((void**)& device_a, sizeof(int) * n);
	cudaMalloc((void**)& device_b, sizeof(int) * n);
	cudaMalloc((void**)& device_c, sizeof(int) * n);

	cudaMemcpy(device_a, host_a, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, sizeof(int) * n, cudaMemcpyHostToDevice);

	int gridsize = (int)ceil(sqrt(ceil(n / (BLOCKSIZE * BLOCKSIZE))));

	StopWatchInterface* timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(gridsize, gridsize);

	Add <<<dimGrid, dimBlock>>> (device_a, device_b, device_c, n);
	cudaThreadSynchronize();

	cudaMemcpy(host_c, device_c, sizeof(int) * n, cudaMemcpyDeviceToHost);

	sdkStopTimer(&timer);
	float time1 = sdkGetTimerValue(&timer);
	printf("GPU ����ʱ�䣺%f ms\n", time1);
	sdkDeleteTimer(&timer);

	// CPU ���㲿��

	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < n; i++) 
	{
		host_c2[i] = host_a[i] + host_b[i];
	}

	sdkStopTimer(&timer);
	float time2 = sdkGetTimerValue(&timer);
	printf("CPU ����ʱ�䣺%f ms\n", time2);
	sdkDeleteTimer(&timer);
	printf("���ٱȣ�%f\n", time2 / time1);

	// �����

	int errorNum = 0;
	for (int i = 0; i < n; i++) 
	{
		if (host_c[i] != host_c2[i]) 
		{
			errorNum++;
			printf("������������ %d ����CPU = %d��GPU = %d\n", i, host_c2[i], host_c[i]);
		}
	}
	if (errorNum == 0) 
		printf("�޴���\n");
	else
		printf("���� %d ������\n", errorNum);

	free(host_a);
	free(host_b);
	free(host_c);
	free(host_c2);

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	return 0;
}
