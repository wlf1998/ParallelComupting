#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <device_launch_parameters.h>

// 矩阵大小
#define BLOCK_SIZE 16
#define WA (3 * BLOCK_SIZE) // Matrix A width
#define HA (5 * BLOCK_SIZE) // Matrix A height
#define WB (8 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

// CPU 上运行的计算函数
void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	for (unsigned int i = 0; i < hA; ++i)
		for (unsigned int j = 0; j < wB; ++j)
		{
			double sum = 0;
			for (unsigned int k = 0; k < wA; ++k)
			{
				double a = A[i * wA + k];
				double b = B[k * wB + j];
				sum += a * b;
			}
			C[i * wB + j] = (float)sum;
		}
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float* data1, float* data2, int width, int height)
{
	int i, j, k;
	int error_count = 0;
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			k = j * width + i;
			if (data1[k] != data2[k])
			{
				printf("错误位置(%d, %d) CPU = %4.4f, GPU = %4.4f \n", i, j, data1[k], data2[k]);
				error_count++;
			}
		}
	}
	printf("总共有 %d 处错误。\n", error_count);
}

__device__  float* GetSubMatrix(float* matrix, int m, int index, int width)
{
	return  matrix + width * BLOCK_SIZE * index + BLOCK_SIZE * m;
}

__global__ void matrixMul(float* C, float* A, float* B, int wA, int wB)
{
	// Declaration of the shared memory array As used to
   // store the sub-matrix of A
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

	// Declaration of the shared memory array Bs used to
	// store the sub-matrix of B
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int m = 0; m < wA / BLOCK_SIZE; m++)
	{
		//get the address of submatrixA
		//float *subA=A+wA*BLOCK_SIZE*by+BLOCK_SIZE*m;
		float* subA = GetSubMatrix(A, m, by, wA);
		//get the address of submatrixB
		//float *subB=B+wB*BLOCK_SIZE*m+BLOCK_SIZE*bx;
		float* subB = GetSubMatrix(B, bx, m, wB);
		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = *(subA + wA * ty + tx);
		Bs[ty][tx] = *(subB + wB * ty + tx);

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	//float *subC = C+wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	float* subC = GetSubMatrix(C, bx, by, wB);
	*(subC + wB * ty + tx) = Csub;
}

int main(int argc, char** argv)
{
	// set seed for rand()
	srand((unsigned)time(NULL));

	// allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);
	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);
	printf("问题规模：A(%d, %d), B(%d, %d)\n", WA, HA, WB, HB);

	// initialize host memory
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	// allocate device memory
	float* d_A;
	cudaMalloc((void**)& d_A, mem_size_A);
	float* d_B;
	cudaMalloc((void**)& d_B, mem_size_B);

	// copy host memory to device
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// allocate device memory for result
	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* d_C;
	cudaMalloc((void**)& d_C, mem_size_C);

	// allocate host memory for the result
	float* h_C = (float*)malloc(mem_size_C);

	// create and start timer
	StopWatchInterface* timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	// setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(WC / threads.x, HC / threads.y);

	// execute the kernel
	matrixMul <<< grid, threads >>> (d_C, d_A, d_B, WA, WB);
	cudaThreadSynchronize();

	// stop and destroy timer
	sdkStopTimer(&timer);
	float time1 = sdkGetTimerValue(&timer);
	printf("GPU 处理时间：%f ms\n", time1);
	sdkDeleteTimer(&timer);

	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// compute reference solution
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	float* reference = (float*)malloc(mem_size_C);
	computeGold(reference, h_A, h_B, HA, WA, WB);
	sdkStopTimer(&timer);
	float time2 = sdkGetTimerValue(&timer);
	printf("CPU 处理时间：%f ms\n", time2);
	sdkDeleteTimer(&timer);
	printf("加速比：%f\n", time2 / time1);

	// check result
	bool res = sdkCompareL2fe(reference, h_C, size_C, 1e-6f);
	printf("结果比较：%s\n", (true == res) ? "相同" : "不相同");
	if (res != true) printDiff(reference, h_C, WC, HC);

	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(reference);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaThreadExit();
	exit((true == res) ? EXIT_SUCCESS : EXIT_FAILURE);
}

