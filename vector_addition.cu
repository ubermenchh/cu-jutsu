// Vector Addition from PPMP

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) 
        C[i] = A[i] + B[i];
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
    int size = n * sizeof(float); // size of the memory to be allocated
    float *d_A, *d_B, *d_C; // device memory pointers

    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, size);

    vecAddKernel <<< ceil(n / 256.0), 256 >>> (d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(void) {
    int N = 1000; // num of elements
    size_t size = N * sizeof(float);

    // allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / float(RAND_MAX);
        h_B[i] = rand() / float(RAND_MAX);
    }

    clock_t start_time = clock();
    vecAdd(h_A, h_B, h_C, N);
    clock_t end_time = clock();

    printf("Time Spent: %f secs\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    for (int i = 0; i < 10; i++) 
        printf("h_C[%d] = %f\n", i, h_C[i]);

    free(h_A); free(h_B); free(h_C);
    return 0;
}
