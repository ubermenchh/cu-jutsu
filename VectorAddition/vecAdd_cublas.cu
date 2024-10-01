// Vector Addition in CUDA using cuBLAS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1000000 // Vector size

// Function to initialize a vector with random float values
void initVector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = rand() / (float)RAND_MAX;
    }
}

// Function to verify the result
void verifyResult(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > 1e-5) {
            printf("Error at position %d: Expected %f, got %f\n", i, expected, c[i]);
            return;
        }
    }
    printf("Result verified successfully!\n");
}

int main() {
    float* h_a, *h_b, *h_c;
    float* d_a, *d_b, *d_c;
    int size = N * sizeof(float);
    const float alpha = 1.0f; // Scalar for Addition
    
    // allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // init vectors 
    initVector(h_a, N);
    initVector(h_b, N);
    
    // allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // perform vector addition 
    // SAXPY: Single Precision A*X plus Y
    // https://docs.nvidia.com/cuda/cublas/index.html?highlight=saxpy#cublas-t-axpy
    cublasSaxpy(handle, N, &alpha, d_a, 1, d_b, 1);
    
    // copy back result to host
    cudaMemcpy(h_c, d_b, size, cudaMemcpyDeviceToHost);
    
    // verify the result
    verifyResult(h_a, h_b, h_c, N);
    
    // clean up
    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
