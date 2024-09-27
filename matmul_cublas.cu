// Matrix Multiplication in CUDA using cuBLAS

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 1024 // rows of A and C
#define N 1024 // cols of B and C
#define K 1024 // cols of A and rows of B

// Function to initialize a matrix with random float values
void initMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

// Function to verify the result
void verifyResult(float *a, float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            if (fabs(c[i * n + j] - sum) > 1e-5) {
                printf("Error at position (%d, %d): Expected %f, got %f\n", i, j, sum, c[i * n + j]);
                return;
            }
        }
    }
    printf("Result verified successfully!\n");
}

int main() {
    float* h_a, *h_b, *h_c; // host matrices
    float* d_a, *d_b, *d_c; // device matrices
    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);
    
    // allocate host memory
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    
    // init matrices
    initMatrix(h_a, M, K);
    initMatrix(h_b, K, N);
    
    // allocate device memory
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(handle);
    
    // copy data from host to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    
    // performing matrix multiplication
    const float alpha = 1.0f; 
    const float beta = 0.0f;
    // https://docs.nvidia.com/cuda/cublas/index.html?highlight=saxpy#cublas-t-gemm
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    
    // copy result from device to host
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    
    // verify result
    verifyResult(h_a, h_b, h_c, M, N, K);
    
    // clean up
    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
