// Matrix Multiplication for Rectangular Matrices in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

void init_matrix(float* a, int m, int n) {
    for (int i = 0; i < m*n; i++) {
        a[i] = rand() / RAND_MAX;
    }
}

void verify_result(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += A[i * K + k] * B[k * N + j];
            }
            assert(temp == C[i * N + j]);
        }
    }
}

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float temp = 0.0f;
        for (int k = 0; k < K; k++) {
            temp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = temp;
    }
}

int main() {
    const int M = 1 << 10;
    const int N = 1 << 11;
    const int K = 1 << 12;
    const int bytes_a = M * K * sizeof(float);
    const int bytes_b = K * N * sizeof(float);
    const int bytes_c = M * N * sizeof(float);
    
    float* A_h = (float*)malloc(bytes_a);
    float* B_h = (float*)malloc(bytes_b);
    float* C_h = (float*)malloc(bytes_c);

    init_matrix(A_h, M, N);
    init_matrix(B_h, M, N);
    
    float* A_d, *B_d, *C_d;
    cudaMalloc(&A_d, bytes_a);
    cudaMalloc(&B_d, bytes_b);
    cudaMalloc(&C_d, bytes_c);

    cudaMemcpy(A_d, A_h, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes_b, cudaMemcpyHostToDevice);
    
    dim3 num_threads(16, 16);
    dim3 num_blocks((N + num_threads.x  - 1) / num_threads.x, (N + num_threads.y  - 1) / num_threads.y);
    clock_t start_time = clock();
    matmul <<< num_threads, num_blocks >>> (A_d, B_d, C_d, M, N, K);
    clock_t end_time = clock();
    
    printf("Total Time Taken: %f\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    
    cudaMemcpy(C_h, C_d, bytes_c, cudaMemcpyDeviceToHost);
    verify_result(A_h, B_h, C_h, M, N, K);
    printf("SUCCESS!!!\n");
    
    cudaFree(&A_d); cudaFree(&B_d); cudaFree(&C_d);
    free(A_h); free(B_h); free(C_h);
    return 0;
}
