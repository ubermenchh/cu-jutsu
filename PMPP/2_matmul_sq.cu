// Matrix Multiplication in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

void init_matrix(float* m, int N) {
    for (int i = 0; i < N*N; i++) {
        m[i] = (float)(rand() / RAND_MAX);
    }
}

void matmul_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        float temp = 0;
        for (int j = 0; j < N; j++) {
            temp += A[i] * B[i];
        }
        C[i] = temp;
    }
}

void verify_result(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0;
            for (int k = 0; k < N; k++) {
                temp += A[i * N + k] * B[k * N + j];
            }
            assert(C[i * N + j] == temp);
        }
    }
}

__global__ void matmul_gpu(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float temp = 0;
        for (int i = 0; i < N; i++) {
            temp += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = temp;
    }
}

int main() {
    const int N = 1 << 10;
    
    float* A_h = (float*)malloc(N*N*sizeof(float));
    float* B_h = (float*)malloc(N*N*sizeof(float));
    float* C_h = (float*)malloc(N*N*sizeof(float));
    
    init_matrix(A_h, N);
    init_matrix(B_h, N);
    
    float* A_d, *B_d, *C_d;
    cudaMalloc(&A_d, N*N*sizeof(float));
    cudaMalloc(&B_d, N*N*sizeof(float));
    cudaMalloc(&C_d, N*N*sizeof(float));
    
    cudaMemcpy(A_d, A_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 num_threads(16, 16);
    dim3 num_blocks((N + num_threads.x  - 1) / num_threads.x, (N + num_threads.y  - 1) / num_threads.y);
    clock_t start = clock();
    matmul_gpu <<< num_threads, num_blocks >>> (A_d, B_d, C_d, N);
    clock_t end = clock();
    
    printf("Time Taken: %f\n", (double)(start - end) / CLOCKS_PER_SEC);
    cudaMemcpy(C_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(A_h, B_h, C_h, N);
    printf("SUCCESS!!!\n");
    
    cudaFree(&A_d); cudaFree(&B_d); cudaFree(&C_d);
    free(A_h); free(B_h); free(C_h);
    return 0;
}
