// Matrix Multiplication in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define TILE_DIM 32
#define COARSE_FACTOR 4

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
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    
    float sum[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        sum[c] = 0.0f;
    }
    
    for (int tile = 0; tile < N / TILE_DIM; tile++) {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
        
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = col_start + c * COARSE_FACTOR;
            
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
            __syncthreads();
            
            for (int i = 0; i < TILE_DIM; i++) {
                sum[c] += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = col_start + c * TILE_DIM;
        C[row * N + col] = sum[c];
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
    
    dim3 num_threads(TILE_DIM, TILE_DIM);
    dim3 num_blocks((N + num_threads.x - 1) / num_threads.x / COARSE_FACTOR, (N + num_threads.y  - 1) / num_threads.y);
    clock_t start = clock();
    matmul_gpu <<< num_threads, num_blocks >>> (A_d, B_d, C_d, N);
    clock_t end = clock();
    
    printf("Time Taken: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    cudaMemcpy(C_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    verify_result(A_h, B_h, C_h, N);
    printf("SUCCESS!!!\n");
    
    cudaFree(&A_d); cudaFree(&B_d); cudaFree(&C_d);
    free(A_h); free(B_h); free(C_h);
    return 0;
}
