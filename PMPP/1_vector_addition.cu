// Vector Addition in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


void init_vector(float* v, int N) {
    for (int i = 0; i < N; i++) {
        v[i] = (float)(rand() / RAND_MAX);
    }
}

__global__ void vector_add_gpu(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N)
        C[i] = A[i] + B[i];
}

void verify_result(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        assert(C[i] == A[i] + B[i]);
    }
}

int main() {
    const int N = 1 << 20;
    
    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));
    
    init_vector(A_h, N);
    init_vector(B_h, N);
    
    float* A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(float) * N);
    cudaMalloc(&B_d, sizeof(float) * N);
    cudaMalloc(&C_d, sizeof(float) * N);
    
    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    clock_t start_time = clock();
    vector_add_gpu <<< 32, 32 >>> (A_d, B_d, C_d, N);
    clock_t end_time = clock();
    
    cudaMemcpy(C_h, C_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    verify_result(A_h, B_h, C_h, N);
    printf("SUCCESS!!!\n");
    
    printf("Total time taken (GPU): %f\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    
    cudaFree(&A_d); cudaFree(&B_d); cudaFree(&C_d);
    free(A_h); free(B_h); free(C_h);
    return 0;
}
