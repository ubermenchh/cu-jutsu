// Matrix Multiplication using CUDA

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

__global__ void matmul(int* a, int* b, int* c, int N) {
    // calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // boundary check
    if (row < N && col < N) {
        int temp = 0;
        for (int i = 0; i < N; i++) {
            temp += a[row * N + i] * b[i * N + col];
        }
        // write back the result
        c[row * N + col] = temp;
    }
}

void init_matrix(int* m, int N) {
    // Initializes a square matrix with random numbers b/w 0-100
    for (int i = 0; i < N; i++) {
        m[i] = rand() % 100;
    }
}

void verify_solution(int* a, int* b, int* c, int N) {
    int temp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int k = 0; k < N; k++) {
                temp += a[i * N + k] * b[k * N + j];
            }
            assert(temp == c[i * N + j]);
        }
    }
}

int main() {
    // Square Matrix Dimension (2^10x2^10 default)
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);
    
    // Allocate memory 
    int* a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    
    // Initialize Matrices
    init_matrix(a, N);
    init_matrix(b, N);
    
    // Set out CTA(Cooperative Thread Array aka Thread Block) and Grid dims
    int threads = 16;
    int blocks = (N + threads - 1) / threads;
    
    // Setup our kernel launch params 
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    
    // launch the kernel
    matmul <<< BLOCKS, THREADS >>> (a, b, c, N);
    cudaDeviceSynchronize();
    
    // verify solution
    verify_solution(a, b, c, N);
    printf("SUCCESS!!!");
    
    return 0;
}
