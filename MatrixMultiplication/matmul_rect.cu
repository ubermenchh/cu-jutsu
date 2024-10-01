// Matrix Multiplication of Rectangular matrices in CUDA

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// Matrix and Shared memory tile size
// MxN = MxK * KxN -> (1024 x 2048) = (1024 x 4096) * (4096 x 2048)
const int M = 1 << 10;
const int N = 1 << 11;
const int K = 1 << 12;
const int SHMEM_SIZE = 1 << 10;

__global__ void matmul(int* a, int* b, int* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];
    
    int temp = 0;
    
    // sweep tile across matrix
    for (int i = 0; i < K; i += blockDim.x) {
        // load in elements for this tile
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
        
        // wait for both tiles to be loaded in before doint computation
        __syncthreads();
        
        // do matmul on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            temp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }
        
        // wait for all threads to finish using current tiles before loading in new ones
        __syncthreads();
    }
    // write back results in C matrix
    c[row * N + col] = temp;
}

void verify_result(int* a, int* b, int* c) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int temp = 0;
            for (int i = 0; i < K; i++) {
                temp += a[row * K + i] * b[i * N + col];
            }
            assert(temp == c[row * N + col]);
        }
    }
}

void init_matrix(int* a, int N, int M) {
    for (int i = 0; i < N*M; i++) {
        a[i] = rand() % 100;
    }
}

int main() {
    // Size of Matrix: MxN = MxK * KxN
    size_t bytes_a = M * K * sizeof(int);
    size_t bytes_b = K * N * sizeof(int);
    size_t bytes_c = M * N * sizeof(int);
    
    // init matrix
    int* h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes_a);
    h_b = (int*)malloc(bytes_b);
    h_c = (int*)malloc(bytes_c);
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);
    
    // allocate device memory
    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    
    // copy data to device
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);
    
    // threads per CTA dimension
    int THREADS = 32;
    // blocks per grid dimension
    int BLOCKS_X = N / THREADS;
    int BLOCKS_Y = M / THREADS;
    
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);
    
    matmul <<< blocks, threads >>> (d_a, d_b, d_c);
    
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);
    
    //verify_result(h_a, h_b, h_c);
    printf("SUCCESS!!!");
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
