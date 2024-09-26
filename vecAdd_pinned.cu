// Vector Addition in CUDA using Pinned Memory
//
// Blog: https://giahuy04.medium.com/pinned-memory-5d408b72241d
        
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void vector_add(int* a, int* b, int* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID
    
    if (tid < N) // Boundary check
        c[tid] = a[tid] + b[tid];
}

void verify_result(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
    const int N = 1 << 26;
    size_t bytes = sizeof(int) * N;
    
    int* h_a, *h_b, *h_c;
    
    // allocate pinned memory
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    
    // init vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }
    
    // allocate memory on the device
    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);
    
    int THREADS = 1 << 10; // 1024 threads
    int BLOCKS = (N + THREADS - 1) / THREADS;
    
    vector_add <<< BLOCKS, THREADS >>> (d_a, d_b, d_c, N);
    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    verify_result(h_a, h_b, h_c, N);
    
    // free pinned memory
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    
    // free device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    printf("SUCCESS!!!");
    return 0;
}
