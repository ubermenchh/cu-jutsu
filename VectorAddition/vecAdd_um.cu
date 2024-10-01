// Vector Addition in CUDA using Unified Memory
//
// Blog: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
// Unified Memory: a single memory address space accessible from any processor
// in a system. (Data that can be read or written from either CPUs or GPUs)
// `cudaError_t cudaMallocManaged(void** ptr, size_t size);`

#include <stdio.h>
#include <assert.h>

__global__ void vector_add(int* a, int* b, int* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID
    
    if (tid < N) // Boundary check
        c[tid] = a[tid] + b[tid];
}

int main() {
    const int N = 1 << 16; // 2^16 -> 65536
    size_t bytes = N * sizeof(int);
    
    int* a, *b, *c;
    
    // cudaMallocManaged allocates memory that will be managed by the 
    // unified memory system
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    
    // init vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
    
    int BLOCK_SIZE = 1 << 10; // 1024 threads per CTA
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // calling the kernel
    vector_add <<< GRID_SIZE, BLOCK_SIZE >>> (a, b, c, N);
    
    // blocks all the threads from further tasks until all the previous threads our done
    cudaDeviceSynchronize();
    
    // verifying the result
    for (int i = 0; i < N; i++) 
        assert(c[i] == a[i] + b[i]);
    
    cudaFree(a); cudaFree(b); cudaFree(c);
    
    printf("SUCCESS!!!");
    return 0;
}
