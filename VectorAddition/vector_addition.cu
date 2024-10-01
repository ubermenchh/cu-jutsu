// Vector Addition in CUDA

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

__global__ void vector_add(int* a, int* b, int* c, int N) {
    // calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Range Check
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

void init_array(int* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
    }
}

// Verify the Vector Addition computation on CPU
void verify_solution(int* a, int* b, int* c, int N) {
    for (int i = 0; i < N; i++) {
        assert(a[i] + b[i] == c[i]);
    }
}

int main() {
    // Vector size (2^20: 1048576)
    int N = 1 << 20;
    size_t bytes = N * sizeof(bytes);
    
    // Allocate memory 
    int* a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    
    // Initialize the vectors
    init_array(a, N);
    init_array(b, N);
    
    //Initialize out CTA and Grid dimension
    int THREADS = 256;
    int BLOCKS = (N + THREADS - 1) / THREADS; // 4096 = (1048576 + 255) / 256

    // Call the kernel
    vector_add <<< BLOCKS, THREADS >>> (a, b, c, N); 
    cudaDeviceSynchronize();
    
    // verify solution
    verify_solution(a, b, c, N);
    printf("PROGRAM COMPLETED CORRECTLY\n");
    
    return 0;
}
