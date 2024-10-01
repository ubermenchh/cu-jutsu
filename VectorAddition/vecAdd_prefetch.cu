// Vector Addition in CUDA using Unified Memory
//
// Blog: https://giahuy04.medium.com/unified-memory-81bb7c0f0270

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
    
    // Device ID for prefetching calls
    int id = cudaGetDevice(&id);
    
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cudaMemAdviseSetPreferredLocation#data-usage-hints
    //
    // This hint tells the system that migrating this memory region away from 
    // its preferred location is undesired, by setting the preferred location 
    // for the data to be the physical memory belonging to device. Passing in 
    // a value of cudaCpuDeviceId for device sets the preferred location as CPU 
    // memory
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    
    // cudaMemPrefetchAsync: This function is used to prefetch data from a memory 
    // region on the host or device to another region on the device or host. It 
    // allows explicit control of the data prefetching process to optimize 
    // performance and efficient data access on the GPU.
    cudaMemPrefetchAsync(c, bytes, id);  // prefetch to device (GPU)
    
    // init vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
    
    // suggesting the memory region will be read frequently
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    // prefetch to device (GPU)
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    
    // Threads per CTA (1024 threads per CTA)
    int BLOCK_SIZE = 1 << 10;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    vector_add <<< GRID_SIZE, BLOCK_SIZE >>> (a, b, c, N);
    
    cudaDeviceSynchronize();
    
    // prefetch to the host (CPU)
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
    
    for (int i = 0; i < N; i++) 
        assert(c[i] == a[i] + b[i]);
    
    cudaFree(a); cudaFree(b); cudaFree(c);
    
    printf("SUCCESS!!!");
    return 0;
}
