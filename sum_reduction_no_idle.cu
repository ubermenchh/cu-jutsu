// Optimized Sum Reduction Algorithm in CUDA using sequential addressing and no idle threads

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__device__ void warp_reduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sum_reduction(int* v, int* v_r) {
    // allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];
    
    // calculate thread ID
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    
    // load elements into shared memory
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();
    
    // iterate of log base 2 the block dimension
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        // each thread does work unless index goes off the block
        if (tid < s) 
            partial_sum[tid] += partial_sum[tid + s];
        
        __syncthreads();
    }
    
    if (tid < 32) 
        warp_reduce(partial_sum, tid);
    
    // let the thread 0 for this block write it's result to main memory
    if (tid == 0) 
        v_r[blockIdx.x] = partial_sum[0];
}

void initialize_vector(int *v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;//rand() % 10;
	}
}

int main() {
    // Vector size
    int n = 1 << 16;
    size_t bytes = n * sizeof(int);

    // Original vector and result vector
    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    // Allocate memory
    h_v = (int*)malloc(bytes);
    h_v_r = (int*)malloc(bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    // Initialize vector
    initialize_vector(h_v, n);

    // Copy to device
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    // TB Size
    int TB_SIZE = SIZE;

    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = n / TB_SIZE / 2;

    // Call kernel
    sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

    sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

    // Copy to host;
    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    // Print the result
    //printf("Accumulated result is %d \n", h_v_r[0]);
    //scanf("Press enter to continue: ");
    assert(h_v_r[0] == 65536);

    printf("COMPLETED SUCCESSFULLY\n");

    return 0;
}
