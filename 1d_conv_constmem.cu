// 1D Convolution implemented in CUDA using Constant Memory
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#constant-memory
// https://leimao.github.io/blog/CUDA-Constant-Memory/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MASK_LEN 7

// allocate space for the mask in constant memory
__constant__ int mask[MASK_LEN];

__global__ void conv1d(int* array, int* result, int n) {
    /* 1D Convolution Kernel */
    // Arguments:
    //      array -> padded array
    //     result -> resulting array
    //          n -> num of elements in array
    
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate radius of mask 
    int r = MASK_LEN / 2;
    // Calculate starting point for the element
    int start = tid - r;
    // Temp value for calculation
    int temp = 0;
    
    for (int j = 0; j < MASK_LEN; j++) {
        // ignore the elements that hang off 
        if (((start + j) >= 0) && (start + j < n)) {
            // accumulate partial results
            temp += array[start + j] * mask[j];
        }
    }
    // write back the results
    result[tid] = temp;
}

void verify_result(int* arr, int* mask, int* result, int n) {
    int radius = MASK_LEN / 2;
    int temp, start;
    
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < MASK_LEN; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += arr[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    int n = 1 << 20;                           // num of elements in result array
    int bytes_n = n * sizeof(int);             // size of the array in bytes
    int m = 7;                                 // num of elements in convolution mask
    int bytes_m = MASK_LEN * sizeof(int);      // size of the mask in bytes
    
    // allocate the array, mask and result
    int* h_arr, *h_mask, *h_result;
    h_arr = new int[n];
    h_mask = new int[m];
    h_result = new int[n];
    
    // init array and masK
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100;
    }
    for (int i = 0; i < MASK_LEN; i++) {
        h_mask[i] = rand() % 10;
    }
    
    // allocate device memory
    int* d_arr, *d_result;
    cudaMalloc(&d_arr, bytes_n);
    cudaMalloc(&d_result, bytes_n);
    
    cudaMemcpy(d_arr, h_arr, bytes_n, cudaMemcpyHostToDevice);
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9bcf02b53644eee2bef9983d807084c7
    // copying the data(h_mask) to the given symbol(h_mask) on the device
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);
    
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;
    
    // call the kernl
    conv1d <<< GRID, THREADS >>> (d_arr, d_result, n);
    
    // copy the resulting array back to host
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);
    
    // verify the result
    verify_result(h_arr, h_mask, h_result, n);
    
    printf("SUCCESS!!!");
    
    // clean up
    delete[] h_arr; delete[] h_mask; delete[] h_result;
    cudaFree(d_arr); cudaFree(d_result);
    return 0;
}
