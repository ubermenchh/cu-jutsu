// 1D Convulation in CUDA
// mask is stored in constant memory 
// and reused values are loaded into shared memory, but not halo elements

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MASK_LEN 7
__constant__ int mask[MASK_LEN];

__global__ void conv1d(int* arr, int* result, int n) {
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // store all elements needed to compute output in shared memory
    extern __shared__ int s_arr[];
    // load elements from main memory to shared memory 
    // naturally offset by `r` due to padding
    s_arr[threadIdx.x] = arr[tid];
    __syncthreads();
    
    // temp value
    int temp = 0;
    for (int j = 0; j < MASK_LEN; j++) {
        if ((threadIdx.x + j) >= blockDim.x) {
            temp += arr[tid + j] * mask[j];
        } else {
            temp += s_arr[threadIdx.x + j] * mask[j];
        }
    }
    // write-back the results
    result[tid] = temp;
}

void verify_result(int* arr, int* mask, int* result, int n) {
    int temp;
    for (int i = 0; i < n; i++) {
        temp = 0;
        for (int j = 0; j < MASK_LEN; j++) {
            temp += arr[i + j] * mask[j];
        }
        assert(temp == result[i]);
    }
}

int main() {
    // num of elements in result array
    int n = 1 << 20;
    // size of the array in bytes
    int bytes_n = n * sizeof(int);
    // size of mask in bytes
    size_t bytes_m = MASK_LEN * sizeof(int);
    
    // radius for padding the array
    int r = MASK_LEN / 2;
    int n_p = n + r * 2;
    
    // size of padded array in bytes
    size_t bytes_p = n_p * sizeof(int);
    
    // allocate the memory
    int* h_arr = new int[n_p];
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
            h_arr[i] = 0;
        } else {
            h_arr[i] = rand() % 100;
        }
    }
    
    int* h_mask = new int[MASK_LEN];
    for (int i = 0; i < MASK_LEN; i++) {
        h_mask[i] = rand() % 10;
    }
    
    // allocate memory for the result
    int* h_result = new int[n];
    
    int* d_arr, *d_result;
    cudaMalloc(&d_arr, bytes_p);
    cudaMalloc(&d_result, bytes_n);
    
    cudaMemcpy(d_arr, h_arr, bytes_p, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);
    
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;
    size_t SH_MEM = (THREADS + r * 2) * sizeof(int);
    
    conv1d <<< GRID, THREADS, SH_MEM >>> (d_arr, d_result, n);
    
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);
    
    verify_result(h_arr, h_mask, h_result, n);
    printf("SUCCESS!!!");
    
    delete[] h_arr; delete[] h_result; delete[] h_mask;
    cudaFree(d_arr); cudaFree(d_result);
    return 0;
}
