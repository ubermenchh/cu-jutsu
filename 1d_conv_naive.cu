// 1D Convolution implemented in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void conv1d(int* array, int* mask, int* result, int n, int m) {
    /* 1D Convolution Kernel */
    // Arguments:
    //      array -> padded array
    //       mask -> convolution mask
    //     result -> resulting array
    //          n -> num of elements in array
    //          m -> num of elements in the mask
    
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate radius of mask 
    int r = m / 2;
    // Calculate starting point for the element
    int start = tid - r;
    // Temp value for calculation
    int temp = 0;
    
    for (int j = 0; j < m; j++) {
        // ignore the elements that hang off 
        if (((start + j) >= 0) && (start + j < n)) {
            // accumulate partial results
            temp += array[start + j] * mask[j];
        }
    }
    // write back the results
    result[tid] = temp;
}

void verify_result(int* arr, int* mask, int* result, int n, int m) {
    int radius = m / 2;
    int temp, start;
    
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < m; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += arr[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    int n = 1 << 20;                    // num of elements in result array
    int bytes_n = n * sizeof(int);      // size of the array in bytes
    int m = 7;                          // num of elements in convolution mask
    int bytes_m = m * sizeof(int);      // size of the mask in bytes
    
    // allocate the array, mask and result
    int* h_arr, *h_mask, *h_result;
    h_arr = (int*)malloc(bytes_n);
    h_mask = (int*)malloc(bytes_m);
    h_result = (int*)malloc(bytes_n);
    
    // init array and masK
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 100;
    }
    for (int i = 0; i < m; i++) {
        h_mask[i] = rand() % 10;
    }
    
    // allocate device memory
    int* d_arr, *d_mask, *d_result;
    cudaMalloc(&d_arr, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);
    
    cudaMemcpy(d_arr, h_arr, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);
    
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;
    
    // call the kernl
    conv1d <<< GRID, THREADS >>> (d_arr, d_mask, d_result, n, m);
    
    // copy the resulting array back to host
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);
    
    // verify the result
    verify_result(h_arr, h_mask, h_result, n, m);
    
    printf("SUCCESS!!!");
    
    // clean up
    cudaFree(d_arr); cudaFree(d_mask); cudaFree(d_result);
    free(h_arr); free(h_mask); free(h_result);
    return 0;
}
