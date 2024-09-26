// Cache Tiled Matrix Multiplication(Matmul with shared memory) in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define SHMEM_SIZE (16 * 16)

__global__ void matmul(int* a, int* b, int* c, int N) {
    // allocate shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];
    
    // calculate each thread's global row and column
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;
    
    // move the tile across the length of the grid
    int temp = 0;
    for (int i = 0; i < ((N + dim - 1) / dim); i++) {
        A[ty * dim + tx] = a[(row * N) + (i * dim) + tx];
        B[ty * dim + tx] = b[(i * dim * N) + (ty * N) + col];
        __syncthreads();
        
        // accumulate the partial results
        for (int j = 0; j < dim; j++) {
            temp += A[ty * dim + j] * B[j * dim + tx];           
        }
        __syncthreads();
    }
    c[row * N + col] = temp;
}

void init_matrix(int* m, int N) {
    // Initializes a square matrix with random numbers b/w 0-100
    for (int i = 0; i < N; i++) {
        m[i] = rand() % 100;
    }
}

void verify(int* a, int* b, int* c, int N) {
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
    // matrix dimensions (default: 2^10)
    int N = 1 << 10;
    size_t bytes = N * sizeof(int);
    
    // allocate memory
    int* a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    
    // intialize input matrices
    init_matrix(a, N);
    init_matrix(b, N);
    
    // set CTA and grid sizes
    int threads = 16;
    int blocks = (N + threads - 1) / threads;
    
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    
    // launch the kernel
    matmul <<< BLOCKS, THREADS >>> (a, b, c, N);
    cudaDeviceSynchronize();
    
    // verify result
    //verify(a, b, c, N);
    //printf("SUCCESS!!!");
    
    return 0;
}
