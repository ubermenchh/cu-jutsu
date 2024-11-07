// Sum Reduction Algorithm implemented in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum_reduction_kernel(float* in, float* out) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            in[i] += in[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = in[0];
    }
}

void init_vector(float* in, int N) {
    for (size_t i = 0; i < N; i++) {
        in[i] = (float)rand() / RAND_MAX;
    }
}

float cpu_sum(float* in, int N) {
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += in[i];
    }
    return sum;
}

int main() {
    const int N = 1 << 12;
    const int BLOCK_SIZE = N / 2;
    float* in = (float*)malloc(sizeof(float) * N);
    float out = 0.0f;
    
    init_vector(in, N);
    
    float* in_d, *out_d;
    cudaMalloc(&in_d, sizeof(float) * N);
    cudaMalloc(&out_d, sizeof(float));

    cudaMemcpy(in_d, in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, &out, sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_time = clock();
    sum_reduction_kernel <<< 1, BLOCK_SIZE >>> (in_d, out_d);
    clock_t end_time = clock();

    cudaMemcpy(&out, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    float cpu_result = cpu_sum(in, N);
    printf("GPU sum: %f\n", out);
    printf("CPU sum: %f\n", cpu_result);
    printf("Time Taken: %f\n", (float)(end_time - start_time) / CLOCKS_PER_SEC);

    free(in); cudaFree(in_d); free(out_d);
    return 0;
}
