#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define Bs 2
#define N 3
#define d 3

void init_matrix(float* M) {
    for (size_t i = 0; i < Bs*N*d; i++) 
        M[i] = (float)rand() / RAND_MAX;
}

void print_matrix(float* M) {
    printf("[\n");
    for (size_t b = 0; b < Bs; b++) {
        printf("  [\n");
        for (size_t n = 0; n < N; n++) {
            printf("    [");
            for (size_t i = 0; i < d; i++) {
                printf(" %f", M[b * (N * d) + n * d + i]);
            }
            printf(" ]\n");
        }
        printf("  ]\n");
    }
    printf("]\n");
}

void matmul(float* A, float* B, float* C, int X, int Y, int Z) {
    // A: (Bs, X, Y)
    // B: (Bs, Y, Z)
    // C: (Bs, X, Z)

    for (size_t b = 0; b < Bs; b++) {
        for (size_t x = 0; x < X; x++) {
            for (size_t z = 0; z < Z; z++) {
                float temp = 0.0f;
                for (size_t y = 0; y < Y; y++) {
                    // A[b, x, i] * B[b, i, y]
                    temp += A[b * (X * Z) + x * Y + y] * B[b * (Y * Z) + y * Z + z];
                }
                // C[b, x, z]
                C[b * (X * Z) + x * Z + z] = temp;
            }
        }
    }
}

void softmax(float* out, float* in, int X, int Y) {
    float max = -INFINITY;
    float denom = 0.f;

    float new_max, new_denom;
    for (int j = 0; j < Bs*X*Y; j++) {
        new_max = fmax(max, in[j]);
        new_denom = denom * expf(max - new_max) + expf(in[j] - new_max);

        max = new_max;
        denom = new_denom;
    }

    for (int i = 0; i < Bs*X*Y; i++) {
        out[i] = expf(in[i] - max) / denom;
    }
}

void scale(float* m, float scale, int size) {
    for (size_t i = 0; i < size; i++) {
        m[i] *= scale;
    }
}

int main() {
    float* Q = malloc(sizeof(float) * Bs * N * d);
    float* K = malloc(sizeof(float) * Bs * d * N); // K = K.T
    float* V = malloc(sizeof(float) * Bs * N * d);
    float* S = malloc(sizeof(float) * Bs * N * N);
    float* P = malloc(sizeof(float) * Bs * N * N);
    float* O = malloc(sizeof(float) * Bs * N * d);

    init_matrix(Q); init_matrix(K); init_matrix(V);

    matmul(Q, K, S, N, d, N);
    print_matrix(S);

    float scale_factor = 1.0f / sqrtf(d);
    scale(S, scale_factor, Bs*N*N);
    print_matrix(S);

    softmax(P, S, N, N);
    print_matrix(P);
    matmul(P, V, O, N, N, d);
    print_matrix(O);

    return 0;
}
