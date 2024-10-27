// Attention implemented in C
// O = (Q @ Kt)
// O = O / sqrt(d)
// O = softmax(O) * V

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10
#define d 3


void init_matrix(float* m, int rows, int cols) {
    for (int i = 0; i < rows*cols; i++) {
        m[i] = (float)rand() / RAND_MAX;
    }
}

void print_matrix(float* M, int X, int Y) {
    printf("([\n");
    for (size_t i = 0; i < X; i++) {
        printf("  [");
        for (size_t j = 0; j < Y; j++) {
            printf(" %f", M[i * Y + j]);
        }
        printf(" ]\n");
    }
    printf("], size=(%d, %d))\n", X, Y);
}

void matmul(float* A, float* B, float* C, int X, int Y, int Z) {
    // A @ B = C --> (x, Y) @ (Y, Z) = (X, Z)
    for (int row = 0; row < X; row++) {
        for (int col = 0; col < Z; col++) {
            float temp = 0.0f;
            for (int k = 0; k < Y; k++) {
                temp += A[row * Y + k] * B[k * Z + col];
            }
            C[row * Z + col] = temp;
        }
    }
}

void matrix_transpose(float* out, float* in, int X, int Y) {
    for (size_t i = 0; i < X; i++) {
        for (size_t j = 0; j < Y; j++) {
            out[j * X + i] = in[i * Y + j];
        }
    }
}

void softmax(float* out, float* in, int X, int Y) {
    float max = -INFINITY;
    float denom = 0.f;

    float new_max, new_denom;
    for (int j = 0; j < X*Y; j++) {
        new_max = fmax(max, in[j]);
        new_denom = denom * expf(max - new_max) + expf(in[j] - new_max);

        max = new_max;
        denom = new_denom;
    }

    for (int i = 0; i < X*Y; i++) {
        out[i] = expf(in[i] - max) / denom;
    }
}

void matrix_scalar_div(float* out, float* in, float scalar, int X, int Y) {
    for (size_t i = 0; i < X * Y; i++) 
        out[i] = in[i] / scalar;
}

int main() {
    // Q: (N, d), K: (N, d), V: (N, d), Kt: (d, N)
    float* Q = malloc(sizeof(float) * N * d);
    float* K = malloc(sizeof(float) * N * d);
    float* V = malloc(sizeof(float) * N * d);
    float* Kt = malloc(sizeof(float) * d * N);

    float* QKt = malloc(sizeof(float) * N * N);
    float* O1 = malloc(sizeof(float) * N * N);
    float* O2 = malloc(sizeof(float) * N * N);
    float* O = malloc(sizeof(float) * N * d);

    init_matrix(Q, N, d);
    init_matrix(K, N, d);
    init_matrix(V, N, d);
    matrix_transpose(Kt, K, N, d);
    matmul(Q, Kt, QKt, N, d, N);
    softmax(O1, QKt, N, N);

    free(Q); free(K); free(V);
}
