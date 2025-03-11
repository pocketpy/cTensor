#include "cten.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void cten_assert(bool cond, const char* fmt, ...) {
    if(!cond) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
    }
}

void cten_assert_shape(const char* title, TensorShape a, TensorShape b) {
    bool cond = memcmp(a, b, sizeof(TensorShape)) == 0;
    char buf_a[64];
    char buf_b[64];
    TensorShape_tostring(a, buf_a, sizeof(buf_a));
    TensorShape_tostring(b, buf_b, sizeof(buf_b));
    cten_assert(cond, "%s: %s != %s", title, buf_a, buf_b);
}

void cten_assert_dim(const char* title, int a, int b) {
    cten_assert(a == b, "%s: %d != %d", title, a, b);
}

bool cten_elemwise_broadcast(Tensor* a, Tensor* b) {
    int a_dim = TensorShape_dim(a->shape);
    int b_dim = TensorShape_dim(b->shape);

    if (a_dim != b_dim) return false;

    int a_broadcast = -1;

    for (int i = 0; i < a_dim; i++) {
        if (a->shape[i] == b->shape[i]) continue;
        if (a->shape[i] == 1) {
            if (a_broadcast == 0) return false;
            a_broadcast = 1;
        } else if (b->shape[i] == 1) {
            if (a_broadcast == 1) return false;
            a_broadcast = 0;
        } else {
            return false;
        }
    }

    if (a_broadcast != -1) {
        if (a_broadcast == 0) { 
            Tensor* tmp = a;
            a = b;
            b = tmp;
            a_broadcast = 1;
        }

        Tensor a_ = Tensor_zeros(b->shape, a->node != NULL);

        int stride_a_1 = (a_dim > 1) ? a->shape[1] : 1;
        int stride_a_2 = (a_dim > 2) ? a->shape[2] : 1;
        int stride_a_3 = (a_dim > 3) ? a->shape[3] : 1;

        int stride_a_1_new = (a_dim > 1) ? a_.shape[1] : 1;
        int stride_a_2_new = (a_dim > 2) ? a_.shape[2] : 1;
        int stride_a_3_new = (a_dim > 3) ? a_.shape[3] : 1;

        for (int i = 0; i < a_.shape[0]; i++) {
            int i_ = (a->shape[0] == 1) ? 0 : i;
            for (int j = 0; j < ((a_dim > 1) ? a_.shape[1] : 1); j++) {
                int j_ = (a_dim > 1 && a->shape[1] == 1) ? 0 : j;
                for (int k = 0; k < ((a_dim > 2) ? a_.shape[2] : 1); k++) {
                    int k_ = (a_dim > 2 && a->shape[2] == 1) ? 0 : k;
                    for (int l = 0; l < ((a_dim > 3) ? a_.shape[3] : 1); l++) {
                        int l_ = (a_dim > 3 && a->shape[3] == 1) ? 0 : l;

                        int dst_idx = i * stride_a_1_new * stride_a_2_new * stride_a_3_new +
                                      j * stride_a_2_new * stride_a_3_new +
                                      k * stride_a_3_new + l;

                        int src_idx = i_ * stride_a_1 * stride_a_2 * stride_a_3 +
                                      j_ * stride_a_2 * stride_a_3 +
                                      k_ * stride_a_3 + l_;

                        a_.data->flex[dst_idx] = a->data->flex[src_idx];
                    }
                }
            }
        }
        *a = a_;
    }

    return true;
}

void normalize(float X[][4], int num_samples) {
    float mean[4] = {0}, stddev[4] = {0};

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < num_samples; i++) {
            mean[j] += X[i][j];
        }
        mean[j] /= num_samples;
    }

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < num_samples; i++) {
            stddev[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
        stddev[j] = sqrt(stddev[j] / num_samples);
    }

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < num_samples; i++) {
            X[i][j] = (X[i][j] - mean[j]) / (stddev[j] + 1e-8);
        }
    }
}

void swap(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        float temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

void shuffle(float X[N_SAMPLES][N_FEATURES], int y[N_SAMPLES]) {
    for (int i = N_SAMPLES - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        swap(X[i], X[j], N_FEATURES);

        int temp = y[i];
        y[i] = y[j];
        y[j] = temp;
    }
}