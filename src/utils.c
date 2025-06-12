#include "cten.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    
#include <time.h>    

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
    
    if (a_dim == 1 && a->shape[0] == 1 && b_dim > 0) {
        Tensor a_ = Tensor_new(b->shape, a->node != NULL);
        float scalar_value = a->data->flex[0];
        int total_elements = TensorShape_numel(b->shape);
        for (int i = 0; i < total_elements; i++) {
            a_.data->flex[i] = scalar_value;
        }
        *a = a_;
        return true;
    }
    
    if (b_dim == 1 && b->shape[0] == 1 && a_dim > 0) {
        Tensor b_ = Tensor_new(a->shape, b->node != NULL);
        float scalar_value = b->data->flex[0];
        int total_elements = TensorShape_numel(a->shape);
        for (int i = 0; i < total_elements; i++) {
            b_.data->flex[i] = scalar_value;
        }
        *b = b_;
        return true;
    }
    
    if (a_dim != b_dim) return false;
    int a_broadcast = -1;
    for(int i = 0; i < a_dim; i++) {
        if(a->shape[i] == b->shape[i]) continue;
        if(a->shape[i] == 1) {
            if(a_broadcast == 0) return false;
            a_broadcast = 1;
        } else if(b->shape[i] == 1) {
            if(a_broadcast == 1) return false;
            a_broadcast = 0;
        } else {
            return false;
        }
    }
    if(a_broadcast != -1) {
        if(a_broadcast == 0) {
            Tensor* tmp = a;
            a = b;
            b = tmp;
            a_broadcast = 1;
        }
        Tensor a_ = Tensor_new(b->shape, a->node != NULL);
        for(int i = 0; i < a_.shape[0]; i++) {
            int i_ = a->shape[0] == 1 ? 0 : i;
            for(int j = 0; j < a_.shape[1]; j++) {
                int j_ = a->shape[1] == 1 ? 0 : j;
                for(int k = 0; k < a_.shape[2]; k++) {
                    int k_ = a->shape[2] == 1 ? 0 : k;
                    for(int l = 0; l < a_.shape[3]; l++) {
                        int l_ = a->shape[3] == 1 ? 0 : l;
                        // a_[i][j][k][l] = a[i_][j_][k_][l_]
                        a_.data->flex[i * a_.shape[1] * a_.shape[2] * a_.shape[3] +
                                      j * a_.shape[2] * a_.shape[3] + k * a_.shape[3] + l] =
                            a->data->flex[i_ * a->shape[1] * a->shape[2] * a->shape[3] +
                                          j_ * a->shape[2] * a->shape[3] + k_ * a->shape[3] + l_];
                    }
                }
            }
        }
        *a = a_;
    }
    return true;
}

void Tensor_normalize_dataset(const float (*X)[4], float (*X_norm)[4], int n_samples, int n_train_samples, int n_features) {
    float mean[4] = {0}, std[4] = {0};
    
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            mean[j] += X[i][j];
        }
    }
    for (int j = 0; j < n_features; j++) {
        mean[j] /= n_train_samples;
    }
    
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            std[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
    }
    for (int j = 0; j < n_features; j++) {
        std[j] = sqrtf(std[j] / n_train_samples);
        // Avoid division by zero
        if (std[j] == 0) std[j] = 1.0f;
    }

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X_norm[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }
}

void Tensor_shuffle_dataset(const float (*X)[4], const int *y,float (*X_shuffled)[4], int *y_shuffled, int n_samples, int n_features) {
    int* indices = malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    srand((unsigned)time(NULL));
    for (int i = n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    for (int i = 0; i < n_samples; i++) {
        int idx = indices[i];
        for (int j = 0; j < n_features; j++) {
            X_shuffled[i][j] = X[idx][j];
        }
        y_shuffled[i] = y[idx];
    }
    
    free(indices);
}

Tensor Tensor_reduce_dim(Tensor self, int dim, const char* operation) {
    int ndim = TensorShape_dim(self.shape);
    if (dim < 0 || dim >= ndim) {
        return self; // Return original tensor to avoid crash
    }
    
    TensorShape out_shape = {0, 0, 0, 0};
    int dim_size = self.shape[dim];
    int out_dim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_shape[out_dim++] = self.shape[i];
        }
    }
    
    Tensor res = Tensor_zeros(out_shape, false);
    
    // Calculate sum along the specified dimension
    for (int i = 0; i < self.data->numel; i++) {
        // Calculate the multi-dimensional index for this element
        int remaining = i;
        int indices[4] = {0, 0, 0, 0};
        int stride = self.data->numel;
        for (int j = 0; j < ndim; j++) {
            stride /= self.shape[j];
            indices[j] = remaining / stride;
            remaining %= stride;
        }
        
        // Calculate the corresponding index in the output tensor
        int out_idx = 0;
        int out_dim_idx = 0;
        int out_stride = 1;
        for (int j = ndim - 1; j >= 0; j--) {
            if (j != dim) {
                out_idx += indices[j] * out_stride;
                out_stride *= out_shape[out_dim_idx++];
            }
        }
        
        // Add to the accumulator
        res.data->flex[out_idx] += self.data->flex[i];
    }
    
    // If computing mean, divide by dimension size
    if (strcmp(operation, "mean") == 0) {
        for (int i = 0; i < res.data->numel; i++) {
            res.data->flex[i] /= dim_size;
        }
    }
    
    return res;
}