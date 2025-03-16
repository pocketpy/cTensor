#include "cten.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
