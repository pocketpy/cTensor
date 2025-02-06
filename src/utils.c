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
        exit(1);
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

int _broadcast_offset(TensorShape shape, int* index, int dim) {
    int offset = 0;
    for (int i = 0; i < dim; i++) {
        int stride = 1;
        for (int j = i + 1; j < dim; j++) {
            stride *= shape[j];
        }
        offset += index[i] * stride;
    }
    return offset;
}

bool cten_elemwise_broadcast(Tensor* a, Tensor* b) {
    int a_dim = TensorShape_dim(a->shape);
    int b_dim = TensorShape_dim(b->shape);
    if(a_dim != b_dim) return false;
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
        int index_a[4], index_a_;
        for (int i = 0; i < a_.data->numel; i++) {
            int curr_index = i;
            for (int j = 0; j <= a_dim-1; j++) {
                index_a_ = curr_index % a_.shape[j];
                index_a[j] = (a->shape[j] == 1) ? 0 : index_a_;
                curr_index /= a_.shape[j];
            }
            a_.data->flex[i] = a->data->flex[_broadcast_offset(a->shape, index_a, a_dim)];
        }

        *a = a_;
    }
    return true;
}
