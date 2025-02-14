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

int _broadcast_offset_dim2(TensorShape shape, int* index, int dim) {
    int offset = 0, high_dim_offset = 0;
    if (dim > 2) {
        high_dim_offset = dim - 2;
        dim = 2;
    }
    for (int i = 0; i < dim; i++) {
        int stride = 1;
        for (int j = i + 1; j < dim; j++) {
            stride *= shape[j + high_dim_offset];
        }
        offset += index[i] * stride;
    }
    return offset;
}
//- TODO better policy wanted
bool cten_elemwise_broadcast(Tensor* a, Tensor* b) {
    int a_dim = TensorShape_dim(a->shape);
    int b_dim = TensorShape_dim(b->shape);
    bool completely_same = true;
    if(a_dim != b_dim) return false;
    for (int i = 0; i < a_dim; i++) {
        if (a->shape[i] != b->shape[i]) completely_same = false;
    }
    if (completely_same) return true;
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

        int dim_3 = 1, dim_4 = 1;
        int numel_dim_2 = b->data->numel, numel_dim_3 = 0;
        int a_numel_dim_2 = 0, a_numel_dim_3 = 0;
        bool a_has_dim2 = false;
        if (b_dim == 3) {
            dim_3 = b->shape[0];
            numel_dim_2 = b->shape[1] * b->shape[2];
            a_numel_dim_2 = a->shape[1] * a->shape[2];
            if (a->shape[0] == 1) a_numel_dim_2 = 0;
            if (a->data->numel == 1) a_numel_dim_2 = 0;
            if (a->shape[1] != 1 && a->shape[2] != 1) a_has_dim2 = true;
        }
        if (b_dim == 4) {
            dim_4 = b->shape[0];
            dim_3 = b->shape[1];
            numel_dim_2 = b->shape[2] * b->shape[3];
            numel_dim_3 = numel_dim_2 * b->shape[1];
            a_numel_dim_2 = a->shape[2] * a->shape[3];
            a_numel_dim_3 = a_numel_dim_2 * a->shape[1];
            if (a->shape[0] == 1) a_numel_dim_3 = 0;
            if (a->shape[1] == 1) a_numel_dim_2 = 0;
            if (a->data->numel == 1) { a_numel_dim_2 = 0; a_numel_dim_3 = 0; }
            if (a->shape[2] != 1 && a->shape[3] != 1) a_has_dim2 = true;
        }
        if (a_has_dim2 == false && a_dim > 1) {
            int tmp = a->shape[a_dim - 1];
            a->shape[a_dim - 1] = a->shape[a_dim - 2];
            a->shape[a_dim - 2] = tmp;
        }

        Tensor a_ = Tensor_new(b->shape, a->node != NULL);
        int index_a[2], index_a_;
        int high_dim_offset = a_dim - 2, a_dim_iter = 2;
        if (high_dim_offset < 0) {
            high_dim_offset = 0;
            a_dim_iter = a_dim;
        }
        for (int i_dim4 = 0; i_dim4 < dim_4; i_dim4++) {
            for (int i_dim3 = 0; i_dim3 < dim_3; i_dim3++) {
                for (int i = 0; i < numel_dim_2; i++) {
                    if (a_has_dim2) {
                        a_.data->flex[i_dim4 * numel_dim_3 + i_dim3 * numel_dim_2 + i] = \
                            a->data->flex[i_dim4 * a_numel_dim_3 + i_dim3 * a_numel_dim_2 + i];
                    }
                    else {
                        int curr_index = i;
                        for (int j = 0; j < a_dim_iter; j++) {
                            index_a_ = curr_index % a_.shape[a_dim - 1];
                            index_a[j] = (a->shape[j + high_dim_offset] == 1) ? 0 : index_a_;
                            curr_index /= a_.shape[a_dim - 1];
                        }
                        a_.data->flex[i_dim4 * numel_dim_3 + i_dim3 * numel_dim_2 + i] =    \
                            a->data->flex[i_dim4 * a_numel_dim_3 + i_dim3 * a_numel_dim_2 +     \
                            _broadcast_offset_dim2(a->shape, index_a, a_dim)];
                    }
                }
            }
        }
        *a = a_;
    }
    return true;
}
