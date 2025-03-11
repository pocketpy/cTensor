#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float rand_normal() {
    float u1 = (rand() + 1.0) / (RAND_MAX + 1.0);
    float u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}

int TensorShape_numel(TensorShape shape) {
    int numel = 1;
    for(int i = 0; i < sizeof(TensorShape) / sizeof(shape[0]); i++) {
        if(shape[i] == 0) break;
        numel *= shape[i];
    }
    return numel;
}

int TensorShape_dim(TensorShape shape) {
    for(int i = 0; i < sizeof(TensorShape) / sizeof(shape[0]); i++) {
        if(shape[i] == 0) return i;
    }
    return sizeof(TensorShape) / sizeof(shape[0]);
}

int TensorShape_asdim(TensorShape shape, int dim) {
    int shape_dim = TensorShape_dim(shape);
    if(dim < 0) dim += shape_dim;
    cten_assert(dim >= 0 && dim < shape_dim, "dim %d out of range", dim);
    return dim;
}

int TensorShape_tostring(TensorShape shape, char* buf, int size) {
    return snprintf(buf, size, "(%d, %d, %d, %d)", shape[0], shape[1], shape[2], shape[3]);
}

bool TensorShape_equals(Tensor a, Tensor b) {
    for (int i = 0; i < 4; i++) {
        if (a.shape[i] != b.shape[i]) {
            return false;
        }
    }
    return true;
}

Tensor Tensor_new(TensorShape shape, bool requires_grad) {
    Tensor self;
    memcpy(self.shape, shape, sizeof(TensorShape));
    int numel = TensorShape_numel(shape);
    self.data = _cten_malloc(sizeof(FloatBuffer) + sizeof(float) * numel);
    self.data->numel = numel;
    if(requires_grad) {
        self.node = _cten_malloc(sizeof(GradNode));
        memset(self.node, 0, sizeof(GradNode));
    } else {
        self.node = NULL;
    }
    return self;
}

Tensor Tensor_zeros(TensorShape shape, bool requires_grad) {
    Tensor self = Tensor_new(shape, requires_grad);
    memset(self.data->flex, 0, sizeof(float) * self.data->numel);
    return self;
}

Tensor Tensor_ones(TensorShape shape, bool requires_grad) {
    Tensor self = Tensor_new(shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        self.data->flex[i] = 1.0f;
    }
    return self;
}

Tensor Tensor_init_he(Tensor self) {
    float he_std = sqrtf(2.0f / self.shape[0]); 
    for (int i = 0; i < self.data->numel; i++) {
        self.data->flex[i] = he_std * rand_normal(); 
    }
    return self;
}

float Tensor_get(Tensor self, int i, int j, int k, int l) {
    assert((self.shape[0] == 0 && i == 0) || (i >= 0 && i < self.shape[0]));
    assert((self.shape[1] == 0 && j == 0) || (j >= 0 && j < self.shape[1]));
    assert((self.shape[2] == 0 && k == 0) || (k >= 0 && k < self.shape[2]));
    assert((self.shape[3] == 0 && l == 0) || (l >= 0 && l < self.shape[3]));
    return self.data->flex[i * self.shape[1] * self.shape[2] * self.shape[3] +
                           j * self.shape[2] * self.shape[3] + k * self.shape[3] + l];
}

void Tensor_set(Tensor self, int i, int j, int k, int l, float value) {
    assert((self.shape[0] == 0 && i == 0) || (i >= 0 && i < self.shape[0]));
    assert((self.shape[1] == 0 && j == 0) || (j >= 0 && j < self.shape[1]));
    assert((self.shape[2] == 0 && k == 0) || (k >= 0 && k < self.shape[2]));
    assert((self.shape[3] == 0 && l == 0) || (l >= 0 && l < self.shape[3]));
    self.data->flex[i * self.shape[1] * self.shape[2] * self.shape[3] +
                    j * self.shape[2] * self.shape[3] + k * self.shape[3] + l] = value;
}

Tensor Tensor_detach(Tensor self) {
    Tensor detached = self;
    detached.node = NULL;
    return detached;
}

Tensor Tensor_sum_reduce(Tensor self, TensorShape expected_shape) {
    assert(TensorShape_dim(self.shape) == TensorShape_dim(expected_shape));

    Tensor res = Tensor_zeros(expected_shape, false);
    int self_dim = TensorShape_dim(self.shape);
    int input_strides[self_dim], output_strides[self_dim];

    input_strides[self_dim - 1] = 1;
    output_strides[self_dim - 1] = 1;

    for (int i = self_dim - 2; i >= 0; i--) {
        input_strides[i] = input_strides[i + 1] * self.shape[i + 1];
        output_strides[i] = output_strides[i + 1] * expected_shape[i + 1];
    }

    for (int i = 0; i < self.data->numel; i++) {
        int coord[self_dim];
        int res_idx = 0;
        for (int j = 0; j < self_dim; j++) {
            coord[j] = (i / input_strides[j]) % self.shape[j];
            if (expected_shape[j] == 1) coord[j] = 0; 
            res_idx += coord[j] * output_strides[j];
        }
        res.data->flex[res_idx] += self.data->flex[i];
    }
    return res;
}

void Tensor_backward(Tensor self, Tensor grad) {
    if (self.node == NULL) return;

    if (grad.data == NULL) {
        assert(self.data->numel == 1);
        grad = Tensor_ones(self.shape, false);
    }

    assert(grad.node == NULL);

    float clip_value = 1.0f;  // gradient clipping
    for (int i = 0; i < grad.data->numel; i++) {
        grad.data->flex[i] = fmaxf(fminf(grad.data->flex[i], clip_value), -clip_value);
    }

    if (self.node->grad.data == NULL) {
        Tensor zeros = Tensor_zeros(self.shape, false);
        self.node->grad = Tensor_add(zeros, grad);
    } else {
        self.node->grad = Tensor_add(self.node->grad, grad);
    }

    for (int i = 0; i < self.node->n_inputs; i++) {
        Tensor local_grad = self.node->grad_fn(self, i);
        Tensor propagated_grad;

        if (self.node->grad_fn == GradFn_matmul) {
            propagated_grad = local_grad;
        } else {
            propagated_grad = Tensor_mul(grad, local_grad);
        }

        if (!TensorShape_equals(self.node->inputs[i], propagated_grad)) {
            propagated_grad = Tensor_sum_reduce(propagated_grad, self.node->inputs[i].shape);
        }
        // Recurse backward
        Tensor_backward(self.node->inputs[i], propagated_grad);
    }
}

int Tensor_backward_apply(Tensor self, void (*f)(Tensor, void*), void* ctx) {
    if(self.node == NULL) return 0;
    if(f != NULL) f(self, ctx);
    int count = 1;
    for(int i = 0; i < self.node->n_inputs; i++) {
        count += Tensor_backward_apply(self.node->inputs[i], f, ctx);
    }
    return count;
}

void Tensor_print(Tensor self) {
    if(self.data == NULL) {
        printf("Tensor()\n");
        return;
    }
    printf("Tensor([");
    for(int i = 0; i < self.data->numel; i++) {
        printf("%.4f", self.data->flex[i]);
        if(i < self.data->numel - 1) printf(", ");
    }
    printf("], shape=(");
    for(int i = 0; i < 4; i++) {
        if(self.shape[i] == 0) {
            break;
        } else {
            if(i > 0) printf(", ");
        }
        printf("%d", self.shape[i]);
    }

    if(self.node != NULL) {
        printf("), grad_fn=<%p>, grad=", self.node->grad_fn);
        Tensor_print(self.node->grad);
    } else {
        printf(")");
    }
    printf(")\n");
}

void _cten_zero_grad(Tensor* params, int n_params) {
    for(int i = 0; i < n_params; i++) {
        Tensor t = params[i];
        if(t.node == NULL) continue;
        t.node->grad = Tensor_zeros(t.shape, false);
    }
}