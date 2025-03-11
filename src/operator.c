#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static Tensor GradFn_add(Tensor self, int i) {
    // f(x, y) = x + y; f'(x) = 1; f'(y) = 1
    Tensor input = self.node->inputs[i];
    return Tensor_ones(input.shape, false);
}

static Tensor GradFn_mul(Tensor self, int i) {
    // f(x, y) = x * y; f'(x) = y; f'(y) = x
    return Tensor_detach(self.node->inputs[1 - i]);
}

Tensor Tensor_add(Tensor self, Tensor other) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);

    if(requires_grad) {
        res.node->grad_fn = GradFn_add;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }

    if(!TensorShape_equals(self, other) && !cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_add() cannot broadcast", self.shape, other.shape);
    }    

    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] + other.data->flex[i];
    }

    return res;
}

Tensor Tensor_mul(Tensor self, Tensor other) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    
    if (!TensorShape_equals(self, other) && !cten_elemwise_broadcast(&self, &other)) {
        printf("Tensor_mul() cannot broadcast: ");
        Tensor_print(self);
        Tensor_print(other);
        exit(1);
    }

    Tensor res = Tensor_ones(self.shape, requires_grad);
    for (int i = 0; i < res.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] * other.data->flex[i];
    }

    if (requires_grad) {
        res.node->grad_fn = GradFn_mul;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

Tensor Tensor_mulf(Tensor self, float other) {
    Tensor tmp = Tensor_new(self.shape, false);
    for(int i = 0; i < tmp.data->numel; i++) {
        tmp.data->flex[i] = other;
    }
    Tensor res = Tensor_mul(self, tmp);
    return res;
}

void Tensor_argmax(Tensor self, int* out) {
    // reduce last dim
    int last_dim = self.shape[TensorShape_dim(self.shape) - 1];
    int n = TensorShape_numel(self.shape) / last_dim;
    for(int i = 0; i < n; i++) {
        float* p = self.data->flex + i * last_dim;
        float max_val = p[0];
        int max_idx = 0;
        for(int j = 1; j < last_dim; j++) {
            if(p[j] > max_val) {
                max_val = p[j];
                max_idx = j;
            }
        }
        out[i] = max_idx;
    }
}

static Tensor GradFn_mean(Tensor self, int i) {
    // f(x) = mean(x); f'(x) = 1 / x.numel()
    Tensor res = Tensor_new(self.node->inputs[i].shape, false);
    float scale = 1.0f / self.node->inputs[i].data->numel;
    for(int i = 0; i < res.data->numel; i++) {
        res.data->flex[i] = scale;
    }
    return res;
}

Tensor Tensor_mean(Tensor self) {
    Tensor res = Tensor_new((TensorShape){1,1}, self.node != NULL);
    float sum = 0;
    for(int i = 0; i < self.data->numel; i++) {
        sum += self.data->flex[i];
    }
    res.data->flex[0] = sum / self.data->numel;
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_mean;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

static Tensor GradFn_sum(Tensor self, int i) {
    // f(x) = sum(x); f'(x) = 1
    return Tensor_ones(self.node->inputs[i].shape, false);
}

Tensor Tensor_sum(Tensor self) {
    Tensor res = Tensor_new((TensorShape){0}, self.node != NULL);
    float sum = 0;
    for(int i = 0; i < self.data->numel; i++) {
        sum += self.data->flex[i];
    }
    res.data->flex[0] = sum;
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_sum;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

Tensor Tensor_transpose(Tensor self, int dim0, int dim1) {
    int self_dim = TensorShape_dim(self.shape);
    dim0 = TensorShape_asdim(self.shape, dim0);
    dim1 = TensorShape_asdim(self.shape, dim1);
    assert(self_dim >= 2 && dim0 < self_dim && dim1 < self_dim);

    TensorShape new_shape;
    memcpy(new_shape, self.shape, sizeof(TensorShape));
    new_shape[dim0] = self.shape[dim1];
    new_shape[dim1] = self.shape[dim0];

    Tensor res = Tensor_zeros(new_shape, self.node != NULL);

    int strides[self_dim];
    strides[self_dim - 1] = 1;
    for (int i = self_dim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * self.shape[i + 1];
    }

    int new_strides[self_dim];
    new_strides[self_dim - 1] = 1;
    for (int i = self_dim - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    for (int i = 0; i < self.data->numel; i++) {
        int src_idx = i;
        int coord[self_dim];

        for (int j = 0; j < self_dim; j++) {
            coord[j] = (src_idx / strides[j]) % self.shape[j];
        }

        int temp = coord[dim0];
        coord[dim0] = coord[dim1];
        coord[dim1] = temp;

        int dst_idx = 0;
        for (int j = 0; j < self_dim; j++) {
            dst_idx += coord[j] * new_strides[j];
        }

        assert(dst_idx >= res.data->numel);

        res.data->flex[dst_idx] = self.data->flex[i];
    }

    return res;
}

Tensor GradFn_matmul(Tensor self, int i) {
    Tensor _0 = self.node->inputs[0]; 
    Tensor _1 = self.node->inputs[1]; 

    _0 = Tensor_detach(_0);
    _1 = Tensor_detach(_1);
    
    if (i == 0) {
        return Tensor_matmul(self.node->grad, Tensor_transpose(_1, 0, 1));
    } else {
        return Tensor_matmul(Tensor_transpose(_0, 0, 1), self.node->grad);
    }
}

Tensor Tensor_matmul(Tensor self, Tensor other) {
    int self_dim = TensorShape_dim(self.shape);
    int other_dim = TensorShape_dim(other.shape);
    assert(self_dim >= 2);
    assert(other_dim >= 2);

    int m = self.shape[self_dim - 2];
    int n = self.shape[self_dim - 1];
    int p = other.shape[other_dim - 1];

    assert(n == other.shape[other_dim - 2]);

    TensorShape res_shape;
    memcpy(res_shape, self.shape, sizeof(TensorShape));
    res_shape[self_dim - 1] = p;
    Tensor res = Tensor_new(res_shape, self.node != NULL || other.node != NULL);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            float sum = 0;
            for(int k = 0; k < n; k++) {
                sum += self.data->flex[i * n + k] * other.data->flex[k * p + j];
            }
            res.data->flex[i * p + j] = sum;
        }
    }

    if(res.node != NULL) {
        res.node->grad_fn = GradFn_matmul;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }

    return res;
}

bool TensorShape_equals(Tensor a, Tensor b) {
    for (int i = 0; i < 4; i++) {
        if (a.shape[i] != b.shape[i]) {
            return false;
        }
    }
    return true;
}