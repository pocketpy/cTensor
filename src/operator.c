#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_add() cannot broadcast", self.shape, other.shape);
    }
    bool require_grad = self.node != NULL || other.node != NULL;
    Tensor res = Tensor_new(self.shape, require_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] + other.data->flex[i];
    }
    if(require_grad) {
        res.node->grad_fn = GradFn_add;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
    }
    return res;
}

Tensor Tensor_mul(Tensor self, Tensor other) {
    int f_index = -1;
    if (TensorShape_dim(self.shape) == 0 || TensorShape_dim(other.shape) == 0) {
        f_index = (TensorShape_dim(self.shape) == 0) ? 0 : 1;
    }
    else{
        if (!cten_elemwise_broadcast(&self, &other)) {
            cten_assert_shape("Tensor_mul() cannot broadcast", self.shape, other.shape);
        }
    }
    TensorShape* res_shape =  f_index ? &self.shape : &other.shape ;
    bool require_grad = self.node != NULL || other.node != NULL;
    Tensor res = Tensor_new(*res_shape, require_grad);
    if (f_index == -1) {
        for (int i = 0; i < self.data->numel; i++) {
            res.data->flex[i] = self.data->flex[i] * other.data->flex[i];
        }
    }
    else if (f_index == 0) {
        for (int i = 0; i < other.data->numel; i++) {
            res.data->flex[i] = self.data->flex[0] * other.data->flex[i];
        }
    }
    else {
        for (int i = 0; i < self.data->numel; i++) {
            res.data->flex[i] = self.data->flex[i] * other.data->flex[0];
        }
    }
    if(require_grad) {
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
    // Tensor_delete(tmp);
    return res;
}

int* Tensor_argmax(Tensor self, int dim) {
    dim = TensorShape_asdim(self.shape, dim);
    int* res = (int*)malloc(sizeof(int) * self.shape[dim]);
    for(int i = 0; i < self.shape[dim]; i++) {
        res[i] = 0;
        for(int j = 0; j < self.shape[dim]; j++) {
            float _0 = self.data->flex[res[i] * self.shape[dim] + i];
            float _1 = self.data->flex[j * self.shape[dim] + i];
            if(_0 < _1) res[i] = j;
        }
    }
    return res;
}

static Tensor GradFn_mean(Tensor self, int i) {
    // f(x) = mean(x); f'(x) = 1 / x.numel()
    Tensor res = Tensor_new(self.shape, false);
    for(int i = 0; i < res.data->numel; i++) {
        res.data->flex[i] = 1.0f / self.data->numel;
    }
    return res;
}

Tensor Tensor_mean(Tensor self) {
    Tensor res = Tensor_new((TensorShape){0}, self.node != NULL);
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

static Tensor GradFn_matmul(Tensor self, int i) {
    Tensor _0 = self.node->inputs[i];
    Tensor _1 = self.node->inputs[1 - i];
    _0 = Tensor_detach(_0);
    _1 = Tensor_detach(_1);
    return Tensor_matmul(_0, _1);
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