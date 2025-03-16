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
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] + other.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_add;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
        res.node->name = "Add";
    }
    return res;
}

Tensor Tensor_mul(Tensor self, Tensor other) {
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_mul() cannot broadcast", self.shape, other.shape);
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] * other.data->flex[i];
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_mul;
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
        res.node->name = "Mul";
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
        res.node->name = "Mean";
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
        res.node->name = "Sum";
    }
    return res;
}

static Tensor GradFn_matmul(Tensor self, int i) {
    // Tensor grad;
    // if (i == 0) {
    //     // Gradient with respect to first input: dL/dC * B^T
    //     Tensor upstream_grad = self.node->grad; // This should be dL/dC
    //     Tensor transposed_other = Tensor_transpose(Tensor_detach(self.node->inputs[1]));
    //     Tensor_print(self);
    //     Tensor_print(upstream_grad);
    //     Tensor_print(transposed_other);
    //     grad = Tensor_matmul(upstream_grad, transposed_other);
    // } else {
    //     // Gradient with respect to second input: A^T * dL/dC
    //     Tensor upstream_grad = self.node->grad; // This should be dL/dC
    //     Tensor transposed_first = Tensor_transpose(Tensor_detach(self.node->inputs[0]));
    //     grad = Tensor_matmul(transposed_first, upstream_grad);
    // }
    return Tensor_transpose(Tensor_detach(self.node->inputs[1-i]));;
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
    Tensor res = Tensor_new(res_shape, self.node != NULL || other.node != NULL); //here weight/bias have .node != NULL, so res have GradNode

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
        res.node->name = "Matmul";
    }

    return res;
}

static Tensor GradFn_sub(Tensor self, int i) {
    // f(x, y) = x - y; f'(x) = 1; f'(y) = -1
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_ones(input.shape, false);
    if(i == 1) {
        res = Tensor_mulf(res, -1);
    }
    return res;
}

Tensor Tensor_sub(Tensor self, Tensor other) {
    if (!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_sub() cannot broadcast", self.shape, other.shape);
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL || other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for (int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] - other.data->flex[i];
    }
    if (requires_grad) {
        res.node->grad_fn = GradFn_sub; // Define GradFn_sub if needed
        res.node->inputs[0] = self;
        res.node->inputs[1] = other;
        res.node->n_inputs = 2;
        res.node->name = "Sub";
    }
    return res;
}