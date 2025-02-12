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

static void _dim_extend(TensorShape small_shape, TensorShape big_shape) {
    int small_dim = TensorShape_dim(small_shape);
    int big_dim = TensorShape_dim(big_shape);
    if (small_dim < 2) {
        int j = min(big_dim, 2);
        for (int i = small_dim; i < j; i++) small_shape[i] = 1;
        small_dim = j;
    }
    if (big_dim > 2) {
        for (int i = small_dim - 1; i >= 0; i--) {
            small_shape[i + big_dim - small_dim] = small_shape[i];
        }
        for (int i = 0; i < big_dim - small_dim; i++) small_shape[i] = 1;
    }
}

Tensor Tensor_mul(Tensor self, Tensor other) {
    int self_dim = TensorShape_dim(self.shape);
    int other_dim = TensorShape_dim(other.shape);
    if (self_dim != other_dim) {
        if (self_dim < other_dim) {
            _dim_extend(self.shape, other.shape);
        }
        else {
            _dim_extend(other.shape, self.shape);
        }
    }
    cten_elemwise_broadcast(&self, &other);
    bool require_grad = self.node != NULL || other.node != NULL;
    Tensor res = Tensor_new(self.shape, require_grad);
    for (int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] * other.data->flex[i];
    }
    if (require_grad) {
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
    TensorShape res_shape;
    memcpy(res_shape, _0.shape, sizeof(TensorShape));
    Tensor res = Tensor_new(res_shape, false);

    int _1_dim = TensorShape_dim(_1.shape);
    int dim_3 = 1, dim_4 = 1;
    int numel_dim_2 = _1.data->numel, numel_dim_3 = _1.data->numel;
    int res_numel_dim_2 = res.data->numel, res_numel_dim_3 = res.data->numel;
    if (_1_dim == 3) {
        dim_3 = _1.shape[0];
        numel_dim_2 = _1.shape[1] * _1.shape[2];
        numel_dim_3 = numel_dim_2;
        res_numel_dim_2 = _0.shape[1] * _0.shape[2];
        res_numel_dim_3 = res_numel_dim_2;
    }
    if (_1_dim == 4) {
        dim_4 = _1.shape[0];
        dim_3 = _1.shape[1];
        numel_dim_2 = _1.shape[2] * _1.shape[3];
        numel_dim_3 = numel_dim_2 * _1.shape[1];
        res_numel_dim_2 = _0.shape[2] * _0.shape[3];
        res_numel_dim_3 = res_numel_dim_2 * _0.shape[1];
    }

    for (int i_dim4 = 0; i_dim4 < dim_4; i_dim4++)
    {
        for (int i_dim3 = 0; i_dim3 < dim_3; i_dim3++) {
            for (int index = 0; index < res_numel_dim_2; index++) {
                int l = index / res_shape[1];
                int k = index % res_shape[1];
                float sum = 0;
                if (i == 0) {
                    for (int kj = k * _1.shape[1]; kj < (k + 1) * _1.shape[1]; kj++) {
                        sum += _1.data->flex[i_dim4*numel_dim_3 + i_dim3*numel_dim_2 + kj];
                    }
                }
                else {
                    for (int il = l; il < _1.shape[0] * _1.shape[1]; il += _1.shape[1]) {
                        sum += _1.data->flex[i_dim4 * numel_dim_3 + i_dim3 * numel_dim_2 + il];
                    }
                }
                res.data->flex[i_dim4 * res_numel_dim_3 + i_dim3 * res_numel_dim_2 + index] = sum;
            }
        }
    }
    return res;
}

Tensor Tensor_matmul(Tensor self, Tensor other) {
    int self_dim = TensorShape_dim(self.shape);
    int other_dim = TensorShape_dim(other.shape);
    assert(self_dim >= 2);
    assert(other_dim >= 2);
    //dimension support for 3, 4
    int dim_3 = 1, dim_4 = 1;
    int numel_dim_2 = self.data->numel, numel_dim_3 = self.data->numel;
    if (self_dim == 3) { 
        dim_3 = self.shape[0]; 
        numel_dim_2 = self.shape[1] * self.shape[2];
        numel_dim_3 = numel_dim_2;
    }
    if (self_dim == 4) {
        dim_4 = self.shape[0];
        dim_3 = self.shape[1];
        numel_dim_2 = self.shape[2] * self.shape[3];
        numel_dim_3 = numel_dim_2 * self.shape[1];
    }

    int m = self.shape[self_dim - 2];
    int n = self.shape[self_dim - 1];
    int p = other.shape[other_dim - 1];

    assert(n == other.shape[other_dim - 2]);

    TensorShape res_shape;
    memcpy(res_shape, self.shape, sizeof(TensorShape));
    res_shape[self_dim - 1] = p;
    Tensor res = Tensor_new(res_shape, self.node != NULL || other.node != NULL);

    for (int i_dim4 = 0; i_dim4 < dim_4; i_dim4++) {
        for (int i_dim3 = 0; i_dim3 < dim_3; i_dim3++) {
            //- TODO cache preload for matrix
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += self.data->flex[i_dim4*numel_dim_3 + i_dim3*numel_dim_2 + i * n + k] *   \
                            other.data->flex[i_dim4 * numel_dim_3 + i_dim3 * numel_dim_2 + k * p + j];
                    }
                    res.data->flex[i_dim4 * numel_dim_3 + i_dim3 * numel_dim_2 + i * p + j] = sum;
                }
            }
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