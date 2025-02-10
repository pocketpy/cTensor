#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

Tensor nn_linear(Tensor input, Tensor weight, Tensor bias) {
    Tensor tmp = Tensor_matmul(input, weight);
    tmp = Tensor_add(tmp, bias);
    return tmp;
}

/* nn.relu */
static Tensor GradFn_relu(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int i = 0; i < input.data->numel; i++) {
        res.data->flex[i] = input.data->flex[i] > 0 ? 1 : 0;
    }
    return res;
}

Tensor nn_relu(Tensor self) {
    Tensor res = Tensor_new(self.shape, self.node != NULL);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = fmaxf(0, self.data->flex[i]);
    }

    if(self.node != NULL) {
        res.node->grad_fn = GradFn_relu;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

/* nn.softmax */
static Tensor GradFn_softmax(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    int input_dim = TensorShape_dim(input.shape);
    TensorShape res_shape = {0,0,0,0};
    for (int i = 0; i < input_dim * 2; i++) {
        res_shape[i] = input.shape[i % input_dim];
    }
    Tensor res = Tensor_new(res_shape, false);
    for(int mat_index = 0; mat_index < res.data->numel; mat_index++) {
        int i = mat_index / res_shape[input_dim - 1];
        int j = mat_index % res_shape[input_dim - 1];
        int kronecker = (i == j) ? 1 : 0;
        float yi = self.data->flex[i];
        float yj = self.data->flex[j];
        res.data->flex[mat_index] = yi * (kronecker - yj);
    }
    return res;
}

Tensor nn_softmax(Tensor self) {
    Tensor res = Tensor_new(self.shape, self.node != NULL);

    int self_dim = TensorShape_dim(self.shape);
    assert(self_dim > 0);
    int last_dim_size = self.shape[self_dim - 1];
    int outer_size = self.data->numel / last_dim_size;

    for(int outer = 0; outer < outer_size; outer++) {
        float max_val = -INFINITY;
        float sum = 0;

        for(int d = 0; d < last_dim_size; d++) {
            int index = outer * last_dim_size + d;
            max_val = fmaxf(max_val, self.data->flex[index]);
        }

        for(int d = 0; d < last_dim_size; d++) {
            int index = outer * last_dim_size + d;
            res.data->flex[index] = expf(self.data->flex[index] - max_val);
            sum += res.data->flex[index];
        }

        for(int d = 0; d < last_dim_size; d++) {
            int index = outer * last_dim_size + d;
            res.data->flex[index] /= sum;
        }
    }

    if(self.node != NULL) {
        res.node->grad_fn = GradFn_softmax;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }

    return res;
}

/* nn.cross_entropy */
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred) {
    // y_true: [None, n_classes]
    // y_pred: [None, n_classes]
    assert(TensorShape_dim(y_true.shape) == 2);
    assert(TensorShape_dim(y_pred.shape) == 2);

    int n_samples = y_true.shape[0];
    int n_classes = y_true.shape[1];
    assert(n_samples == y_pred.shape[0]);
    assert(n_classes == y_pred.shape[1]);

    Tensor res = Tensor_new((TensorShape){n_samples}, true);
    for(int i = 0; i < n_samples; i++) {
        float loss = 0;
        for(int j = 0; j < n_classes; j++) {
            loss += y_true.data->flex[i * n_classes + j] * logf(y_pred.data->flex[i * n_classes + j]);
        }
        res.data->flex[i] = -loss;
    }
    return Tensor_mean(res);
}