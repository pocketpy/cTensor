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
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = fmaxf(0, self.data->flex[i]);
    }

    if(requires_grad) {
        res.node->grad_fn = GradFn_relu;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

/* nn.softmax */
static Tensor GradFn_softmax(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    int num_classes = input.shape[TensorShape_dim(input.shape) - 1]; 
    int batch_size = input.data->numel / num_classes;

    Tensor res = Tensor_zeros(input.shape, false);

    for (int batch = 0; batch < batch_size; batch++) {
        float grad_sum = 0.0f;

        for (int k = 0; k < num_classes; k++) {
            grad_sum += self.data->flex[batch * num_classes + k] * self.node->grad.data->flex[batch * num_classes + k];
        }

        for (int j = 0; j < num_classes; j++) {
            float softmax_j = self.data->flex[batch * num_classes + j];
            float grad_j = self.node->grad.data->flex[batch * num_classes + j];

            res.data->flex[batch * num_classes + j] = softmax_j * (grad_j - grad_sum);
        }
    }
    return res;
}

Tensor nn_softmax(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
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

    if(requires_grad) {
        res.node->grad_fn = GradFn_softmax;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
    }
    return res;
}

Tensor GradFn_crossentropy(Tensor self, int i) {
    Tensor y_true = self.node->inputs[0];
    Tensor y_pred = self.node->inputs[1];

    Tensor grad = Tensor_zeros(y_pred.shape, false);

    if (i == 1) {
        // f'(y_true, y_pred) = -y_true / y_pred
        int n_samples = y_pred.shape[0];
        int n_classes = y_pred.shape[1];

        for (int s = 0; s < n_samples; s++) {
            for (int c = 0; c < n_classes; c++) {
                int idx = s * n_classes + c;
                grad.data->flex[idx] = -y_true.data->flex[idx] / y_pred.data->flex[idx];
            }
        }
    }
    return grad;
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

    bool requires_grad = !cten_is_eval() && (y_true.node != NULL || y_pred.node != NULL);
    Tensor res = Tensor_new((TensorShape){n_samples, 1}, requires_grad);
    for(int i = 0; i < n_samples; i++) {
        float loss = 0;
        for(int j = 0; j < n_classes; j++) {
            loss +=
                y_true.data->flex[i * n_classes + j] * logf(y_pred.data->flex[i * n_classes + j]);
        }
        res.data->flex[i] = -loss;
    }

    if (requires_grad) {
        res.node->grad_fn = GradFn_crossentropy;
        res.node->inputs[0] = y_true;
        res.node->inputs[1] = y_pred;
        res.node->n_inputs = 2;
    }
    return Tensor_mean(res);
}