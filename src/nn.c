#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

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
    Tensor res = Tensor_zeros(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = fmaxf(0, self.data->flex[i]);
    }

    if(requires_grad) {
        res.node->grad_fn = GradFn_relu;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Relu";
    }
    return res;
}

Tensor nn_random_init(TensorShape shape, bool requires_grad) {
    Tensor res = Tensor_new(shape, requires_grad);
    int fan_in = shape[0];
    int fan_out = shape[1];
    float scale = sqrtf(6.0f / (fan_in + fan_out));
    
    for(int i = 0; i < res.data->numel; i++) {
        float r = (float)rand() / RAND_MAX * 2.0f - 1.0f; 
        res.data->flex[i] = r * scale;
    }
    return res;
}

/* nn.softmax - Correct implementation */
static Tensor GradFn_softmax(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor grad = Tensor_new(input.shape, false);
    
    int dim = TensorShape_dim(self.shape);
    int batch_size = self.shape[0];
    int num_classes = self.shape[1];  
    for(int b = 0; b < batch_size; b++){
        for(int i = 0; i < num_classes; i++) {
            for(int j = 0; j < num_classes; j++) {
                float softmax_i = self.data->flex[b * num_classes + i];
                float softmax_j = self.data->flex[b * num_classes + j];
                float value;
                if(i == j){
                    value = softmax_i * (1.0f - softmax_i);
                } 
                else{
                    value = -softmax_i * softmax_j;
                }
                
                if(i == j){
                    grad.data->flex[b * num_classes + i] = value;
                }
            }
        }
    }
    return grad;
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
        res.node->name = "Softmax";     
    }
    return res;
}

/* nn.cross_entropy */
static Tensor GradFn_crossentropy(Tensor self, int i) {
    if (i == 0) {
        return Tensor_zeros(self.node->inputs[0].shape, false);
    } else if (i == 1) {
        Tensor y_true = self.node->inputs[0];
        Tensor y_pred = self.node->inputs[1];
        
        int n_samples = y_true.shape[0];
        int n_classes = y_true.shape[1];
        
        Tensor grad = Tensor_new(y_pred.shape, false);
        
        for(int i = 0; i < n_samples; i++) {
            for(int j = 0; j < n_classes; j++) {
                float y_true_val = y_true.data->flex[i * n_classes + j];
                float y_pred_val = y_pred.data->flex[i * n_classes + j];
                grad.data->flex[i * n_classes + j] = (y_pred_val - y_true_val) / n_samples;
            }
        }
        return grad;
    }
    return Tensor_zeros((TensorShape){1}, false);
}

Tensor nn_crossentropy(Tensor y_true, Tensor y_pred) {
    // y_true: [None, n_classes]
    // y_pred: [None, n_classes]
    assert(TensorShape_dim(y_true.shape) == 2);
    assert(TensorShape_dim(y_pred.shape) == 2);

    int n_samples = y_true.shape[0];
    int n_classes = y_true.shape[1];
    assert(n_samples == y_pred.shape[0]);
    assert(n_classes == y_pred.shape[1]);

    bool requires_grad = !cten_is_eval() && (y_true.node != NULL || y_pred.node != NULL); //No eval but rather training so requires grad is True
    Tensor res = Tensor_zeros((TensorShape){1}, requires_grad);
    
    // Calculate cross-entropy loss
    float total_loss = 0.0f;
    for(int i = 0; i < n_samples; i++) {
        float sample_loss = 0.0f;
        for(int j = 0; j < n_classes; j++) {
            float true_val = y_true.data->flex[i * n_classes + j];
            float pred_val = y_pred.data->flex[i * n_classes + j];
            float epsilon = 1e-8f; // avoid log(0) so we add a small epsilon
            if (true_val > 0) { // one-hot encoding
                sample_loss -= true_val * logf(pred_val + epsilon);
            }
        }
        total_loss += sample_loss;
    }
    
    res.data->flex[0] = total_loss / n_samples;
    
    if(requires_grad) {
        res.node->grad_fn = GradFn_crossentropy;
        res.node->inputs[0] = y_true;
        res.node->inputs[1] = y_pred;
        res.node->n_inputs = 2;
        res.node->name = "Cross-entropy";
        printf("Cross-entropy Grad_fn\n");
        Tensor_print(res.node->grad_fn(res, 1));    
    }

    return res;
}