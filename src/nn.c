#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

static float elu_alpha_value = 1.0f;
static float huber_delta_value = 1.0f;

Tensor nn_linear(Tensor input, Tensor weight, Tensor bias) {
    Tensor tmp = Tensor_matmul(input, weight);
    tmp = Tensor_add(tmp, bias);
    return tmp;
}

static Tensor GradFn_relu(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int i = 0; i < input.data->numel; i++) {
        res.data->flex[i] = input.data->flex[i] > 0 ? 1.0f : 0.0f;
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

static Tensor GradFn_log(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int j = 0; j < input.data->numel; j++) {
        res.data->flex[j] = 1.0f / input.data->flex[j];
    }
    return res;
}

Tensor nn_log(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = logf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_log;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Log";
    }
    return res;
}

static Tensor GradFn_exp(Tensor self, int i) {
    return self;
}

Tensor nn_exp(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = expf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_exp;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Exp";
    }
    return res;
}

static Tensor GradFn_sin(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int j = 0; j < input.data->numel; j++) {
        res.data->flex[j] = cosf(input.data->flex[j]);
    }
    return res;
}

Tensor nn_sin(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = sinf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_sin;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Sin";
    }
    return res;
}

static Tensor GradFn_cos(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int j = 0; j < input.data->numel; j++) {
        res.data->flex[j] = -sinf(input.data->flex[j]);
    }
    return res;
}

Tensor nn_cos(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = cosf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_cos;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Cos";
    }
    return res;
}

static Tensor GradFn_tan(Tensor self, int i) {
    // d/dx(tan(x)) = 1 + tan^2(x)
    Tensor res = Tensor_new(self.shape, false);
    for(int j = 0; j < self.data->numel; j++) {
        float y = self.data->flex[j];
        res.data->flex[j] = 1.0f + y*y;
    }
    return res;
}

Tensor nn_tan(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = tanf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_tan;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Tan";
    }
    return res;
}

static Tensor GradFn_sigmoid(Tensor self, int i) {
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    Tensor res = Tensor_new(self.shape, false);
    for(int j = 0; j < self.data->numel; j++) {
        float y = self.data->flex[j];
        res.data->flex[j] = y * (1.0f - y);
    }
    return res;
}

Tensor nn_sigmoid(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = 1.0f / (1.0f + expf(-self.data->flex[i]));
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_sigmoid;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Sigmoid";
    }
    return res;
}

static Tensor GradFn_tanh(Tensor self, int i) {
    // d/dx tanh(x) = 1 - tanh^2(x)
    Tensor res = Tensor_new(self.shape, false);
    for(int j = 0; j < self.data->numel; j++) {
        float y = self.data->flex[j];
        res.data->flex[j] = 1.0f - y*y;
    }
    return res;
}

Tensor nn_tanh(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = tanhf(self.data->flex[i]);
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_tanh;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Tanh";
    }
    return res;
}

static Tensor GradFn_elu(Tensor self, int i) {
    float alpha = elu_alpha_value;
    Tensor input = self.node->inputs[0];
    Tensor grad = Tensor_new(input.shape, false);
    for(int j = 0; j < input.data->numel; j++) {
        float x = input.data->flex[j];
        if (x > 0) {
            grad.data->flex[j] = 1.0f;
        } else {
            // derivative is alpha * e^x = alpha * (e^x - 1) + alpha = y + alpha
            grad.data->flex[j] = self.data->flex[j] + alpha;
        }
    }
    return grad;
}

Tensor nn_elu(Tensor self, float alpha) {
    elu_alpha_value = alpha;
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        float x = self.data->flex[i];
        if (x > 0) {
            res.data->flex[i] = x;
        } else {
            res.data->flex[i] = alpha * (expf(x) - 1.0f);
        }
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_elu;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Elu";
    }
    return res;
}

static Tensor GradFn_selu(Tensor self, int i) {
    Tensor input = self.node->inputs[0];
    Tensor grad = Tensor_new(input.shape, false);
    const float alpha = 1.67326324f;
    const float lambda = 1.05070098f;
    for(int j = 0; j < input.data->numel; j++) {
        float x = input.data->flex[j];
        if (x > 0) {
            grad.data->flex[j] = lambda;
        } else {
            // derivative is lambda * alpha * e^x = y + lambda*alpha
            grad.data->flex[j] = self.data->flex[j] + lambda * alpha;
        }
    }
    return grad;
}

Tensor nn_selu(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    const float alpha = 1.67326324f;
    const float lambda = 1.05070098f;
    for(int i = 0; i < self.data->numel; i++) {
        float x = self.data->flex[i];
        if (x > 0) {
            res.data->flex[i] = lambda * x;
        } else {
            res.data->flex[i] = lambda * alpha * (expf(x) - 1);
        }
    }
    if(requires_grad) {
        res.node->grad_fn = GradFn_selu;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Selu";
    }
    return res;
}

Tensor Glorot_init(TensorShape shape, bool requires_grad) {
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

static Tensor GradFn_crossentropy(Tensor self, int i) {
    if (i == 1) { // Gradient w.r.t. y_pred
        Tensor y_true = self.node->inputs[0];
        Tensor y_pred = self.node->inputs[1];
        int n_samples = y_true.shape[0];
        int n_classes = y_true.shape[1];
        
        Tensor grad = Tensor_new(y_pred.shape, false);
        
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                float y_true_val = y_true.data->flex[i * n_classes + j];
                float y_pred_val = y_pred.data->flex[i * n_classes + j];
                if (y_true_val == 0) {
                    grad.data->flex[i * n_classes + j] = 0;
                } else {
                    grad.data->flex[i * n_classes + j] = -y_true_val / y_pred_val;
                }
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
    }

    return res;
}

static Tensor GradFn_softmax_crossentropy(Tensor self, int i) {
    if (i == 1) {
        Tensor y_true = self.node->inputs[0];
        Tensor logits = self.node->inputs[1];
        
        Tensor y_pred = Tensor_new(logits.shape, false);
        int self_dim = TensorShape_dim(logits.shape);
        int last_dim_size = logits.shape[self_dim - 1];
        int outer_size = logits.data->numel / last_dim_size;

        for(int outer = 0; outer < outer_size; outer++) {
            float max_val = -INFINITY;
            float sum = 0;

            for(int d = 0; d < last_dim_size; d++) {
                int index = outer * last_dim_size + d;
                max_val = fmaxf(max_val, logits.data->flex[index]);
            }

            for(int d = 0; d < last_dim_size; d++) {
                int index = outer * last_dim_size + d;
                y_pred.data->flex[index] = expf(logits.data->flex[index] - max_val);
                sum += y_pred.data->flex[index];
            }

            for(int d = 0; d < last_dim_size; d++) {
                int index = outer * last_dim_size + d;
                y_pred.data->flex[index] /= sum;
            }
        }
        
        Tensor grad = Tensor_new(y_pred.shape, false);
        int n_samples = y_pred.shape[0];
        int n_classes = y_pred.shape[1];
        
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                grad.data->flex[i * n_classes + j] = 
                    y_pred.data->flex[i * n_classes + j] - y_true.data->flex[i * n_classes + j];
            }
        }
        
        return grad;
    }
    return Tensor_zeros((TensorShape){1}, false);
}

Tensor nn_softmax_crossentropy(Tensor y_true, Tensor logits) {
    bool requires_grad = !cten_is_eval() && logits.node != NULL;
    //disable gradient computation
    cten_begin_eval(); 
    Tensor y_pred = nn_softmax(logits);
    Tensor loss = nn_crossentropy(y_true, y_pred);
    cten_end_eval();
    Tensor res = Tensor_zeros((TensorShape){1}, requires_grad);
    res.data->flex[0] = loss.data->flex[0];
    
    if(requires_grad) {
        res.node->grad_fn = GradFn_softmax_crossentropy;
        res.node->inputs[0] = y_true;
        res.node->inputs[1] = logits;
        res.node->n_inputs = 2;
        res.node->name = "SoftmaxCrossEntropy"; 
    }
    
    return res;
}

static Tensor GradFn_mse_loss(Tensor self, int i) {
    if (i == 1) {  // Gradient w.r.t y_pred
        Tensor y_true = self.node->inputs[0];
        Tensor y_pred = self.node->inputs[1];
        int n = y_pred.data->numel;

        Tensor grad = Tensor_new(y_pred.shape, false);
        for (int j = 0; j < n; j++) {
            grad.data->flex[j] = 2.0f * (y_pred.data->flex[j] - y_true.data->flex[j]) / n;
        }
        return grad;
    }
    return Tensor_zeros((TensorShape){1}, false);
}

Tensor nn_mse_loss(Tensor y_true, Tensor y_pred) {
    bool requires_grad = !cten_is_eval() && y_pred.node != NULL;

    cten_begin_eval();
    Tensor error = Tensor_sub(y_pred, y_true);
    Tensor squared_error = Tensor_square(error);
    Tensor loss = Tensor_mean(squared_error);
    cten_end_eval();

    Tensor res = Tensor_new((TensorShape){1}, requires_grad);
    res.data->flex[0] = loss.data->flex[0];

    if (requires_grad) {
        res.node->grad_fn = GradFn_mse_loss;
        res.node->inputs[0] = y_true;
        res.node->inputs[1] = y_pred;
        res.node->n_inputs = 2;
        res.node->name = "MSELoss";
    }
    return res;
}

static Tensor GradFn_mae_loss(Tensor self, int i) {
    if (i == 1) { // Gradient w.r.t y_pred
        Tensor y_true = self.node->inputs[0];
        Tensor y_pred = self.node->inputs[1];
        int n = y_pred.data->numel;

        Tensor grad = Tensor_new(y_pred.shape, false);
        for (int j = 0; j < n; j++) {
            float error = y_pred.data->flex[j] - y_true.data->flex[j];
            if (error > 0) {
                grad.data->flex[j] = 1.0f / n;
            } else if (error < 0) {
                grad.data->flex[j] = -1.0f / n;
            } else {
                grad.data->flex[j] = 0.0f;
            }
        }
        return grad;
    }
    return Tensor_zeros((TensorShape){1}, false);
}

Tensor nn_mae_loss(Tensor y_true, Tensor y_pred) {
    bool requires_grad = !cten_is_eval() && y_pred.node != NULL;

    cten_begin_eval();
    Tensor error = Tensor_sub(y_pred, y_true);
    Tensor abs_error = Tensor_abs(error);
    Tensor loss = Tensor_mean(abs_error);
    cten_end_eval();

    Tensor res = Tensor_new((TensorShape){1}, requires_grad);
    res.data->flex[0] = loss.data->flex[0];

    if (requires_grad) {
        res.node->grad_fn = GradFn_mae_loss;
        res.node->inputs[0] = y_true;
        res.node->inputs[1] = y_pred;
        res.node->n_inputs = 2;
        res.node->name = "MAELoss";
    }
    return res;
}

static Tensor GradFn_huber_loss(Tensor self, int i) {
    if (i == 1) { // Gradient w.r.t y_pred
        Tensor y_true = self.node->inputs[0];
        Tensor y_pred = self.node->inputs[1];
        float delta = huber_delta_value;
        int n = y_pred.data->numel;

        Tensor grad = Tensor_new(y_pred.shape, false);
        // Gradient of Huber loss is (error / n) for small errors,
        // and (delta * sign(error) / n) for large errors.
        for (int j = 0; j < n; j++) {
            float error = y_pred.data->flex[j] - y_true.data->flex[j];
            if (fabsf(error) <= delta) {
                grad.data->flex[j] = error / n;
            } else {
                if (error > 0) {
                    grad.data->flex[j] = delta / n;
                } else {
                    grad.data->flex[j] = -delta / n;
                }
            }
        }
        return grad;
    }
    return Tensor_zeros((TensorShape){1}, false);
}

Tensor nn_huber_loss(Tensor y_true, Tensor y_pred, float delta) {
    huber_delta_value = delta; // Store delta for the backward pass
    bool requires_grad = !cten_is_eval() && y_pred.node != NULL;

    int n = y_pred.data->numel;
    float total_loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = y_pred.data->flex[i] - y_true.data->flex[i];
        float abs_error = fabsf(error);
        if (abs_error <= delta) {
            total_loss += 0.5f * error * error; // MSE part
        } else {
            total_loss += delta * (abs_error - 0.5f * delta); // MAE part
        }
    }

    Tensor res = Tensor_new((TensorShape){1}, requires_grad);
    res.data->flex[0] = total_loss / n; // Mean Huber Loss

    if (requires_grad) {
        res.node->grad_fn = GradFn_huber_loss;
        res.node->inputs[0] = y_true;
        res.node->inputs[1] = y_pred;
        res.node->n_inputs = 2;
        res.node->name = "HuberLoss";
    }
    return res;
}