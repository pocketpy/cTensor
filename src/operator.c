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

Tensor Tensor_mean_dim(Tensor self, int dim) {
    int ndim = TensorShape_dim(self.shape);
    if (dim < 0 || dim >= ndim) {
        return self; // Return original tensor to avoid crash
    }
    
    TensorShape out_shape = {0, 0, 0, 0};
    int dim_size = self.shape[dim];
    int out_dim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_shape[out_dim++] = self.shape[i];
        }
    }
    
    Tensor res = Tensor_new(out_shape, self.node != NULL);
    

    int total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            total_elements *= self.shape[i];
        }
    }
    
    // Initialize result tensor with zeros
    res = Tensor_zeros(out_shape, false);
    
    // Calculate mean along the specified dimension
    for (int i = 0; i < self.data->numel; i++) {
        // Calculate the multi-dimensional index for this element
        int remaining = i;
        int indices[4] = {0, 0, 0, 0};
        int stride = self.data->numel;
        for (int j = 0; j < ndim; j++) {
            stride /= self.shape[j];
            indices[j] = remaining / stride;
            remaining %= stride;
        }
        
        int out_idx = 0;
        int out_dim_idx = 0;
        int out_stride = 1;
        for (int j = ndim - 1; j >= 0; j--) {
            if (j != dim) {
                out_idx += indices[j] * out_stride;
                out_stride *= out_shape[out_dim_idx++];
            }
        }
        
        res.data->flex[out_idx] += self.data->flex[i];
    }
    
    for (int i = 0; i < res.data->numel; i++) {
        res.data->flex[i] /= dim_size;
    }
    
    if (res.node != NULL) {
        res.node->grad_fn = GradFn_mean;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "MeanDim";
    }
    
    return res;
}

Tensor Tensor_mean(Tensor self) {
    int ndim = TensorShape_dim(self.shape);
    if (ndim == 3) {
        return Tensor_mean_dim(self, 1);
    } else if (ndim == 4) {
        return Tensor_mean_dim(self, 2);
    } else {
        // Default behavior for other cases - reduce to scalar
        Tensor res = Tensor_new((TensorShape){1, 0, 0, 0}, self.node != NULL);
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
}
static Tensor GradFn_sum(Tensor self, int i) {
    // f(x) = sum(x); f'(x) = 1
    return Tensor_ones(self.node->inputs[i].shape, false);
}

Tensor Tensor_sum_dim(Tensor self, int dim) {
    int ndim = TensorShape_dim(self.shape);
    if (dim < 0 || dim >= ndim) {
        return self; // Return original tensor to avoid crash
    }
    
    TensorShape out_shape = {0, 0, 0, 0};
    int dim_size = self.shape[dim];
    int out_dim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_shape[out_dim++] = self.shape[i];
        }
    }
    
    Tensor res = Tensor_zeros(out_shape, false);
    
    // Calculate sum along the specified dimension
    for (int i = 0; i < self.data->numel; i++) {
        // Calculate the multi-dimensional index for this element
        int remaining = i;
        int indices[4] = {0, 0, 0, 0};
        int stride = self.data->numel;
        for (int j = 0; j < ndim; j++) {
            stride /= self.shape[j];
            indices[j] = remaining / stride;
            remaining %= stride;
        }
        
        // Calculate the corresponding index in the output tensor
        int out_idx = 0;
        int out_dim_idx = 0;
        int out_stride = 1;
        for (int j = ndim - 1; j >= 0; j--) {
            if (j != dim) {
                out_idx += indices[j] * out_stride;
                out_stride *= out_shape[out_dim_idx++];
            }
        }
        
        // Add to the accumulator
        res.data->flex[out_idx] += self.data->flex[i];
    }
    
    if (res.node != NULL) {
        res.node->grad_fn = GradFn_sum; // We still use the same grad function
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Sum";
    }
    return res;
}

Tensor Tensor_sum(Tensor self) {
    int ndim = TensorShape_dim(self.shape);
    
    if (ndim == 3) {
        return Tensor_sum_dim(self, 1); 
    }
    else if (ndim == 4) {
        return Tensor_sum_dim(self, 2);
    }
    // Default case: sum all elements (scalar result)
    else {
        Tensor res = Tensor_new((TensorShape){1, 0, 0, 0}, self.node != NULL);
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
}

static Tensor GradFn_matmul(Tensor self, int i) {
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