#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef Tensor_mean
#undef Tensor_mean
#endif
#ifdef Tensor_sum
#undef Tensor_sum
#endif
#ifdef Tensor_max
#undef Tensor_max
#endif
#ifdef Tensor_min
#undef Tensor_min
#endif

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
    Tensor orig_self = self;
    Tensor orig_other = other;
    
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_add() cannot broadcast", orig_self.shape, orig_other.shape);
    }
    
    bool requires_grad = !cten_is_eval() && (orig_self.node != NULL || orig_other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] + other.data->flex[i];
    }
    
    if(requires_grad) {
        res.node->grad_fn = GradFn_add;
        res.node->inputs[0] = orig_self;
        res.node->inputs[1] = orig_other;
        res.node->n_inputs = 2;
        res.node->name = "Add";
    }
    return res;
}

Tensor Tensor_mul(Tensor self, Tensor other) {
    Tensor orig_self = self;
    Tensor orig_other = other;
    
    if(!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_mul() cannot broadcast", orig_self.shape, orig_other.shape);
    }
    
    bool requires_grad = !cten_is_eval() && (orig_self.node != NULL || orig_other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] * other.data->flex[i];
    }
    
    if(requires_grad) {
        res.node->grad_fn = GradFn_mul;
        res.node->inputs[0] = orig_self;
        res.node->inputs[1] = orig_other;
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

Tensor GradFn_mean(Tensor self, int i) {
    Tensor input_tensor = self.node->inputs[i];
    int divisor;
    
    if (TensorShape_numel(self.shape) == 1 && TensorShape_numel(input_tensor.shape) > 1) {
        divisor = TensorShape_numel(input_tensor.shape);
    } else {
        int input_ndim = TensorShape_dim(input_tensor.shape);
        int output_ndim = TensorShape_dim(self.shape);
        if (input_ndim > output_ndim) {
            int out_idx = 0;
            int reduced_dim_size = 1;
            for(int d=0; d < input_ndim; ++d) {
                if(out_idx >= output_ndim || input_tensor.shape[d] != self.shape[out_idx]) {
                    reduced_dim_size = input_tensor.shape[d];
                    break;
                }
                out_idx++;
            }
            divisor = reduced_dim_size;
        } else {
            // scalar input
            divisor = TensorShape_numel(input_tensor.shape);
        }
    }

    // gradient ==> SAME SHAPE as the ORIGINAL INPUT.
    Tensor res = Tensor_new(input_tensor.shape, false);
    
    // gradient value is 1 divided by the number of elements that were averaged.
    float grad_val = 1.0f / divisor;
    
    for(int j = 0; j < res.data->numel; j++) {
        res.data->flex[j] = grad_val;
    }   
    return res;
}

Tensor Tensor_mean(Tensor self, ...) {
    int ndim = TensorShape_dim(self.shape);
    int dim = INT_MIN; // Default value to trigger the "else" block
    
    va_list args;
    va_start(args, self);
    
    if (va_arg_is_present(args)) {
        dim = va_arg(args, int);
    }
    va_end(args);
    

    if (dim != INT_MIN) {
        Tensor res = Tensor_reduce_dim(self, dim, "mean");
        if(res.node != NULL) {
            res.node->grad_fn = GradFn_mean;
            res.node->inputs[0] = self;
            res.node->n_inputs = 1;
            res.node->name = "Mean";
        }
        return res;
    } else {
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

Tensor GradFn_sum(Tensor self, int i) {
    // f(x) = sum(x); f'(x) = 1
    return Tensor_ones(self.node->inputs[i].shape, false);
}

Tensor Tensor_sum(Tensor self, ...) {
    int ndim = TensorShape_dim(self.shape);
    int dim = INT_MIN; // Default value to trigger the "else" block
    
    va_list args;
    va_start(args, self);
    
    if (va_arg_is_present(args)) {
        dim = va_arg(args, int);
    }
    va_end(args);
    

    if (dim != INT_MIN) {
        Tensor res = Tensor_reduce_dim(self, dim, "sum");
        if(res.node != NULL) {
            res.node->grad_fn = GradFn_sum;
            res.node->inputs[0] = self;
            res.node->n_inputs = 1;
            res.node->name = "Sum";
        }
        return res;
    } else {
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

static Tensor GradFn_div(Tensor self, int i) {
    Tensor res = Tensor_new(self.shape, false);
    Tensor x = self.node->inputs[0];
    Tensor y = self.node->inputs[1];

    if (i == 0) { // Gradient w.r.t. x: 1/y
        for (int j = 0; j < res.data->numel; j++) {
            res.data->flex[j] = 1.0f / y.data->flex[j % y.data->numel];
        }
    } else { // Gradient w.r.t. y: -x/y²
        for (int j = 0; j < res.data->numel; j++) {
            float x_val = x.data->flex[j % x.data->numel];
            float y_val = y.data->flex[j % y.data->numel];
            res.data->flex[j] = -x_val / (y_val * y_val);
        }
    }
    return res;
}

Tensor Tensor_div(Tensor self, Tensor other) {
    Tensor orig_self = self;
    Tensor orig_other = other;

    if (!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_div() cannot broadcast", orig_self.shape, orig_other.shape);
    }
    bool requires_grad = !cten_is_eval() && (orig_self.node != NULL || orig_other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for (int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] / other.data->flex[i];
    }
    if (requires_grad) {
        res.node->grad_fn = GradFn_div;
        res.node->inputs[0] = orig_self;
        res.node->inputs[1] = orig_other;
        res.node->n_inputs = 2;
        res.node->name = "Div";
    }
    return res;
}

static Tensor GradFn_square(Tensor self, int i) {
    // f(x) = x²; f'(x) = 2x
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for (int j = 0; j < res.data->numel; j++) {
        res.data->flex[j] = 2.0f * input.data->flex[j];
    }
    return res;
}

Tensor Tensor_square(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for (int i = 0; i < self.data->numel; i++) {
        float val = self.data->flex[i];
        res.data->flex[i] = val * val;
    }
    if (requires_grad) {
        res.node->grad_fn = GradFn_square;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Square";
    }
    return res;
}

static Tensor GradFn_reciprocal(Tensor self, int i) {
    // f(x) = 1/x; f'(x) = -1/x^2
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for (int j = 0; j < res.data->numel; j++) {
        float x_val = input.data->flex[j];
        res.data->flex[j] = -1.0f / (x_val * x_val);
    }
    return res;
}

Tensor Tensor_reciprocal(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for (int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = 1.0f / self.data->flex[i];
    }
    if (requires_grad) {
        res.node->grad_fn = GradFn_reciprocal;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Reciprocal";
    }
    return res;
}

static Tensor GradFn_pow(Tensor self, int i) {
    // f(x, y) = x^y;  ∂f/∂x = y*x^(y-1);  ∂f/∂y = x^y * ln(x)
    Tensor res = Tensor_new(self.shape, false);
    Tensor x = self.node->inputs[0];
    Tensor y = self.node->inputs[1];
    
    if (i == 0) {
        // Gradient w.r.t. x: y*x^(y-1)
        for (int j = 0; j < res.data->numel; j++) {
            float x_val = x.data->flex[j % x.data->numel];
            float y_val = y.data->flex[j % y.data->numel];
            if (x_val == 0.0f && y_val > 1.0f) {
                res.data->flex[j] = 0.0f;
            } else {
                res.data->flex[j] = y_val * powf(x_val, y_val - 1.0f);
            }
        }
    } else {
        // Gradient w.r.t. y: x^y * ln(x)
        for (int j = 0; j < res.data->numel; j++) {
            float x_val = x.data->flex[j % x.data->numel];
            float self_val = self.data->flex[j];
            if (x_val <= 0.0f) {
                // Gradient of x^y w.r.t y is undefined or complex for x <= 0.
                // Returning 0 for simplicity, but this might need specific handling depending on use case.
                // For example, if x can be negative and y is an integer, the behavior is different.
                // If x is 0, and y > 0, derivative is 0. If x is 0 and y <= 0, it's undefined.
                // logf(negative) is NaN. powf(negative, non-integer) is complex.
                // We assume positive x for logf(x) to be real.
                // A robust solution might involve checking domain or returning NaN.
                res.data->flex[j] = 0.0f; 
            } else {
                res.data->flex[j] = self_val * logf(x_val);
            }
        }
    }
    return res;
}

Tensor Tensor_pow(Tensor self, Tensor other) {
    Tensor orig_self = self;
    Tensor orig_other = other;
    if (!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_pow() cannot broadcast", orig_self.shape, orig_other.shape);
    }
    bool requires_grad = !cten_is_eval() && (orig_self.node != NULL || orig_other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for (int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = powf(self.data->flex[i], other.data->flex[i]);
    }
    if (requires_grad) {
        res.node->grad_fn = GradFn_pow;
        res.node->inputs[0] = orig_self;
        res.node->inputs[1] = orig_other;
        res.node->n_inputs = 2;
        res.node->name = "Pow";
    }
    return res;
}

Tensor Tensor_sub(Tensor self, Tensor other) {
    Tensor orig_self = self;
    Tensor orig_other = other;
    if (!cten_elemwise_broadcast(&self, &other)) {
        cten_assert_shape("Tensor_sub() cannot broadcast", orig_self.shape, orig_other.shape);
    }
    bool requires_grad = !cten_is_eval() && (orig_self.node != NULL || orig_other.node != NULL);
    Tensor res = Tensor_new(self.shape, requires_grad);
    for (int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = self.data->flex[i] - other.data->flex[i];
    }
    if (requires_grad) {
        res.node->grad_fn = GradFn_sub;
        res.node->inputs[0] = orig_self;
        res.node->inputs[1] = orig_other;
        res.node->n_inputs = 2;
        res.node->name = "Sub";
    }
    return res;
}

Tensor GradFn_reduce_dim(Tensor self, int i) {
    Tensor input = self.node->inputs[0];
    Tensor indices_tensor = self.node->inputs[1];
    Tensor grad_out = Tensor_zeros(input.shape, false);

    int out_numel = indices_tensor.data->numel;
    int ndim = TensorShape_dim(input.shape);
    int reduced_dim = -1;

    for(int d = 0, out_d = 0; d < ndim; d++){
        if(out_d >= TensorShape_dim(self.shape) || input.shape[d] != self.shape[out_d]){
            reduced_dim = d;
            break;
        }
        out_d++;
    }
    cten_assert(reduced_dim != -1, "Could not determine reduced dimension in gradient calculation");
    
    for (int j = 0; j < out_numel; j++) {
        int index_along_dim = (int)indices_tensor.data->flex[j];
        
        int linear_idx = 0, stride = 1, out_j_rem = j, out_shape_idx = TensorShape_dim(self.shape) - 1;
        for (int k = ndim - 1; k >= 0; --k) {
            int current_dim_idx;
            if (k == reduced_dim) {
                current_dim_idx = index_along_dim;
            } else {
                int dim_k = self.shape[out_shape_idx--];
                current_dim_idx = out_j_rem % dim_k;
                out_j_rem /= dim_k;
            }
            linear_idx += current_dim_idx * stride;
            stride *= input.shape[k];
        }
        grad_out.data->flex[linear_idx] = 1.0f;
    }
    return grad_out;
}

Tensor GradFn_max_all(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_zeros(input.shape, false);
    float max_val = self.data->flex[0];
    
    int max_count = 0;
    for (int j = 0; j < input.data->numel; j++) {
        if (input.data->flex[j] == max_val) max_count++;
    }
    
    float grad_value = (max_count > 0) ? 1.0f / max_count : 0.0f;
    for (int j = 0; j < input.data->numel; j++) {
        if (input.data->flex[j] == max_val) res.data->flex[j] = grad_value;
    }
    return res;
}

Tensor Tensor_max(Tensor self) {
    if (self.data->numel == 0){
        cten_assert(false, "Error: max() on an empty tensor.");
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new((TensorShape){1, 0, 0, 0}, requires_grad);
    
    float max_val = self.data->flex[0];
    for (int i = 1; i < self.data->numel; i++) {
        if (self.data->flex[i] > max_val) {
            max_val = self.data->flex[i];
        }
    }
    
    res.data->flex[0] = max_val;
    
    if (requires_grad) {
        res.node->grad_fn = GradFn_max_all;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "MaxAll";
    }
    
    return res;
}

Tensor GradFn_min_all(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_zeros(input.shape, false);
    float min_val = self.data->flex[0];
    
    int min_count = 0;
    for (int j = 0; j < input.data->numel; j++) {
        if (input.data->flex[j] == min_val) min_count++;
    }
    
    float grad_value = (min_count > 0) ? 1.0f / min_count : 0.0f;
    for (int j = 0; j < input.data->numel; j++) {
        if (input.data->flex[j] == min_val) res.data->flex[j] = grad_value;
    }
    return res;
}

Tensor Tensor_min(Tensor self) {
    if (self.data->numel == 0){
        cten_assert(false, "Error: min() on an empty tensor.");
    }
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new((TensorShape){1, 0, 0, 0}, requires_grad);
    
    // Find minimum value
    float min_val = self.data->flex[0];
    for (int i = 1; i < self.data->numel; i++) {
        if (self.data->flex[i] < min_val) {
            min_val = self.data->flex[i];
        }
    }
    
    res.data->flex[0] = min_val;
    
    if (requires_grad) {
        res.node->grad_fn = GradFn_min_all;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "MinAll";
    }
    
    return res;
}

static Tensor GradFn_abs(Tensor self, int i) {
    Tensor input = self.node->inputs[i];
    Tensor res = Tensor_new(input.shape, false);
    for(int j = 0; j < input.data->numel; j++) {
        float val = input.data->flex[j];
        if (val > 0) {
            res.data->flex[j] = 1.0f;
        } else if (val < 0) {
            res.data->flex[j] = -1.0f;
        } else {
            res.data->flex[j] = 0.0f;
        }
    }
    return res;
}

Tensor Tensor_abs(Tensor self) {
    bool requires_grad = !cten_is_eval() && self.node != NULL;
    Tensor res = Tensor_new(self.shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        res.data->flex[i] = fabsf(self.data->flex[i]);
    }

    if(requires_grad) {
        res.node->grad_fn = GradFn_abs;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Abs";
    }
    return res;
}