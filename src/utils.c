#include "cten.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    
#include <time.h>    
#include <limits.h>

bool va_arg_is_present(va_list args) {
    (void)args;
    return false;
}

Tensor GradFn_mean(Tensor self, int i);
Tensor GradFn_sum(Tensor self, int i);

Tensor Tensor_mean_all(Tensor self) {
    float total = 0.0f;
    for(int i = 0; i < self.data->numel; i++) total += self.data->flex[i];
    Tensor res = Tensor_new((TensorShape){1,0,0,0}, self.node != NULL);
    res.data->flex[0] = total / self.data->numel;
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_mean;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Mean";
    }
    return res;
}

Tensor Tensor_mean_dim(Tensor self, int dim) {
    Tensor res = Tensor_reduce_dim(self, dim, "mean");
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_mean;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Mean";
    }
    return res;
}

Tensor Tensor_sum_all(Tensor self) {
    float total = 0.0f;
    for(int i = 0; i < self.data->numel; i++) total += self.data->flex[i];
    Tensor res = Tensor_new((TensorShape){1,0,0,0}, self.node != NULL);
    res.data->flex[0] = total;
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_sum;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Sum";
    }
    return res;
}

Tensor Tensor_sum_dim(Tensor self, int dim) {
    Tensor res = Tensor_reduce_dim(self, dim, "sum");
    if(res.node != NULL) {
        res.node->grad_fn = GradFn_sum;
        res.node->inputs[0] = self;
        res.node->n_inputs = 1;
        res.node->name = "Sum";
    }
    return res;
}

void cten_assert(bool cond, const char* fmt, ...) {
    if(!cond) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
    }
}

void cten_assert_shape(const char* title, TensorShape a, TensorShape b) {
    bool cond = memcmp(a, b, sizeof(TensorShape)) == 0;
    char buf_a[64];
    char buf_b[64];
    TensorShape_tostring(a, buf_a, sizeof(buf_a));
    TensorShape_tostring(b, buf_b, sizeof(buf_b));
    cten_assert(cond, "%s: %s != %s", title, buf_a, buf_b);
}

void cten_assert_dim(const char* title, int a, int b) {
    cten_assert(a == b, "%s: %d != %d", title, a, b);
}

bool cten_elemwise_broadcast(Tensor* a, Tensor* b) {
    Tensor orig_a = *a;
    Tensor orig_b = *b;

    // 1. Determine the result shape from the two input shapes
    TensorShape result_shape;
    for (int i = 0; i < 4; i++) {
        int a_dim = orig_a.shape[i];
        int b_dim = orig_b.shape[i];

        if (a_dim == 0 && b_dim == 0) result_shape[i] = 0;
        else if (a_dim == 0) result_shape[i] = b_dim;
        else if (b_dim == 0) result_shape[i] = a_dim;
        else if (a_dim == 1) result_shape[i] = b_dim;
        else if (b_dim == 1) result_shape[i] = a_dim;
        else if (a_dim == b_dim) result_shape[i] = a_dim;
        else {
            return false;
        }
    }

    // 2. Check if tensor 'a' needs to be expanded
    if (memcmp(orig_a.shape, result_shape, sizeof(TensorShape)) != 0) {
        Tensor new_a = Tensor_new(result_shape, orig_a.node != NULL);
        for (int i = 0; i < new_a.data->numel; i++) {
            int rem = i;
            int idx[4] = {0, 0, 0, 0};
            for (int dim = 3; dim >= 0; dim--) {
                if (result_shape[dim] > 0) {
                    idx[dim] = rem % result_shape[dim];
                    rem /= result_shape[dim];
                }
            }
            int source_idx = 0;
            int stride = 1;
            for (int dim = 3; dim >= 0; dim--) {
                if (orig_a.shape[dim] > 0) {
                    int dim_idx = (idx[dim] % orig_a.shape[dim]);
                    source_idx += dim_idx * stride;
                    stride *= orig_a.shape[dim];
                }
            }
            new_a.data->flex[i] = orig_a.data->flex[source_idx];
        }
        *a = new_a;
    }

    // 3. Check if tensor 'b' needs to be expanded
    if (memcmp(orig_b.shape, result_shape, sizeof(TensorShape)) != 0) {
        Tensor new_b = Tensor_new(result_shape, orig_b.node != NULL);
        for (int i = 0; i < new_b.data->numel; i++) {
            int rem = i;
            int idx[4] = {0, 0, 0, 0};
            for (int dim = 3; dim >= 0; dim--) {
                if (result_shape[dim] > 0) {
                    idx[dim] = rem % result_shape[dim];
                    rem /= result_shape[dim];
                }
            }
            int source_idx = 0;
            int stride = 1;
            for (int dim = 3; dim >= 0; dim--) {
                if (orig_b.shape[dim] > 0) {
                    int dim_idx = (idx[dim] % orig_b.shape[dim]);
                    source_idx += dim_idx * stride;
                    stride *= orig_b.shape[dim];
                }
            }
            new_b.data->flex[i] = orig_b.data->flex[source_idx];
        }
        *b = new_b;
    }
    return true;
}

Tensor reduce_gradient_for_broadcasting(Tensor grad, TensorShape original_shape, TensorShape broadcasted_shape) {
    Tensor result = grad;
    
    for (int dim = 3; dim >= 0; dim--) {
        int orig_size = original_shape[dim];
        int broad_size = broadcasted_shape[dim];
        int grad_size = result.shape[dim];
        
        // Case 1: dim was broadcasted from size 1 to size N
        if (orig_size == 1 && broad_size > 1 && grad_size == broad_size) {
            Tensor summed = Tensor_sum(result, dim);  
            TensorShape new_shape = {result.shape[0], result.shape[1], result.shape[2], result.shape[3]};
            new_shape[dim] = 1;  
            result = Tensor_new(new_shape, false);
            
            if (summed.data->numel == 1) {
                for (int i = 0; i < result.data->numel; i++) {
                    result.data->flex[i] = summed.data->flex[0];
                }
            } else {
                for (int i = 0; i < result.data->numel && i < summed.data->numel; i++) {
                    result.data->flex[i] = summed.data->flex[i];
                }
            }
        }
        // Case 2: dim was added (original was 0, broadcasted > 0) 
        else if (orig_size == 0 && broad_size > 0 && grad_size == broad_size) {
            Tensor summed = Tensor_sum(result, dim);
            TensorShape new_shape = {result.shape[0], result.shape[1], result.shape[2], result.shape[3]};
            new_shape[dim] = 0;
            for (int d = dim; d < 3; d++) {
                if (d + 1 < 4) {
                    new_shape[d] = new_shape[d + 1];
                }
            }
            new_shape[3] = 0; //clearing last dim
            result = Tensor_new(new_shape, false);
            for (int i = 0; i < result.data->numel && i < summed.data->numel; i++) {
                result.data->flex[i] = summed.data->flex[i];
            }
        }
        // Case 3: no broadcasting on this dim  
        else if (orig_size == broad_size && grad_size == broad_size) {
            //do nothing
        }
        else {
            //have to think about this
            cten_assert(false, "reduce_gradient_for_broadcasting: unexpected broadcasting pattern");
        }
    }
    return result;
}

void Tensor_normalize_dataset(const float (*X)[4], float (*X_norm)[4], int n_samples, int n_train_samples, int n_features) {
    float mean[4] = {0}, std[4] = {0};
    
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            mean[j] += X[i][j];
        }
    }
    for (int j = 0; j < n_features; j++) {
        mean[j] /= n_train_samples;
    }
    
    for (int i = 0; i < n_train_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            std[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
    }
    for (int j = 0; j < n_features; j++) {
        std[j] = sqrtf(std[j] / n_train_samples);
        // Avoid division by zero
        if (std[j] == 0) std[j] = 1.0f;
    }

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X_norm[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }
}

void Tensor_shuffle_dataset(const float (*X)[4], const int *y,float (*X_shuffled)[4], int *y_shuffled, int n_samples, int n_features) {
    int* indices = malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    srand((unsigned)time(NULL));
    for (int i = n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    for (int i = 0; i < n_samples; i++) {
        int idx = indices[i];
        for (int j = 0; j < n_features; j++) {
            X_shuffled[i][j] = X[idx][j];
        }
        y_shuffled[i] = y[idx];
    }
    
    free(indices);
}

Tensor Tensor_reduce_dim(Tensor self, int dim, const char* operation) {
    int ndim = TensorShape_dim(self.shape);
    if (dim < 0){
        if (dim < -ndim) {
            printf("dim %d out of range", dim);
            exit(-1);
        }
        dim += ndim;
    }
    if (dim >= ndim) {
        printf("dim %d out of range", dim);
        exit(-1);
    }
    
    TensorShape out_shape = {0, 0, 0, 0};
    int out_idx = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_shape[out_idx++] = self.shape[i];
        }
    }
    
    int dim_size = self.shape[dim];
    Tensor res = Tensor_zeros(out_shape, self.node != NULL);
    
    int total_out_elements = res.data->numel;
    
    for (int out_i = 0; out_i < total_out_elements; out_i++) {
        int out_indices[4] = {0};
        int remaining = out_i;
        for (int j = out_idx - 1; j >= 0; j--) {
            out_indices[j] = remaining % out_shape[j];
            remaining /= out_shape[j];
        }
        
        for (int d = 0; d < dim_size; d++) {
            int in_indices[4] = {0};
            int out_pos = 0;
            for (int j = 0; j < ndim; j++) {
                if (j == dim) {
                    in_indices[j] = d;
                } else {
                    in_indices[j] = out_indices[out_pos++];
                }
            }
            
            int in_linear = 0;
            int stride = 1;
            for (int j = ndim - 1; j >= 0; j--) {
                in_linear += in_indices[j] * stride;
                stride *= self.shape[j];
            }
            
            res.data->flex[out_i] += self.data->flex[in_linear];
        }
        
        if (strcmp(operation, "mean") == 0) {
            res.data->flex[out_i] /= dim_size;
        }
    }
    
    return res;
}