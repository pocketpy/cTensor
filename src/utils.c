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
    int a_dim = TensorShape_dim(a->shape);
    int b_dim = TensorShape_dim(b->shape);
    
    if (a_dim == 1 && a->shape[0] == 1 && b_dim > 0) {
        Tensor a_ = Tensor_new(b->shape, a->node != NULL);
        float scalar_value = a->data->flex[0];
        int total_elements = TensorShape_numel(b->shape);
        for (int i = 0; i < total_elements; i++) {
            a_.data->flex[i] = scalar_value;
        }
        *a = a_;
        return true;
    }
    
    if (b_dim == 1 && b->shape[0] == 1 && a_dim > 0) {
        Tensor b_ = Tensor_new(a->shape, b->node != NULL);
        float scalar_value = b->data->flex[0];
        int total_elements = TensorShape_numel(a->shape);
        for (int i = 0; i < total_elements; i++) {
            b_.data->flex[i] = scalar_value;
        }
        *b = b_;
        return true;
    }
    
    if (a_dim != b_dim) return false;
    int a_broadcast = -1;
    for(int i = 0; i < a_dim; i++) {
        if(a->shape[i] == b->shape[i]) continue;
        if(a->shape[i] == 1) {
            if(a_broadcast == 0) return false;
            a_broadcast = 1;
        } else if(b->shape[i] == 1) {
            if(a_broadcast == 1) return false;
            a_broadcast = 0;
        } else {
            return false;
        }
    }
    if(a_broadcast != -1) {
        if(a_broadcast == 0) {
            Tensor* tmp = a;
            a = b;
            b = tmp;
            a_broadcast = 1;
        }
        Tensor a_ = Tensor_new(b->shape, a->node != NULL);
        for(int i = 0; i < a_.shape[0]; i++) {
            int i_ = a->shape[0] == 1 ? 0 : i;
            for(int j = 0; j < a_.shape[1]; j++) {
                int j_ = a->shape[1] == 1 ? 0 : j;
                for(int k = 0; k < a_.shape[2]; k++) {
                    int k_ = a->shape[2] == 1 ? 0 : k;
                    for(int l = 0; l < a_.shape[3]; l++) {
                        int l_ = a->shape[3] == 1 ? 0 : l;
                        // a_[i][j][k][l] = a[i_][j_][k_][l_]
                        a_.data->flex[i * a_.shape[1] * a_.shape[2] * a_.shape[3] +
                                      j * a_.shape[2] * a_.shape[3] + k * a_.shape[3] + l] =
                            a->data->flex[i_ * a->shape[1] * a->shape[2] * a->shape[3] +
                                          j_ * a->shape[2] * a->shape[3] + k_ * a->shape[3] + l_];
                    }
                }
            }
        }
        *a = a_;
    }
    return true;
}

Tensor reduce_gradient_for_broadcasting(Tensor grad, TensorShape original_shape, TensorShape broadcasted_shape) {
    printf("    - reduce_gradient_for_broadcasting called:\n");
    printf("      - Original shape: [%d,%d,%d,%d]\n", 
           original_shape[0], original_shape[1], original_shape[2], original_shape[3]);
    printf("      - Broadcasted shape: [%d,%d,%d,%d]\n", 
           broadcasted_shape[0], broadcasted_shape[1], broadcasted_shape[2], broadcasted_shape[3]);
    printf("      - Gradient shape: [%d,%d,%d,%d]\n", 
           grad.shape[0], grad.shape[1], grad.shape[2], grad.shape[3]);
    
    Tensor result = grad;  // Start with the full gradient
    
    // Handle each dimension from right to left (following NumPy broadcasting rules)
    for (int dim = 3; dim >= 0; dim--) {
        int orig_size = original_shape[dim];
        int broad_size = broadcasted_shape[dim];
        int grad_size = result.shape[dim];
        
        printf("      - Dim %d: orig=%d, broad=%d, grad=%d\n", dim, orig_size, broad_size, grad_size);
        
        // Case 1: Dimension was broadcasted from size 1 to size N
        if (orig_size == 1 && broad_size > 1 && grad_size == broad_size) {
            printf("        - Reducing dimension %d (was broadcasted from 1 to %d)\n", dim, broad_size);
            
            // Sum along this dimension
            Tensor summed = Tensor_sum(result, dim);  // This gives us a scalar or reduced tensor
            printf("        - Summed result: "); Tensor_print(summed); printf("\n");
            
            // Now we need to reshape to have size 1 in the reduced dimension
            // and preserve other dimensions
            TensorShape new_shape = {result.shape[0], result.shape[1], result.shape[2], result.shape[3]};
            new_shape[dim] = 1;  // Set the reduced dimension to 1
            
            printf("        - Target shape after reduction: [%d,%d,%d,%d]\n", 
                   new_shape[0], new_shape[1], new_shape[2], new_shape[3]);
            
            // Create new tensor with the correct shape
            result = Tensor_new(new_shape, false);
            
            // Copy the summed data
            if (summed.data->numel == 1) {
                // If summed is a scalar, replicate it to fill the new shape
                for (int i = 0; i < result.data->numel; i++) {
                    result.data->flex[i] = summed.data->flex[0];
                }
            } else {
                // If summed has multiple elements, copy them
                for (int i = 0; i < result.data->numel && i < summed.data->numel; i++) {
                    result.data->flex[i] = summed.data->flex[i];
                }
            }
            
            printf("        - After reshaping: "); Tensor_print(result); 
            printf(" (shape=[%d,%d,%d,%d])\n", result.shape[0], result.shape[1], result.shape[2], result.shape[3]);
        }
        // Case 2: Dimension was added (original was 0, broadcasted > 0) 
        else if (orig_size == 0 && broad_size > 0 && grad_size == broad_size) {
            printf("        - Reducing dimension %d (was added, need to remove)\n", dim);
            
            // Sum along this dimension to remove it
            Tensor summed = Tensor_sum(result, dim);
            printf("        - Summed result: "); Tensor_print(summed); printf("\n");
            
            // Create new shape with this dimension removed (set to 0)
            TensorShape new_shape = {result.shape[0], result.shape[1], result.shape[2], result.shape[3]};
            new_shape[dim] = 0;
            
            // Shift dimensions after the removed one
            for (int d = dim; d < 3; d++) {
                if (d + 1 < 4) {
                    new_shape[d] = new_shape[d + 1];
                }
            }
            new_shape[3] = 0;  // Clear the last dimension
            
            printf("        - Target shape after removal: [%d,%d,%d,%d]\n", 
                   new_shape[0], new_shape[1], new_shape[2], new_shape[3]);
            
            result = Tensor_new(new_shape, false);
            
            // Copy the summed data
            for (int i = 0; i < result.data->numel && i < summed.data->numel; i++) {
                result.data->flex[i] = summed.data->flex[i];
            }
            
            printf("        - After dimension removal: "); Tensor_print(result); 
            printf(" (shape=[%d,%d,%d,%d])\n", result.shape[0], result.shape[1], result.shape[2], result.shape[3]);
        }
        // Case 3: No broadcasting on this dimension  
        else if (orig_size == broad_size && grad_size == broad_size) {
            printf("        - No broadcasting on dimension %d\n", dim);
            // No reduction needed
        }
        else {
            printf("        - WARNING: Unexpected broadcasting pattern on dim %d\n", dim);
        }
    }
    
    printf("      - Final result: "); Tensor_print(result); 
    printf(" (shape=[%d,%d,%d,%d])\n", result.shape[0], result.shape[1], result.shape[2], result.shape[3]);
    printf("      - Expected original shape: [%d,%d,%d,%d]\n", 
           original_shape[0], original_shape[1], original_shape[2], original_shape[3]);
    
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