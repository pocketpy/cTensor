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
Tensor GradFn_max_all(Tensor self, int i);
Tensor GradFn_min_all(Tensor self, int i);
Tensor GradFn_reduce_dim(Tensor self, int i);

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

Tensor Tensor_max_all(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new((TensorShape){1, 0, 0, 0}, requires_grad);
    
    if (self.data->numel == 0) cten_assert(false, "max on empty tensor");
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

TensorMaxMinResult Tensor_max_dim(Tensor self, int dim) {
    int ndim = TensorShape_dim(self.shape);
    dim = TensorShape_asdim(self.shape, dim);

    TensorShape out_shape = {0};
    int out_shape_len = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) out_shape[out_shape_len++] = self.shape[i];
    }
    
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor values = Tensor_new(out_shape, requires_grad);
    Tensor indices = Tensor_new(out_shape, false);

    int dim_size = self.shape[dim];
    for (int i = 0; i < values.data->numel; ++i) {
        float best_val = -INFINITY;
        int best_idx = -1;

        for (int j = 0; j < dim_size; ++j) {
            int in_linear_idx = 0, stride = 1, out_i_rem = i, out_idx_tracker = out_shape_len - 1;
            for (int k = ndim - 1; k >= 0; --k) {
                int current_dim_idx;
                if (k == dim) {
                    current_dim_idx = j;
                } else {
                    int dim_k = out_shape[out_idx_tracker--];
                    current_dim_idx = out_i_rem % dim_k;
                    out_i_rem /= dim_k;
                }
                in_linear_idx += current_dim_idx * stride;
                stride *= self.shape[k];
            }
            float current_val = self.data->flex[in_linear_idx];
            if (current_val > best_val) { best_val = current_val; best_idx = j; }
        }
        values.data->flex[i] = best_val;
        indices.data->flex[i] = (float)best_idx;
    }

    if (requires_grad) {
        values.node->grad_fn = GradFn_reduce_dim;
        values.node->inputs[0] = self;
        values.node->inputs[1] = indices;
        values.node->n_inputs = 2;
        values.node->name = "MaxDim";
    }
    
    TensorMaxMinResult result = {values, indices};
    return result;
}

Tensor Tensor_min_all(Tensor self) {
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor res = Tensor_new((TensorShape){1, 0, 0, 0}, requires_grad);

    if (self.data->numel == 0) cten_assert(false, "min on empty tensor");
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

TensorMaxMinResult Tensor_min_dim(Tensor self, int dim) {
    int ndim = TensorShape_dim(self.shape);
    dim = TensorShape_asdim(self.shape, dim);

    TensorShape out_shape = {0};
    int out_shape_len = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) out_shape[out_shape_len++] = self.shape[i];
    }
    
    bool requires_grad = !cten_is_eval() && (self.node != NULL);
    Tensor values = Tensor_new(out_shape, requires_grad);
    Tensor indices = Tensor_new(out_shape, false);

    int dim_size = self.shape[dim];
    for (int i = 0; i < values.data->numel; ++i) {
        float best_val = INFINITY;
        int best_idx = -1;

        for (int j = 0; j < dim_size; ++j) {
            int in_linear_idx = 0, stride = 1, out_i_rem = i, out_idx_tracker = out_shape_len - 1;
            for (int k = ndim - 1; k >= 0; --k) {
                int current_dim_idx;
                if (k == dim) {
                    current_dim_idx = j;
                } else {
                    int dim_k = out_shape[out_idx_tracker--];
                    current_dim_idx = out_i_rem % dim_k;
                    out_i_rem /= dim_k;
                }
                in_linear_idx += current_dim_idx * stride;
                stride *= self.shape[k];
            }
            float current_val = self.data->flex[in_linear_idx];
            if (current_val < best_val) { best_val = current_val; best_idx = j; }
        }
        values.data->flex[i] = best_val;
        indices.data->flex[i] = (float)best_idx;
    }
    
    if (requires_grad) {
        values.node->grad_fn = GradFn_reduce_dim;
        values.node->inputs[0] = self;
        values.node->inputs[1] = indices;
        values.node->n_inputs = 2;
        values.node->name = "MinDim";
    }
    
    TensorMaxMinResult result = {values, indices};
    return result;
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
    int a_ndims = TensorShape_dim(orig_a.shape);
    int b_ndims = TensorShape_dim(orig_b.shape);
    int max_ndims = (a_ndims > b_ndims) ? a_ndims : b_ndims;

    if (max_ndims > 4) return false;
    memset(result_shape, 0, sizeof(TensorShape));

    for (int i = 0; i < max_ndims; i++) {
        int a_idx = a_ndims - 1 - i;
        int b_idx = b_ndims - 1 - i;
        int result_idx = max_ndims - 1 - i;
        int a_dim = (a_idx >= 0) ? orig_a.shape[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? orig_b.shape[b_idx] : 1;
        if (a_dim == b_dim || a_dim == 1 || b_dim == 1) {
            result_shape[result_idx] = (a_dim > b_dim) ? a_dim : b_dim;
        } else {
            return false;
        }
    }

    // 2. Check if tensor 'a' needs to be expanded
    if (memcmp(orig_a.shape, result_shape, sizeof(TensorShape)) != 0) {
        Tensor new_a = Tensor_new(result_shape, orig_a.node != NULL);
        for (int i = 0; i < new_a.data->numel; i++) {
            int rem = i;
            int idx[4] = {0};
            for (int d = max_ndims - 1; d >= 0; d--) {
                idx[d] = rem % result_shape[d];
                rem /= result_shape[d];
            }

            int source_idx = 0;
            int stride = 1;
            //iterating backwards over the original tensor's dimensions
            for (int d = a_ndims - 1; d >= 0; d--) {
                int original_dim_size = orig_a.shape[d];
                int result_dim_coord = idx[max_ndims - a_ndims + d];
                //if original dimension was 1, it's broadcast; its index is 0.
                int dim_idx = (original_dim_size == 1) ? 0 : result_dim_coord;
                source_idx += dim_idx * stride;
                stride *= original_dim_size;
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
            int idx[4] = {0};
            for (int d = max_ndims - 1; d >= 0; d--) {
                idx[d] = rem % result_shape[d];
                rem /= result_shape[d];
            }

            int source_idx = 0;
            int stride = 1;
            for (int d = b_ndims - 1; d >= 0; d--) {
                int original_dim_size = orig_b.shape[d];
                int result_dim_coord = idx[max_ndims - b_ndims + d];
                int dim_idx = (original_dim_size == 1) ? 0 : result_dim_coord;
                source_idx += dim_idx * stride;
                stride *= original_dim_size;
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

Tensor Tensor_unsqueeze(Tensor self, int dim) {
    int old_ndim = TensorShape_dim(self.shape);
    cten_assert(dim >= 0 && dim <= old_ndim, "Unsqueeze dim out of bounds");

    TensorShape new_shape = {0};
    int old_idx = 0;
    // insert a '1' at the 'dim' position in the new shape.
    for (int i = 0; i < old_ndim + 1 && i < 4; i++) {
        if (i == dim) {
            new_shape[i] = 1;
        } else {
            if(old_idx < 4) {
               new_shape[i] = self.shape[old_idx++];
            }
        }
    }

    Tensor res = self;
    memcpy(res.shape, new_shape, sizeof(TensorShape));
    
    return res;
}

Tensor Tensor_reduce_with_indices(Tensor self, int dim, const char* operation, int* indices_out) {
    int ndim = TensorShape_dim(self.shape);
    dim = TensorShape_asdim(self.shape, dim);

    TensorShape out_shape = {0};
    int out_idx = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_shape[out_idx++] = self.shape[i];
        }
    }

    Tensor res = Tensor_new(out_shape, self.node != NULL);
    int dim_size = self.shape[dim];
    int total_out_elements = res.data->numel;

    for (int i = 0; i < total_out_elements; ++i) {
        float best_val;
        int best_idx = -1;

        if (strcmp(operation, "max") == 0) {
            best_val = -INFINITY;
        } else { // "min"
            best_val = INFINITY;
        }

        // This loop iterates 'dim_size' times for each output element
        for (int j = 0; j < dim_size; ++j) {
            // Calculate the linear index in the source tensor
            int in_linear_idx = 0;
            int stride = 1;
            int out_i_rem = i;

            // This logic maps an output index back to an input index
            for (int k = ndim - 1; k >= 0; --k) {
                int current_dim_idx;
                if (k == dim) {
                    current_dim_idx = j;
                } else {
                    int out_shape_k = out_shape[--out_idx];
                    current_dim_idx = out_i_rem % out_shape_k;
                    out_i_rem /= out_shape_k;
                }
                in_linear_idx += current_dim_idx * stride;
                stride *= self.shape[k];
            }
            // Reset out_idx for next iteration of outer loop
            out_idx = TensorShape_dim(out_shape);

            float current_val = self.data->flex[in_linear_idx];
            if (strcmp(operation, "max") == 0) {
                if (current_val > best_val) {
                    best_val = current_val;
                    best_idx = j;
                }
            } else { // "min"
                if (current_val < best_val) {
                    best_val = current_val;
                    best_idx = j;
                }
            }
        }
        res.data->flex[i] = best_val;
        if (indices_out != NULL) {
            indices_out[i] = best_idx;
        }
    }
    return res;
}