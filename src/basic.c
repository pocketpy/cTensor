#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int TensorShape_numel(TensorShape shape) {
    int numel = 1;
    for(int i = 0; i < sizeof(TensorShape) / sizeof(shape[0]); i++) {
        if(shape[i] == 0) break;
        numel *= shape[i];
    }
    return numel;
}

int TensorShape_dim(TensorShape shape) {
    for(int i = 0; i < sizeof(TensorShape) / sizeof(shape[0]); i++) {
        if(shape[i] == 0) return i;
    }
    return sizeof(TensorShape) / sizeof(shape[0]);
}

int TensorShape_asdim(TensorShape shape, int dim) {
    int shape_dim = TensorShape_dim(shape);
    if(dim < 0) dim += shape_dim;
    cten_assert(dim >= 0 && dim < shape_dim, "dim %d out of range", dim);
    return dim;
}

int TensorShape_tostring(TensorShape shape, char* buf, int size) {
    return snprintf(buf, size, "(%d, %d, %d, %d)", shape[0], shape[1], shape[2], shape[3]);
}

Tensor Tensor_new(TensorShape shape, bool requires_grad) {
    Tensor self;
    memcpy(self.shape, shape, sizeof(TensorShape));
    int numel = TensorShape_numel(shape);
    self.data = _cten_malloc(sizeof(FloatBuffer) + sizeof(float) * numel);
    self.data->numel = numel;
    
    //Initialize tensor with random values
    float* data_ptr = self.data->flex;
    for (int i = 0; i < numel; i++) {
        data_ptr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    if(requires_grad) {
        self.node = _cten_malloc(sizeof(GradNode));
        memset(self.node, 0, sizeof(GradNode));
    } else {
        self.node = NULL;
    }
    return self;
}

Tensor Tensor_zeros(TensorShape shape, bool requires_grad) {
    Tensor self = Tensor_new(shape, requires_grad);
    memset(self.data->flex, 0, sizeof(float) * self.data->numel);
    return self;
}

Tensor Tensor_ones(TensorShape shape, bool requires_grad) {
    Tensor self = Tensor_new(shape, requires_grad);
    for(int i = 0; i < self.data->numel; i++) {
        self.data->flex[i] = 1.0f;
    }
    return self;
}
Tensor Tensor_transpose(Tensor self) {
    int dim = TensorShape_dim(self.shape);
    if(dim < 2){
        return self; 
    }
    TensorShape new_shape;
    new_shape[0] = self.shape[1];
    new_shape[1] = self.shape[0];
    for(int i = 2; i < 4; i++) {
        new_shape[i] = self.shape[i];
    }
    Tensor result = Tensor_new(new_shape, false);
    int rows = self.shape[0];
    int cols = self.shape[1];
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            result.data->flex[j * rows + i] = self.data->flex[i * cols + j];
        }
    }
    return result;
}

float Tensor_get(Tensor self, int i, int j, int k, int l) {
    assert((self.shape[0] == 0 && i == 0) || (i >= 0 && i < self.shape[0]));
    assert((self.shape[1] == 0 && j == 0) || (j >= 0 && j < self.shape[1]));
    assert((self.shape[2] == 0 && k == 0) || (k >= 0 && k < self.shape[2]));
    assert((self.shape[3] == 0 && l == 0) || (l >= 0 && l < self.shape[3]));
    return self.data->flex[i * self.shape[1] * self.shape[2] * self.shape[3] +
                           j * self.shape[2] * self.shape[3] + k * self.shape[3] + l];
}

void Tensor_set(Tensor self, int i, int j, int k, int l, float value) {
    assert((self.shape[0] == 0 && i == 0) || (i >= 0 && i < self.shape[0]));
    assert((self.shape[1] == 0 && j == 0) || (j >= 0 && j < self.shape[1]));
    assert((self.shape[2] == 0 && k == 0) || (k >= 0 && k < self.shape[2]));
    assert((self.shape[3] == 0 && l == 0) || (l >= 0 && l < self.shape[3]));
    self.data->flex[i * self.shape[1] * self.shape[2] * self.shape[3] +
                    j * self.shape[2] * self.shape[3] + k * self.shape[3] + l] = value;
}

Tensor Tensor_detach(Tensor self) {
    Tensor detached = self;
    detached.node = NULL;
    return detached;
}

void Tensor_backward(Tensor self, Tensor grad) {
    printf("\n=== Starting Tensor_backward ===\n");
    printf("self: "); Tensor_print(self); printf("\n");
    printf("grad: "); Tensor_print(grad); printf("\n");
    
    if(self.node == NULL) {
        printf("  - No computation graph (node is NULL), returning early\n");
        return;
    }
    
    if(grad.data == NULL) {
        printf("  - Gradient is NULL, initializing to ones (expected for scalar output)\n");
        assert(self.data->numel == 1);
        grad = Tensor_ones((TensorShape){1, 0, 0, 0}, false);
        printf("  - New grad: "); Tensor_print(grad); printf("\n");
    }
    
    assert(grad.node == NULL);
    
    // Print current gradient status
    printf("  - Current node gradient: ");
    if (self.node->grad.data == NULL) {
        printf("NULL\n");
    } else {
        Tensor_print(self.node->grad);
        printf("\n");
    }
    
    // Accumulate gradient
    if(self.node->grad.data == NULL) {
        printf("  - Setting initial gradient\n");
        Tensor_print(grad);
        self.node->grad = grad;
    } else {
        printf("  - Adding to existing gradient\n");
        printf("    Before add - self.node->grad: "); Tensor_print(self.node->grad); printf("\n");
        printf("    Adding grad: "); Tensor_print(grad); printf("\n");
        self.node->grad = Tensor_add(self.node->grad, grad);
        printf("    After add - self.node->grad: "); Tensor_print(self.node->grad); printf("\n");
    }

    printf("\n  - Processing %d inputs\n", self.node->n_inputs);
    printf("Where each input tensor are:\n");
    for(int i = 0; i < self.node->n_inputs; i++) {
        Tensor_print(self.node->inputs[i]);
    }
    
    for(int i = 0; i < self.node->n_inputs; i++) {
        printf("\n  --- Input %d ---\n", i);
        
        if (self.node->inputs[i].data == NULL) {
            printf("    - Input tensor is NULL, skipping\n");
            continue;
        }
        
        printf("    - Input tensor: "); Tensor_print(self.node->inputs[i]); printf("\n");
        
        // Get the original input tensor (before any broadcasting)
        Tensor input_tensor = self.node->inputs[i];
        
        // Get gradient function for this input
        printf("    - Getting gradient function for input %d\n", i);
        Tensor input_grad = self.node->grad_fn(self, i);
        printf("    - Gradient function returned: "); 
        if (input_grad.data == NULL) {
            printf("NULL\n");
        } else {
            Tensor_print(input_grad); 
            printf(" (shape=[%d,%d,%d,%d])\n", 
                   input_grad.shape[0], input_grad.shape[1], 
                   input_grad.shape[2], input_grad.shape[3]);
        }
        
        // *** FIRST: HANDLE BROADCASTING FOR INPUT_GRAD ***
        // Check if input_grad shape needs reduction to match original input shape
        bool input_grad_needs_reduction = false;
        for (int dim = 0; dim < 4; dim++) {
            if (input_grad.shape[dim] != input_tensor.shape[dim]) {
                input_grad_needs_reduction = true;
                break;
            }
        }
        
        if (input_grad_needs_reduction) {
            printf("    !!! INPUT_GRAD BROADCASTING REDUCTION NEEDED !!!\n");
            printf("    - Input tensor shape: [%d,%d,%d,%d]\n", 
                   input_tensor.shape[0], input_tensor.shape[1], 
                   input_tensor.shape[2], input_tensor.shape[3]);
            printf("    - Input grad shape: [%d,%d,%d,%d]\n", 
                   input_grad.shape[0], input_grad.shape[1], 
                   input_grad.shape[2], input_grad.shape[3]);
            
            // For element-wise operations, figure out the broadcasted shape
            TensorShape broadcasted_shape;
            for(int dim = 0; dim < 4; dim++) {
                broadcasted_shape[dim] = (self.shape[dim] == 0) ? input_tensor.shape[dim] : self.shape[dim];
            }
            
            printf("    - Inferred broadcasted shape: [%d,%d,%d,%d]\n", 
                   broadcasted_shape[0], broadcasted_shape[1], 
                   broadcasted_shape[2], broadcasted_shape[3]);
            
            // Reduce input_grad to match original input tensor shape
            input_grad = reduce_gradient_for_broadcasting(input_grad, input_tensor.shape, broadcasted_shape);
            
            printf("    - Reduced input_grad: "); 
            Tensor_print(input_grad); 
            printf(" (shape=[%d,%d,%d,%d])\n", 
                   input_grad.shape[0], input_grad.shape[1], 
                   input_grad.shape[2], input_grad.shape[3]);
        } else {
            printf("    - No input_grad reduction needed - shapes match\n");
        }
        
        // *** SECOND: COMPUTE COMBINED GRADIENT ***
        // Handle different operation types
        Tensor combined_grad;
        if(strcmp(self.node->name, "Matmul") == 0) {
            printf("    - Matmul operation detected\n");
            if (i == 0) {
                printf("    - First input (left matrix)\n");
                printf("      Computing grad @ input_grad\n");
                combined_grad = Tensor_matmul(grad, input_grad);
            } else {
                printf("    - Second input (right matrix)\n");
                printf("      Computing input_grad @ grad\n");
                combined_grad = Tensor_matmul(input_grad, grad);
            }
        } else {
            printf("    - Element-wise operation detected\n");
            printf("      Computing grad * input_grad\n");
            Tensor_print(grad);
            printf("\n*\n");
            Tensor_print(input_grad);
            combined_grad = Tensor_mul(grad, input_grad);
        }
        
        printf("    - Combined gradient: "); 
        if (combined_grad.data == NULL) {
            printf("NULL\n");
        } else {
            Tensor_print(combined_grad); 
            printf(" (shape=[%d,%d,%d,%d])\n", 
                   combined_grad.shape[0], combined_grad.shape[1], 
                   combined_grad.shape[2], combined_grad.shape[3]);
        }
        
        // *** THIRD: FINAL SAFETY CHECK FOR COMBINED_GRAD ***
        // Check if combined_grad shape matches the original input shape
        bool final_needs_reduction = false;
        for (int dim = 0; dim < 4; dim++) {
            if (combined_grad.shape[dim] != input_tensor.shape[dim]) {
                final_needs_reduction = true;
                break;
            }
        }
        
        if (final_needs_reduction) {
            printf("    !!! FINAL COMBINED_GRAD REDUCTION NEEDED !!!\n");
            printf("    - Input tensor shape: [%d,%d,%d,%d]\n", 
                   input_tensor.shape[0], input_tensor.shape[1], 
                   input_tensor.shape[2], input_tensor.shape[3]);
            printf("    - Combined grad shape: [%d,%d,%d,%d]\n", 
                   combined_grad.shape[0], combined_grad.shape[1], 
                   combined_grad.shape[2], combined_grad.shape[3]);
            
            // For element-wise operations, figure out the broadcasted shape
            TensorShape broadcasted_shape;
            for(int dim = 0; dim < 4; dim++) {
                broadcasted_shape[dim] = (self.shape[dim] == 0) ? input_tensor.shape[dim] : self.shape[dim];
            }
            
            printf("    - Inferred broadcasted shape: [%d,%d,%d,%d]\n", 
                   broadcasted_shape[0], broadcasted_shape[1], 
                   broadcasted_shape[2], broadcasted_shape[3]);
            
            // Apply final broadcasting reduction
            combined_grad = reduce_gradient_for_broadcasting(combined_grad, input_tensor.shape, broadcasted_shape);
            
            printf("    - Final reduced gradient: "); 
            Tensor_print(combined_grad); 
            printf(" (shape=[%d,%d,%d,%d])\n", 
                   combined_grad.shape[0], combined_grad.shape[1], 
                   combined_grad.shape[2], combined_grad.shape[3]);
        } else {
            printf("    - No final reduction needed - shapes match perfectly\n");
        }
        
        printf("    - Calling backward on input %d with gradient: ", i);
        Tensor_print(combined_grad); 
        printf("\n");
        
        Tensor_backward(self.node->inputs[i], combined_grad);
        printf("    - Back from backward() for input %d\n", i);
    }
    printf("=== Finished Tensor_backward ===\n\n");
}

int Tensor_backward_apply(Tensor self, void (*f)(Tensor, void*), void* ctx) {
    if(self.node == NULL) return 0;
    if(f != NULL) f(self, ctx);
    int count = 1;
    for(int i = 0; i < self.node->n_inputs; i++) {
        count += Tensor_backward_apply(self.node->inputs[i], f, ctx);
    }
    return count;
}

void Tensor_print(Tensor self) {
    if(self.data == NULL) {
        printf("Tensor()\n");
        return;
    }
    printf("Tensor([");
    for(int i = 0; i < self.data->numel; i++) {
        printf("%.4f", self.data->flex[i]);
        if(i < self.data->numel - 1) printf(", ");
    }
    printf("], shape=(");
    for(int i = 0; i < 4; i++) {
        if(self.shape[i] == 0) {
            break;
        } else {
            if(i > 0) printf(", ");
        }
        printf("%d", self.shape[i]);
    }

    if(self.node != NULL) {
        printf("), grad_fn=<%p>, grad=", self.node->grad_fn);
        Tensor_print(self.node->grad);
    } else {
        printf(")");
    }
    printf(")\n");
}

void _cten_zero_grad(Tensor* params, int n_params) {
    for(int i = 0; i < n_params; i++) {
        Tensor t = params[i];
        if(t.node == NULL) continue;
        t.node->grad = Tensor_zeros(t.shape, false);
    }
}