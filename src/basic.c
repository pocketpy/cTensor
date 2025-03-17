#include "cten.h"
#include "cten_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if(self.node == NULL) return;
    if(grad.data == NULL) {
        assert(self.data->numel == 1);
        grad = Tensor_ones((TensorShape){1}, false);
    }
    
    assert(grad.node == NULL);
    if(self.node->grad.data == NULL) {
        self.node->grad = grad;
    } else {
        self.node->grad = Tensor_add(self.node->grad, grad);
    }

    for(int i = 0; i < self.node->n_inputs; i++) {
        if (self.node->inputs[i].data == NULL) continue;
        Tensor combined_grad;  
        Tensor input_grad = self.node->grad_fn(self, i); 
        if(strcmp(self.node->name, "Matmul") == 0){
            if (i == 0){
                combined_grad = Tensor_matmul(grad, input_grad);
            }
            else{
                combined_grad = Tensor_matmul(input_grad, grad);
            }
        }
        else{
            combined_grad = Tensor_mul(grad, input_grad);
        }       
        Tensor_backward(self.node->inputs[i], combined_grad);
    }
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