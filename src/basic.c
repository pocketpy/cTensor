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
    self.data = malloc(sizeof(FloatBuffer) + sizeof(float) * numel);
    self.data->refcount = 1;
    self.data->numel = numel;
    if(requires_grad) {
        self.node = malloc(sizeof(GradNode));
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

void Tensor_delete(Tensor self) {
    if(--self.data->refcount == 0) free(self.data);
    if(self.node != NULL) {
        if(self.node->grad.data != NULL) Tensor_delete(self.node->grad);
        free(self.node);
    }
}

Tensor Tensor_detach(Tensor self) {
    Tensor detached = self;
    detached.data->refcount++;
    detached.node = NULL;
    return detached;
}

void Tensor_backward(Tensor self, Tensor grad) {
    if(self.node == NULL) return;
    if(grad.data == NULL) {
        assert(self.data->numel == 1);
        grad = Tensor_ones((TensorShape){0}, false);
    }
    assert(grad.node == NULL);
    if(self.node->grad.data == NULL) {
        self.node->grad = grad;
    } else {
        self.node->grad = Tensor_add(self.node->grad, grad);
    }
    for(int i = 0; i < self.node->n_inputs; i++) {
        grad = Tensor_mul(self.node->grad, self.node->grad_fn(self, i));
        //-TODO judgement below is for softmax-return's matrix, but imperfect
        if (TensorShape_dim(grad.shape) - TensorShape_dim(self.node->grad.shape) == 1   \
            && TensorShape_dim(self.node->grad.shape) != 0) {
            for (int i = 0; i < self.node->grad.data->numel; i++) {
                float sum = 0;
                for (int j = 0; j < grad.shape[1]; j++) {
                    sum += grad.data->flex[i*grad.shape[0]+j];
                }
                grad.data->flex[i] = sum;
            }
            for (int i = 0; i < 4; i++) grad.shape[i] = self.node->grad.shape[i];
        }
        Tensor_backward(self.node->inputs[i], grad);
    }
    // Tensor_delete(grad);
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
