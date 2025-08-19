#include "cten.h"
#include "cten_internal.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct optim_adagrad {
    int n_params;
    Tensor* params;
    float lr;
    float ε;
    Tensor* sum_sq_grad;
    float weight_decay;
} optim_adagrad;

optim_adagrad*
    optim_adagrad_new(int n_params, Tensor* params, float lr, float ε, float weight_decay) {
    cten_assert(n_params >= 0, "AdaGrad: n_params cannot be negative, but got %d.", n_params);
    if(n_params > 0) {
        cten_assert(params != NULL, "AdaGrad: params array cannot be NULL when n_params > 0.");
    }
    cten_assert(lr >= 0.0f, "AdaGrad: learning rate must be non-negative, but got %f.", lr);
    cten_assert(ε >= 0.0f, "AdaGrad: epsilon must be non-negative, but got %f.", ε);
    cten_assert(weight_decay >= 0.0f,
                "AdaGrad: weight decay must be non-negative, but got %f.",
                weight_decay);

    optim_adagrad* self = _cten_malloc(sizeof(optim_adagrad));
    self->n_params = n_params;
    self->params = params;
    self->lr = lr;
    self->ε = ε;
    self->sum_sq_grad = _cten_malloc(sizeof(Tensor) * n_params);
    self->weight_decay = weight_decay;
    for(int i = 0; i < n_params; i++) {
        self->sum_sq_grad[i] = Tensor_zeros(params[i].shape, false);
    }
    return self;
}

void optim_adagrad_zerograd(optim_adagrad* self) { _cten_zero_grad(self->params, self->n_params); }

void optim_adagrad_step(optim_adagrad* self) {
    for(int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        if(t.node == NULL || t.node->grad.data == NULL) continue;

        Tensor grad = t.node->grad;
        Tensor* sum_sq = &self->sum_sq_grad[i];

        for(int j = 0; j < t.data->numel; j++) {
            float g = grad.data->flex[j];
            if(self->weight_decay > 0.0f) { g += self->weight_decay * t.data->flex[j]; }
            sum_sq->data->flex[j] += g * g;
            t.data->flex[j] -= self->lr * g / (sqrtf(sum_sq->data->flex[j]) + self->ε);
        }
    }
}
