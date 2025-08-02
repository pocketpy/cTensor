#include "cten.h"
#include "cten_internal.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct optim_sgd {
    int n_params;
    Tensor* params;
    float lr;
    float momentum;
    // Tensor* velocity;
} optim_sgd;

optim_sgd* optim_sgd_new(int n_params, Tensor* params) {
    optim_sgd* self = _cten_malloc(sizeof(optim_sgd));
    self->n_params = n_params;
    self->params = params;
    self->lr = 0.001f;
    self->momentum = 0.0f;
    return self;
}

void optim_sgd_config(optim_sgd* self, float lr, float momentum) {
    self->lr = lr;
    self->momentum = momentum;
}

void optim_sgd_zerograd(optim_sgd* self) { _cten_zero_grad(self->params, self->n_params); }

void optim_sgd_step(optim_sgd* self) {
    assert(self->momentum == 0);
    for(int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        if(t.node == NULL) continue;
        assert(t.node->grad.data != NULL);
        // step
        for(int j = 0; j < t.data->numel; j++) {
            t.data->flex[j] -= self->lr * t.node->grad.data->flex[j];
        }
    }
}

typedef struct optim_adagrad {
    int n_params;
    Tensor* params;
    float lr;
    float ε;
    Tensor* sum_sq_grad;
} optim_adagrad;

optim_adagrad* optim_adagrad_new(int n_params, Tensor* params, float lr, float ε) {
    optim_adagrad* self = _cten_malloc(sizeof(optim_adagrad));
    self->n_params = n_params;
    self->params = params;
    self->lr = lr;
    self->ε = ε;
    self->sum_sq_grad = _cten_malloc(sizeof(Tensor) * n_params);
    for (int i = 0; i < n_params; i++) {
        self->sum_sq_grad[i] = Tensor_zeros(params[i].shape, false);
    }
    return self;
}

void optim_adagrad_zerograd(optim_adagrad* self) {
    _cten_zero_grad(self->params, self->n_params);
}

void optim_adagrad_step(optim_adagrad* self) {
    for (int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        if (t.node == NULL || t.node->grad.data == NULL) continue;

        Tensor grad = t.node->grad;
        Tensor* sum_sq = &self->sum_sq_grad[i];

        for (int j = 0; j < t.data->numel; j++) {
            float g = grad.data->flex[j];
            sum_sq->data->flex[j] += g * g;
            t.data->flex[j] -= self->lr * g / (sqrtf(sum_sq->data->flex[j]) + self->ε);
        }
    }
}