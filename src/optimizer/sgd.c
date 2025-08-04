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
    Tensor* velocity;
} optim_sgd;

optim_sgd* optim_sgd_new(int n_params, Tensor* params) {
    cten_assert(n_params >= 0, "n_params cannot be negative, but got %d.", n_params);
    if (n_params > 0) {
        cten_assert(params != NULL, "params array cannot be NULL when n_params is greater than 0.");
    }

    optim_sgd* self = _cten_malloc(sizeof(optim_sgd));
    self->n_params = n_params;
    self->params = params;
    self->lr = 0.001f;
    self->momentum = 0.0f;
    self->velocity = NULL;
    return self;
}

void optim_sgd_config(optim_sgd* self, float lr, float momentum) {
    cten_assert(momentum >= 0.0f, "Momentum must be non-negative, but got %f", momentum);
    self->lr = lr;
    self->momentum = momentum;

    if (self->velocity == NULL && self->momentum > 0.0f) {
        self->velocity = _cten_malloc(sizeof(Tensor) * self->n_params);
        for (int i = 0; i < self->n_params; i++) {
            self->velocity[i] = Tensor_zeros(self->params[i].shape, false);
        }
    }
}

void optim_sgd_zerograd(optim_sgd* self) {
    _cten_zero_grad(self->params, self->n_params);
}

void optim_sgd_step(optim_sgd* self) {
    for(int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        if(t.node == NULL || t.node->grad.data == NULL) {
            continue;
        }

        float* param_data = t.data->flex;
        float* grad_data = t.node->grad.data->flex;

        if (self->momentum > 0.0f) {
            // v = momentum * v + grad
            // p = p - lr * v
            cten_assert(self->velocity != NULL, "Velocity buffer is NULL. Did you configure momentum?");
            float* velocity_data = self->velocity[i].data->flex;
            for (int j = 0; j < t.data->numel; j++) {
                velocity_data[j] = self->momentum * velocity_data[j] + grad_data[j];
                param_data[j] -= self->lr * velocity_data[j];
            }
        } else {
            // p = p - lr * grad
            for(int j = 0; j < t.data->numel; j++) {
                param_data[j] -= self->lr * grad_data[j];
            }
        }
    }
}