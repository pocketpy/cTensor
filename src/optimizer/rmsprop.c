#include "cten.h"
#include "cten_internal.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct optim_rmsprop {
    int n_params;
    Tensor* params;
    float lr;
    float β;
    float ε;
    Tensor* squared_avg;
} optim_rmsprop;

optim_rmsprop* optim_rmsprop_new(int n_params, Tensor* params, float lr, float β, float ε) {
    cten_assert(n_params >= 0, "RMSProp: n_params cannot be negative, but got %d.", n_params);
    if (n_params > 0) {
        cten_assert(params != NULL, "RMSProp: params array cannot be NULL when n_params > 0.");
    }
    cten_assert(lr >= 0.0f, "RMSProp: learning rate must be non-negative, but got %f.", lr);
    cten_assert(β >= 0.0f && β < 1.0f, "RMSProp: beta (decay rate) must be in [0, 1), but got %f.", β);
    cten_assert(ε >= 0.0f, "RMSProp: epsilon must be non-negative, but got %f.", ε);

    optim_rmsprop* self = _cten_malloc(sizeof(optim_rmsprop));
    self->n_params = n_params;
    self->params = params;
    self->lr = lr;
    self->β = β;
    self->ε = ε;

    self->squared_avg = _cten_malloc(sizeof(Tensor) * n_params);
    for (int i = 0; i < n_params; i++) {
        self->squared_avg[i] = Tensor_zeros(params[i].shape, false);
    }
    return self;
}

void optim_rmsprop_zerograd(optim_rmsprop* self) {
    _cten_zero_grad(self->params, self->n_params);
}

void optim_rmsprop_step(optim_rmsprop* self) {
    for (int i = 0; i < self->n_params; i++) {
        Tensor t = self->params[i];
        if (t.node == NULL || t.node->grad.data == NULL) continue;

        Tensor grad = t.node->grad;
        Tensor* sq_avg = &self->squared_avg[i];

        for (int j = 0; j < t.data->numel; j++) {
            float g = grad.data->flex[j];
            sq_avg->data->flex[j] = self->β * sq_avg->data->flex[j] + (1 - self->β) * g * g;
            t.data->flex[j] -= self->lr * g / (sqrtf(sq_avg->data->flex[j]) + self->ε);
        }
    }
}