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
    cten_assert(n_params >= 0, "n_params cannot be negative, but got %d.", n_params);
    if (n_params > 0) {
        cten_assert(params != NULL, "params array cannot be NULL when n_params is greater than 0.");
    }

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