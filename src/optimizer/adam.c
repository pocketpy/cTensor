#include "cten.h"
#include "cten_internal.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct optim_adam {
    int n_params;
    Tensor* params;
    float lr;
    float β1;
    float β2;
    float ε;
    Tensor* m;
    Tensor* v;
    int t;
} optim_adam;

optim_adam* optim_adam_new(int n_params, Tensor* params, float lr, float β1, float β2, float ε) {
    cten_assert(n_params >= 0, "Adam: n_params cannot be negative, but got %d.", n_params);
    if (n_params > 0) {
        cten_assert(params != NULL, "Adam: params array cannot be NULL when n_params > 0.");
    }
    cten_assert(lr >= 0.0f, "Adam: learning rate must be non-negative, but got %f.", lr);
    cten_assert(β1 >= 0.0f && β1 < 1.0f, "Adam: beta1 must be in [0, 1), but got %f.", β1);
    cten_assert(β2 >= 0.0f && β2 < 1.0f, "Adam: beta2 must be in [0, 1), but got %f.", β2);
    cten_assert(ε >= 0.0f, "Adam: epsilon must be non-negative, but got %f.", ε);

    optim_adam* self = _cten_malloc(sizeof(optim_adam));
    self->n_params = n_params;
    self->params = params;
    self->lr = lr;
    self->β1 = β1;
    self->β2 = β2;
    self->ε = ε;
    self->t = 0;

    self->m = _cten_malloc(sizeof(Tensor) * n_params);
    self->v = _cten_malloc(sizeof(Tensor) * n_params);
    for (int i = 0; i < n_params; i++) {
        self->m[i] = Tensor_zeros(params[i].shape, false);
        self->v[i] = Tensor_zeros(params[i].shape, false);
    }
    return self;
}

void optim_adam_zerograd(optim_adam* self) {
    _cten_zero_grad(self->params, self->n_params);
}

void optim_adam_step(optim_adam* self) {
    self->t++;
    for (int i = 0; i < self->n_params; i++) {
        Tensor p = self->params[i];
        if (p.node == NULL || p.node->grad.data == NULL) continue;

        Tensor grad = p.node->grad;
        Tensor* m = &self->m[i];
        Tensor* v = &self->v[i];

        for (int j = 0; j < p.data->numel; j++) {
            float g = grad.data->flex[j];
            m->data->flex[j] = self->β1 * m->data->flex[j] + (1 - self->β1) * g;
            v->data->flex[j] = self->β2 * v->data->flex[j] + (1 - self->β2) * g * g;
            float m_hat = m->data->flex[j] / (1 - powf(self->β1, self->t));
            float v_hat = v->data->flex[j] / (1 - powf(self->β2, self->t));
            p.data->flex[j] -= self->lr * m_hat / (sqrtf(v_hat) + self->ε);
        }
    }
}