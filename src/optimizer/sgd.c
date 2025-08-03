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

typedef struct optim_adagrad {
    int n_params;
    Tensor* params;
    float lr;
    float ε;
    Tensor* sum_sq_grad;
} optim_adagrad;

optim_adagrad* optim_adagrad_new(int n_params, Tensor* params, float lr, float ε) {
    cten_assert(n_params >= 0, "AdaGrad: n_params cannot be negative, but got %d.", n_params);
    if (n_params > 0) {
        cten_assert(params != NULL, "AdaGrad: params array cannot be NULL when n_params > 0.");
    }
    cten_assert(lr >= 0.0f, "AdaGrad: learning rate must be non-negative, but got %f.", lr);
    cten_assert(ε >= 0.0f, "AdaGrad: epsilon must be non-negative, but got %f.", ε);

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