#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef int TensorShape[4];
typedef struct GradNode GradNode;

typedef struct FloatBuffer {
    int numel;
    float flex[];
} FloatBuffer;

typedef struct Tensor {
    TensorShape shape;
    FloatBuffer* data;
    GradNode* node;
} Tensor;

typedef struct GradNode {
    struct Tensor grad;
    struct Tensor (*grad_fn)(struct Tensor self, int i);
    struct Tensor inputs[4];
    int n_inputs;
} GradNode;

void cten_initilize();
void cten_finalize();

/* TensorShape */
int TensorShape_numel(TensorShape shape);
int TensorShape_dim(TensorShape shape);
int TensorShape_asdim(TensorShape shape, int dim);
int TensorShape_tostring(TensorShape shape, char* buf, int size);

/* Tensor Basic */
Tensor Tensor_new(TensorShape shape, bool requires_grad);
Tensor Tensor_zeros(TensorShape shape, bool requires_grad);
Tensor Tensor_ones(TensorShape shape, bool requires_grad);

float Tensor_get(Tensor self, int i, int j, int k, int l);
void Tensor_set(Tensor self, int i, int j, int k, int l, float value);

Tensor Tensor_detach(Tensor self);
void Tensor_backward(Tensor self, Tensor grad);
int Tensor_backward_apply(Tensor self, void (*f)(Tensor, void*), void* ctx);

void Tensor_print(Tensor self);

/* Tensor Operations */
Tensor Tensor_add(Tensor self, Tensor other);
Tensor Tensor_sub(Tensor self, Tensor other);
Tensor Tensor_mul(Tensor self, Tensor other);
Tensor Tensor_div(Tensor self, Tensor other);
Tensor Tensor_pow(Tensor self, Tensor other);

Tensor Tensor_addf(Tensor self, float other);
Tensor Tensor_subf(Tensor self, float other);
Tensor Tensor_mulf(Tensor self, float other);
Tensor Tensor_divf(Tensor self, float other);
Tensor Tensor_powf(Tensor self, float other);

Tensor Tensor_matmul(Tensor self, Tensor other);

Tensor Tensor_neg(Tensor self);
Tensor Tensor_abs(Tensor self);

Tensor Tensor_sum(Tensor self);
Tensor Tensor_mean(Tensor self);
Tensor Tensor_max(Tensor self);
Tensor Tensor_min(Tensor self);

void Tensor_argmax(Tensor self, int* out);

/* Neural Networks */
Tensor nn_log(Tensor self);
Tensor nn_exp(Tensor self);

Tensor nn_sin(Tensor self);
Tensor nn_cos(Tensor self);
Tensor nn_tan(Tensor self);

Tensor nn_linear(Tensor input, Tensor weight, Tensor bias);
Tensor nn_relu(Tensor input);
Tensor nn_sigmoid(Tensor input);
Tensor nn_tanh(Tensor input);
Tensor nn_softmax(Tensor input);
Tensor nn_random_init(TensorShape shape, bool requires_grad);
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);

/* Memory Management */
typedef int64_t PoolId;

void cten_begin_malloc(PoolId id);
void cten_end_malloc();
void cten_free(PoolId id);

/* Optimizer */
typedef struct optim_sgd optim_sgd;

optim_sgd* optim_sgd_new(int n_params, Tensor* params);
void optim_sgd_config(optim_sgd* self, float lr, float momentum);
void optim_sgd_zerograd(optim_sgd* self);
void optim_sgd_step(optim_sgd* self);
void optim_sgd_delete(optim_sgd* self);

/* Misc */
void cten_begin_eval();
bool cten_is_eval();
void cten_end_eval();

void cten_assert(bool cond, const char* fmt, ...);
void cten_assert_shape(const char* title, TensorShape a, TensorShape b);
void cten_assert_dim(const char* title, int a, int b);

bool cten_elemwise_broadcast(Tensor* a, Tensor* b);
int load_iris_dataset(const float (**X)[4], const int** y);