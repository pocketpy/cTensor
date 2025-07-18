#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <limits.h>

#define _CTEN_PICK_REDUCE(_1, _2, NAME, ...) NAME
#define Tensor_max(...)  _CTEN_PICK_REDUCE(__VA_ARGS__, Tensor_max_dim,  Tensor_max_all)(__VA_ARGS__)
#define Tensor_min(...)  _CTEN_PICK_REDUCE(__VA_ARGS__, Tensor_min_dim,  Tensor_min_all)(__VA_ARGS__)

#define _CTEN_PICK(_1,_2,NAME,...) NAME
#define Tensor_mean(...) _CTEN_PICK(__VA_ARGS__, Tensor_mean_dim, Tensor_mean_all)(__VA_ARGS__)
#define Tensor_sum(...)  _CTEN_PICK(__VA_ARGS__, Tensor_sum_dim,  Tensor_sum_all )(__VA_ARGS__)

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
    const char* name;
} GradNode;

typedef struct {
    Tensor values;
    Tensor indices;
} TensorMaxMinResult;

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
Tensor Tensor_transpose(Tensor self);

float Tensor_get(Tensor self, int i, int j, int k, int l);
void Tensor_set(Tensor self, int i, int j, int k, int l, float value);
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
Tensor Tensor_square(Tensor self);
Tensor Tensor_reciprocal(Tensor self);

/* Helper functions that the macros dispatch to */
Tensor Tensor_mean_all(Tensor self);
Tensor Tensor_mean_dim(Tensor self, int dim);
Tensor Tensor_sum_all (Tensor self);
Tensor Tensor_sum_dim (Tensor self, int dim);

Tensor Tensor_max_all(Tensor self);
TensorMaxMinResult Tensor_max_dim(Tensor self, int dim);
Tensor Tensor_min_all(Tensor self);
TensorMaxMinResult Tensor_min_dim(Tensor self, int dim);

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
Tensor Glorot_init(TensorShape shape, bool requires_grad);
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);
Tensor nn_softmax_crossentropy(Tensor y_true, Tensor logits);

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
bool va_arg_is_present(va_list args);

/* Utils */
void Tensor_normalize_dataset(const float (*X)[4], float (*X_norm)[4], int n_samples, int n_train_samples, int n_features);Tensor Tensor_detach(Tensor self);
void Tensor_shuffle_dataset(const float (*X)[4], const int *y,float (*X_shuffled)[4], int *y_shuffled, int n_samples, int n_features);
void cten_assert(bool cond, const char* fmt, ...);
void cten_assert_shape(const char* title, TensorShape a, TensorShape b);
void cten_assert_dim(const char* title, int a, int b);
bool cten_elemwise_broadcast(Tensor* a, Tensor* b);
int load_iris_dataset(const float (**X)[4], const int** y);
Tensor Tensor_reduce_dim(Tensor self, int dim, const char* operation);
Tensor reduce_gradient_for_broadcasting(Tensor grad, TensorShape original_shape, TensorShape broadcasted_shape);
Tensor Tensor_unsqueeze(Tensor self, int dim);