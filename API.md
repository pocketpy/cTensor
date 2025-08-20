# cTensor API Reference

This document provides a detailed API reference for the cTensor library, a lightweight tensor library for C with automatic differentiation.

## Table of Contents

1.  [Core Data Structures](#core-data-structures)
2.  [Library Initialization & Finalization](#library-initialization--finalization)
3.  [TensorShape Utilities](#tensorshape-utilities)
4.  [Tensor Creation & Management](#tensor-creation--management)
5.  [Tensor Operations](#tensor-operations)
      * [Element-wise Arithmetic](#element-wise-arithmetic)
      * [Matrix & Unary Operations](#matrix--unary-operations)
      * [Reduction Operations](#reduction-operations)
6.  [Neural Network Functions](#neural-network-functions)
      * [Layers & Initializers](#layers--initializers)
      * [Activation Functions](#activation-functions)
      * [Loss Functions](#loss-functions)
      * [Mathematical Functions](#mathematical-functions)
7.  [Automatic Differentiation](#automatic-differentiation)
8.  [Optimizers](#optimizers)
      * [SGD (Stochastic Gradient Descent)](#sgd-stochastic-gradient-descent)
      * [AdaGrad](#adagrad)
      * [RMSprop](#rmsprop)
      * [Adam](#adam)
9.  [Gradient Clipping](#gradient-clipping)
10. [Memory Management](#memory-management)
11. [Utilities & Miscellaneous](#utilities--miscellaneous)

-----

## Core Data Structures

These are the fundamental data types used throughout the cTensor library.

### `TensorShape`

A type definition for tensor shapes, supporting up to 4 dimensions.

```c
typedef int TensorShape[4];
```

-----

### `FloatBuffer`

A structure storing the raw tensor data.

```c
typedef struct FloatBuffer {
    int numel;      /**< Number of elements in the buffer */
    float flex[];   /**< Flexible array member containing the actual data */
} FloatBuffer;
```

-----

### `Tensor`

The main tensor structure, containing its shape, data, and a node for gradient computation.

```c
typedef struct Tensor {
    TensorShape shape; /**< Tensor dimensions [dim0, dim1, dim2, dim3] */
    FloatBuffer* data; /**< Pointer to data buffer */
    GradNode* node;    /**< Gradient computation node (NULL if no gradients) */
} Tensor;
```

-----

### `GradNode`

A node in the computation graph used for automatic differentiation.

```c
typedef struct GradNode {
    struct Tensor grad;
    struct Tensor (*grad_fn)(struct Tensor self, int i);
    struct Tensor inputs[4];
    int n_inputs;
    const char* name;
    int params[4];
} GradNode;
```

**Fields:**

  * `grad`: The accumulated gradient for the tensor associated with this node.
  * `grad_fn`: A function pointer to the gradient function used in backpropagation.
  * `inputs`: An array of input tensors that produced the current tensor.
  * `n_inputs`: The number of input tensors.
  * `name`: The name of the operation for debugging.
  * `params`: Additional integer parameters required by the operation.

-----

### `TensorMaxMinResult`

A structure to hold the results of `max` or `min` operations along a dimension.

```c
typedef struct {
    Tensor values;  /**< Maximum/minimum values */
    Tensor indices; /**< Indices of maximum/minimum values */
} TensorMaxMinResult;
```

-----

## Library Initialization & Finalization

### `cten_initilize`

Initializes the CTensor library and its internal memory management system. **Must be called before any other CTensor function.**

```c
void cten_initilize();
```

-----

### `cten_finalize`

Frees all allocated memory and cleans up internal library structures. Should be called when finished using CTensor.

```c
void cten_finalize();
```

-----

## TensorShape Utilities

Functions for working with `TensorShape` types.

### `TensorShape_numel`

Calculates the total number of elements in a tensor shape (product of dimensions).

```c
int TensorShape_numel(TensorShape shape);
```

-----

### `TensorShape_dim`

Gets the number of dimensions in a tensor shape (number of non-zero dimensions).

```c
int TensorShape_dim(TensorShape shape);
```

-----

### `TensorShape_asdim`

Normalizes a dimension index to handle negative indices (e.g., -1 for the last dimension).

```c
int TensorShape_asdim(TensorShape shape, int dim);
```

-----

### `TensorShape_tostring`

Converts a tensor shape to its string representation.

```c
int TensorShape_tostring(TensorShape shape, char* buf, int size);
```

-----

## Tensor Creation & Management

Functions for creating and manipulating `Tensor` objects.

### `Tensor_new`

Creates a new tensor with **uninitialized data**.

```c
Tensor Tensor_new(TensorShape shape, bool requires_grad);
```

-----

### `Tensor_zeros`

Creates a new tensor filled with **zeros**.

```c
Tensor Tensor_zeros(TensorShape shape, bool requires_grad);
```

-----

### `Tensor_ones`

Creates a new tensor filled with **ones**.

```c
Tensor Tensor_ones(TensorShape shape, bool requires_grad);
```

-----

### `Tensor_detach`

**Detaches a tensor from the computation graph.** The new tensor shares the same data but does not require gradients.

```c
Tensor Tensor_detach(Tensor self);
```

-----

### `Tensor_unsqueeze`

Adds a singleton dimension (a dimension of size 1) at a specified position.

```c
Tensor Tensor_unsqueeze(Tensor self, int dim);
```

-----

### `Tensor_get`

Gets the element value at the specified indices.

```c
float Tensor_get(Tensor self, int i, int j, int k, int l);
```

-----

### `Tensor_set`

Sets the element value at the specified indices.

```c
void Tensor_set(Tensor self, int i, int j, int k, int l, float value);
```

-----

### `Tensor_print`

Prints the contents of a tensor to `stdout`.

```c
void Tensor_print(Tensor self);
```

-----

## Tensor Operations

### Element-wise Arithmetic

These functions perform element-wise arithmetic. They support **broadcasting** to handle operands with different but compatible shapes.

| Function | Description |
|---|---|
| `Tensor_add(a, b)` | Adds two tensors. |
| `Tensor_sub(a, b)` | Subtracts tensor `b` from `a`. |
| `Tensor_mul(a, b)` | Multiplies two tensors. |
| `Tensor_div(a, b)` | Divides tensor `a` by `b`. |
| `Tensor_pow(a, b)` | Raises tensor `a` to the power of `b`. |
| `Tensor_addf(a, s)` | Adds a scalar `s` to a tensor. |
| `Tensor_subf(a, s)` | Subtracts a scalar `s` from a tensor. |
| `Tensor_mulf(a, s)` | Multiplies a tensor by a scalar `s`. |
| `Tensor_divf(a, s)` | Divides a tensor by a scalar `s`. |
| `Tensor_powf(a, s)` | Raises a tensor to the power of a scalar `s`. |

-----

### Matrix & Unary Operations

### `Tensor_matmul`

Performs **matrix multiplication** of two tensors.

```c
Tensor Tensor_matmul(Tensor self, Tensor other);
```

-----

### `Tensor_transpose`

Transposes a 2D tensor.

```c
Tensor Tensor_transpose(Tensor self);
```

-----

### `Tensor_neg`

Performs element-wise negation (`-self`).

```c
Tensor Tensor_neg(Tensor self);
```

-----

### `Tensor_abs`

Computes the element-wise absolute value (`|self|`).

```c
Tensor Tensor_abs(Tensor self);
```

-----

### `Tensor_square`

Computes the element-wise square (`self^2`).

```c
Tensor Tensor_square(Tensor self);
```

-----

### `Tensor_reciprocal`

Computes the element-wise reciprocal (`1/self`).

```c
Tensor Tensor_reciprocal(Tensor self);
```

-----

### Reduction Operations

These operations reduce a tensor to a single value or along a specified dimension. They are exposed via macros for a simpler API.

### Sum

**Usage:**

```c
// Sum of all elements (returns a scalar tensor)
Tensor sum_all = Tensor_sum(my_tensor);

// Sum along dimension 1 (returns a tensor with the dimension removed)
Tensor sum_dim = Tensor_sum(my_tensor, 1);
```

#### **Underlying Functions:** `Tensor_sum_all(Tensor self)`, `Tensor_sum_dim(Tensor self, int dim)`

### Mean

**Usage:**

```c
// Mean of all elements
Tensor mean_all = Tensor_mean(my_tensor);

// Mean along dimension 1
Tensor mean_dim = Tensor_mean(my_tensor, 1);
```

#### **Underlying Functions:** `Tensor_mean_all(Tensor self)`, `Tensor_mean_dim(Tensor self, int dim)`

### Max

**Usage:**

```c
// Max of all elements
Tensor max_val = Tensor_max(my_tensor);

// Max along dimension 1 (returns values and indices)
TensorMaxMinResult max_res = Tensor_max(my_tensor, 1);
Tensor max_vals = max_res.values;
Tensor max_indices = max_res.indices;
```

#### **Underlying Functions:** `Tensor_max_all(Tensor self)`, `TensorMaxMinResult Tensor_max_dim(Tensor self, int dim)`

### Min

**Usage:**

```c
// Min of all elements
Tensor min_val = Tensor_min(my_tensor);

// Min along dimension 1 (returns values and indices)
TensorMaxMinResult min_res = Tensor_min(my_tensor, 1);
```

#### **Underlying Functions:** `Tensor_min_all(Tensor self)`, `TensorMaxMinResult Tensor_min_dim(Tensor self, int dim)`

### Argmax

### `Tensor_argmax`

Finds the indices of the maximum values along the last dimension.

```c
void Tensor_argmax(Tensor self, int* out);
```

-----

## Neural Network Functions

### Layers & Initializers

### `nn_linear`

Applies a linear transformation (`input @ weight + bias`).

```c
Tensor nn_linear(Tensor input, Tensor weight, Tensor bias);
```

-----

### `Glorot_init`

Initializes a tensor with weights sampled from a **Glorot (Xavier)** uniform distribution.

```c
Tensor Glorot_init(TensorShape shape, bool requires_grad);
```

-----

### Activation Functions

| Function | Description |
|---|---|
| `nn_relu(input)` | Rectified Linear Unit: `max(0, input)`. |
| `nn_sigmoid(input)` | Sigmoid: `1 / (1 + exp(-input))`. |
| `nn_tanh(input)` | Hyperbolic Tangent. |
| `nn_elu(self, alpha)` | Exponential Linear Unit. |
| `nn_selu(self)` | Scaled Exponential Linear Unit. |
| `nn_softmax(input, dim)` | Softmax function along a specified dimension. |

-----

### Loss Functions

### `nn_crossentropy`

Computes the **cross-entropy loss** between true labels and predicted probabilities.

```c
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);
```

-----

### `nn_softmax_crossentropy`

A numerically stable combination of Softmax and Cross-Entropy loss.

```c
Tensor nn_softmax_crossentropy(Tensor y_true, Tensor logits);
```

-----

### `nn_mse_loss`

Computes the **Mean Squared Error** loss.

```c
Tensor nn_mse_loss(Tensor y_true, Tensor y_pred);
```

-----

### `nn_mae_loss`

Computes the **Mean Absolute Error** loss.

```c
Tensor nn_mae_loss(Tensor y_true, Tensor y_pred);
```

-----

### `nn_huber_loss`

Computes the **Huber loss** (a smooth L1 loss).

```c
Tensor nn_huber_loss(Tensor y_true, Tensor y_pred, float delta);
```

-----

### Mathematical Functions

| Function | Description |
|---|---|
| `nn_log(self)` | Element-wise natural logarithm. |
| `nn_exp(self)` | Element-wise exponential function (`e^x`). |
| `nn_sin(self)` | Element-wise sine. |
| `nn_cos(self)` | Element-wise cosine. |
| `nn_tan(self)` | Element-wise tangent. |

-----

## Automatic Differentiation

### `Tensor_backward`

Performs the **backward pass (backpropagation)** from this tensor, computing gradients for all tensors in its computation graph that have `requires_grad=true`.

```c
void Tensor_backward(Tensor self, Tensor grad);
```

**Parameters:**

  * `self`: The tensor to start the backpropagation from (often the final loss).
  * `grad`: The initial gradient to propagate. For a scalar loss, this is typically a tensor containing the value `1.0`.

-----

### `Tensor_backward_apply`

Applies a function to all tensors visited during a backward pass.

```c
int Tensor_backward_apply(Tensor self, void (*f)(Tensor, void*), void* ctx);
```

-----

## Optimizers

### SGD (Stochastic Gradient Descent)

```c
// Create a new SGD optimizer
optim_sgd* optim_sgd_new(int n_params, Tensor* params, float weight_decay);

// Configure learning rate and momentum
void optim_sgd_config(optim_sgd* self, float lr, float momentum);

// Zero out the gradients of all managed parameters
void optim_sgd_zerograd(optim_sgd* self);

// Perform one optimization step
void optim_sgd_step(optim_sgd* self);
```

-----

### AdaGrad

```c
optim_adagrad* optim_adagrad_new(int n_params, Tensor* params, float lr, float ε, float weight_decay);
void optim_adagrad_zerograd(optim_adagrad* self);
void optim_adagrad_step(optim_adagrad* self);
```

-----

### RMSprop

```c
optim_rmsprop* optim_rmsprop_new(int n_params, Tensor* params, float lr, float β, float ε, float weight_decay);
void optim_rmsprop_zerograd(optim_rmsprop* self);
void optim_rmsprop_step(optim_rmsprop* self);
```

-----

### Adam

```c
optim_adam* optim_adam_new(int n_params, Tensor* params, float lr, float β1, float β2, float ε, float weight_decay);
void optim_adam_zerograd(optim_adam* self);
void optim_adam_step(optim_adam* self);
```

-----

## Gradient Clipping

Functions to prevent exploding gradients during training.

### `cten_clip_grad_norm`

Clips the gradients of a set of parameters by their **global L2 norm**.

```c
void cten_clip_grad_norm(Tensor* params, int n_params, float max_norm);
```

-----

### `cten_clip_grad_value`

Clips gradients element-wise to a **maximum absolute value**.

```c
void cten_clip_grad_value(Tensor* params, int n_params, float max_value);
```

-----

### `cten_clip_grad_value_range`

Clips gradients element-wise to be within `[min_value, max_value]`.

```c
void cten_clip_grad_value_range(Tensor* params, int n_params, float min_value, float max_value);
```

-----

## Memory Management

cTensor uses a pool-based memory allocator to manage tensor memory, which is especially useful for controlling memory usage during different phases like training epochs.

### `cten_begin_malloc`

Begins a new memory allocation pool. All subsequent tensor allocations will be associated with this pool ID.

```c
void cten_begin_malloc(PoolId id);
```

-----

### `cten_end_malloc`

Ends the current memory allocation pool, returning to the previous one in the stack.

```c
void cten_end_malloc();
```

-----

### `cten_free`

Frees **all** tensors that were allocated in the specified pool.

```c
void cten_free(PoolId id);
```

-----

## Utilities & Miscellaneous

### Evaluation Mode

Disables gradient computation globally, useful for inference or validation.

### `cten_begin_eval`

Enters evaluation mode.

```c
void cten_begin_eval();
```

-----

### `cten_is_eval`

Checks if the library is currently in evaluation mode.

```c
bool cten_is_eval();
```

-----

### `cten_end_eval`

Exits evaluation mode, re-enabling gradient computation.

```c
void cten_end_eval();
```

-----

### Dataset Helpers

### `load_iris_dataset`

Loads the built-in Iris dataset.

```c
int load_iris_dataset(const float (**X)[4], const int** y);
```

-----

### `Tensor_normalize_dataset`

Normalizes a dataset using the mean and standard deviation from its training split.

```c
void Tensor_normalize_dataset(const float (*X)[4], float (*X_norm)[4], int n_samples, int n_train_samples, int n_features);
```

-----

### `Tensor_shuffle_dataset`

Randomly shuffles a dataset (features and labels together).

```c
void Tensor_shuffle_dataset(const float (*X)[4], const int* y, float (*X_shuffled)[4], int* y_shuffled, int n_samples, int n_features);
```

-----

### Assertions & Broadcasting

### `cten_assert`

Asserts that a condition is true, otherwise prints a formatted error message and exits.

```c
void cten_assert(bool cond, const char* fmt, ...);
```

-----

### `cten_assert_shape`

Asserts that two tensor shapes are equal.

```c
void cten_assert_shape(const char* title, TensorShape a, TensorShape b);
```

-----

### `cten_assert_dim`

Asserts that two dimension sizes are equal.

```c
void cten_assert_dim(const char* title, int a, int b);
```

-----

### `cten_elemwise_broadcast`

Internal function to perform broadcasting on two tensors for element-wise operations.

```c
bool cten_elemwise_broadcast(Tensor* a, Tensor* b);
```