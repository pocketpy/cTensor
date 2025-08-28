/**
 * @file cten.h
 * @brief CTensor - A lightweight tensor library for C with automatic differentiation
 *
 * CTensor provides tensor operations, neural network functions, and automatic
 * differentiation capabilities for machine learning applications in C.
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <limits.h>

#define _CTEN_PICK_REDUCE(_1, _2, NAME, ...) NAME
#define Tensor_max(...) _CTEN_PICK_REDUCE(__VA_ARGS__, Tensor_max_dim, Tensor_max_all)(__VA_ARGS__)
#define Tensor_min(...) _CTEN_PICK_REDUCE(__VA_ARGS__, Tensor_min_dim, Tensor_min_all)(__VA_ARGS__)

#define _CTEN_PICK(_1, _2, NAME, ...) NAME
#define Tensor_mean(...) _CTEN_PICK(__VA_ARGS__, Tensor_mean_dim, Tensor_mean_all)(__VA_ARGS__)
#define Tensor_sum(...) _CTEN_PICK(__VA_ARGS__, Tensor_sum_dim, Tensor_sum_all)(__VA_ARGS__)

/** @brief Tensor shape type supporting up to 4 dimensions */
typedef int TensorShape[4];
typedef struct GradNode GradNode;

/**
 * @brief Float buffer structure with flexible array member
 * @details Stores tensor data with element count and flexible array
 */
typedef struct FloatBuffer {
    int numel;    /**< Number of elements in the buffer */
    float flex[]; /**< Flexible array member containing the actual data */
} FloatBuffer;

/**
 * @brief Main tensor structure
 * @details Contains tensor shape, data buffer, and gradient computation node
 */
typedef struct Tensor {
    TensorShape shape; /**< Tensor dimensions [dim0, dim1, dim2, dim3] */
    FloatBuffer* data; /**< Pointer to data buffer */
    GradNode* node;    /**< Gradient computation node (NULL if no gradients) */
} Tensor;

/**
 * @brief Gradient computation node for automatic differentiation
 * @details Stores gradient function, inputs, and metadata for backpropagation
 */
typedef struct GradNode {
    struct Tensor grad;                                  /**< Accumulated gradient */
    struct Tensor (*grad_fn)(struct Tensor self, int i); /**< Gradient function */
    struct Tensor inputs[4];                             /**< Input tensors */
    int n_inputs;                                        /**< Number of inputs */
    const char* name;                                    /**< Operation name for debugging */
    int params[4];                                       /**< Additional parameters */
} GradNode;

/**
 * @brief Result structure for max/min operations along a dimension
 * @details Contains both the values and their corresponding indices
 */
typedef struct {
    Tensor values;  /**< Maximum/minimum values */
    Tensor indices; /**< Indices of maximum/minimum values */
} TensorMaxMinResult;

/**
 * @brief Initialize the CTensor library
 * @details Sets up internal memory management system. Must be called before using CTensor.
 */
void cten_initilize();

/**
 * @brief Finalize and cleanup the CTensor library
 * @details Frees all allocated memory and cleans up internal structures.
 * Should be called when done using CTensor.
 */
void cten_finalize();

/* TensorShape */

/**
 * @brief Calculate total number of elements in a tensor shape
 * @param shape The tensor shape
 * @return Number of elements (product of all dimensions)
 */
int TensorShape_numel(TensorShape shape);

/**
 * @brief Get the number of dimensions in a tensor shape
 * @param shape The tensor shape
 * @return Number of non-zero dimensions
 */
int TensorShape_dim(TensorShape shape);

/**
 * @brief Normalize dimension index (handle negative indices)
 * @param shape The tensor shape
 * @param dim The dimension index (can be negative)
 * @return Normalized positive dimension index
 */
int TensorShape_asdim(TensorShape shape, int dim);

/**
 * @brief Convert tensor shape to string representation
 * @param shape The tensor shape
 * @param buf Buffer to write the string to
 * @param size Size of the buffer
 * @return Number of characters written
 */
int TensorShape_tostring(TensorShape shape, char* buf, int size);

/* Tensor Basic */

/**
 * @brief Create a new tensor with uninitialized data
 * @param shape The desired tensor shape
 * @param requires_grad Whether to track gradients for this tensor
 * @return New tensor with allocated memory
 */
Tensor Tensor_new(TensorShape shape, bool requires_grad);

/**
 * @brief Create a tensor filled with zeros
 * @param shape The desired tensor shape
 * @param requires_grad Whether to track gradients for this tensor
 * @return New tensor filled with zeros
 */
Tensor Tensor_zeros(TensorShape shape, bool requires_grad);

/**
 * @brief Create a tensor filled with ones
 * @param shape The desired tensor shape
 * @param requires_grad Whether to track gradients for this tensor
 * @return New tensor filled with ones
 */
Tensor Tensor_ones(TensorShape shape, bool requires_grad);

/**
 * @brief Transpose a 2D tensor
 * @param self The input tensor (must be 2D)
 * @return Transposed tensor
 */
Tensor Tensor_transpose(Tensor self);

/**
 * @brief Get element value at specified indices
 * @param self The tensor
 * @param i First dimension index
 * @param j Second dimension index
 * @param k Third dimension index
 * @param l Fourth dimension index
 * @return Element value at the specified position
 */
float Tensor_get(Tensor self, int i, int j, int k, int l);

/**
 * @brief Set element value at specified indices
 * @param self The tensor
 * @param i First dimension index
 * @param j Second dimension index
 * @param k Third dimension index
 * @param l Fourth dimension index
 * @param value Value to set
 */
void Tensor_set(Tensor self, int i, int j, int k, int l, float value);

/**
 * @brief Perform backward pass (backpropagation)
 * @param self The tensor to backpropagate from
 * @param grad The gradient to backpropagate
 */
void Tensor_backward(Tensor self, Tensor grad);

/**
 * @brief Apply a function to all tensors in the computation graph
 * @param self The root tensor
 * @param f Function to apply to each tensor (can be NULL)
 * @param ctx Context pointer passed to the function
 * @return Number of tensors visited in the computation graph
 */
int Tensor_backward_apply(Tensor self, void (*f)(Tensor, void*), void* ctx);

/**
 * @brief Print tensor contents to stdout
 * @param self The tensor to print
 */
void Tensor_print(Tensor self);

/* Tensor Operations */

/**
 * @brief Element-wise addition of two tensors
 * @param self First tensor
 * @param other Second tensor
 * @return Result of self + other (with broadcasting if needed)
 */
Tensor Tensor_add(Tensor self, Tensor other);

/**
 * @brief Element-wise subtraction of two tensors
 * @param self First tensor
 * @param other Second tensor
 * @return Result of self - other (with broadcasting if needed)
 */
Tensor Tensor_sub(Tensor self, Tensor other);

/**
 * @brief Element-wise multiplication of two tensors
 * @param self First tensor
 * @param other Second tensor
 * @return Result of self * other (with broadcasting if needed)
 */
Tensor Tensor_mul(Tensor self, Tensor other);

/**
 * @brief Element-wise division of two tensors
 * @param self First tensor
 * @param other Second tensor
 * @return Result of self / other (with broadcasting if needed)
 */
Tensor Tensor_div(Tensor self, Tensor other);

/**
 * @brief Element-wise power of two tensors
 * @param self Base tensor
 * @param other Exponent tensor
 * @return Result of self ^ other (with broadcasting if needed)
 */
Tensor Tensor_pow(Tensor self, Tensor other);

/**
 * @brief Add a scalar to all elements of a tensor
 * @param self The tensor
 * @param other Scalar value to add
 * @return Result of self + other
 */
Tensor Tensor_addf(Tensor self, float other);

/**
 * @brief Subtract a scalar from all elements of a tensor
 * @param self The tensor
 * @param other Scalar value to subtract
 * @return Result of self - other
 */
Tensor Tensor_subf(Tensor self, float other);

/**
 * @brief Multiply all elements of a tensor by a scalar
 * @param self The tensor
 * @param other Scalar value to multiply by
 * @return Result of self * other
 */
Tensor Tensor_mulf(Tensor self, float other);

/**
 * @brief Divide all elements of a tensor by a scalar
 * @param self The tensor
 * @param other Scalar value to divide by
 * @return Result of self / other
 */
Tensor Tensor_divf(Tensor self, float other);

/**
 * @brief Raise all elements of a tensor to a scalar power
 * @param self The tensor
 * @param other Scalar exponent
 * @return Result of self ^ other
 */
Tensor Tensor_powf(Tensor self, float other);

/**
 * @brief Matrix multiplication of two tensors
 * @param self First tensor (left operand)
 * @param other Second tensor (right operand)
 * @return Result of matrix multiplication self @ other
 */
Tensor Tensor_matmul(Tensor self, Tensor other);

/**
 * @brief Element-wise negation
 * @param self The tensor
 * @return Result of -self
 */
Tensor Tensor_neg(Tensor self);

/**
 * @brief Element-wise absolute value
 * @param self The tensor
 * @return Result of |self|
 */
Tensor Tensor_abs(Tensor self);

/**
 * @brief Element-wise square
 * @param self The tensor
 * @return Result of self^2
 */
Tensor Tensor_square(Tensor self);

/**
 * @brief Element-wise reciprocal
 * @param self The tensor
 * @return Result of 1/self
 */
Tensor Tensor_reciprocal(Tensor self);

/* Helper functions that the macros dispatch to */

/**
 * @brief Calculate mean of all elements in a tensor
 * @param self The tensor
 * @return Scalar tensor containing the mean
 */
Tensor Tensor_mean_all(Tensor self);

/**
 * @brief Calculate mean along a specific dimension
 * @param self The tensor
 * @param dim The dimension to reduce
 * @return Tensor with reduced dimension
 */
Tensor Tensor_mean_dim(Tensor self, int dim);

/**
 * @brief Calculate sum of all elements in a tensor
 * @param self The tensor
 * @return Scalar tensor containing the sum
 */
Tensor Tensor_sum_all(Tensor self);

/**
 * @brief Calculate sum along a specific dimension
 * @param self The tensor
 * @param dim The dimension to reduce
 * @return Tensor with reduced dimension
 */
Tensor Tensor_sum_dim(Tensor self, int dim);

/**
 * @brief Find maximum value among all elements
 * @param self The tensor
 * @return Scalar tensor containing the maximum value
 */
Tensor Tensor_max_all(Tensor self);

/**
 * @brief Find maximum values and indices along a specific dimension
 * @param self The tensor
 * @param dim The dimension to reduce
 * @return TensorMaxMinResult containing values and indices
 */
TensorMaxMinResult Tensor_max_dim(Tensor self, int dim);

/**
 * @brief Find minimum value among all elements
 * @param self The tensor
 * @return Scalar tensor containing the minimum value
 */
Tensor Tensor_min_all(Tensor self);

/**
 * @brief Find minimum values and indices along a specific dimension
 * @param self The tensor
 * @param dim The dimension to reduce
 * @return TensorMaxMinResult containing values and indices
 */
TensorMaxMinResult Tensor_min_dim(Tensor self, int dim);

/**
 * @brief Find indices of maximum values
 * @param self The tensor
 * @param out Output array to store indices
 */
void Tensor_argmax(Tensor self, int* out);

/* Neural Networks */

/**
 * @brief Element-wise natural logarithm
 * @param self The tensor
 * @return Result of ln(self)
 */
Tensor nn_log(Tensor self);

/**
 * @brief Element-wise exponential function
 * @param self The tensor
 * @return Result of exp(self)
 */
Tensor nn_exp(Tensor self);

/**
 * @brief Element-wise sine function
 * @param self The tensor
 * @return Result of sin(self)
 */
Tensor nn_sin(Tensor self);

/**
 * @brief Element-wise cosine function
 * @param self The tensor
 * @return Result of cos(self)
 */
Tensor nn_cos(Tensor self);

/**
 * @brief Element-wise tangent function
 * @param self The tensor
 * @return Result of tan(self)
 */
Tensor nn_tan(Tensor self);

/**
 * @brief Linear transformation (fully connected layer)
 * @param input Input tensor [batch_size, in_features]
 * @param weight Weight tensor [in_features, out_features]
 * @param bias Bias tensor [out_features]
 * @return Result of input @ weight + bias
 */
Tensor nn_linear(Tensor input, Tensor weight, Tensor bias);

/**
 * @brief ReLU activation function
 * @param input The input tensor
 * @return Result of max(0, input)
 */
Tensor nn_relu(Tensor input);

/**
 * @brief Sigmoid activation function
 * @param input The input tensor
 * @return Result of 1 / (1 + exp(-input))
 */
Tensor nn_sigmoid(Tensor input);

/**
 * @brief Hyperbolic tangent activation function
 * @param input The input tensor
 * @return Result of tanh(input)
 */
Tensor nn_tanh(Tensor input);

/**
 * @brief ELU (Exponential Linear Unit) activation function
 * @param self The input tensor
 * @param alpha ELU parameter (typically 1.0)
 * @return Result of ELU(self, alpha)
 */
Tensor nn_elu(Tensor self, float alpha);

/**
 * @brief SELU (Scaled Exponential Linear Unit) activation function
 * @param self The input tensor
 * @return Result of SELU(self)
 */
Tensor nn_selu(Tensor self);

/**
 * @brief Softmax function along a specified dimension
 * @param input The input tensor
 * @param dim The dimension to apply softmax
 * @return Softmax probabilities
 */
Tensor nn_softmax(Tensor input, int dim);

/**
 * @brief Initialize tensor with Glorot/Xavier initialization
 * @param shape The tensor shape (typically [fan_in, fan_out])
 * @param requires_grad Whether to track gradients
 * @return Tensor initialized with Glorot distribution
 */
Tensor Glorot_init(TensorShape shape, bool requires_grad);

/**
 * @brief Cross-entropy loss function
 * @param y_true True labels (one-hot encoded) [batch_size, num_classes]
 * @param y_pred Predicted probabilities [batch_size, num_classes]
 * @return Scalar tensor containing the cross-entropy loss
 */
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);

/**
 * @brief Softmax followed by cross-entropy loss (numerically stable)
 * @param y_true True labels (one-hot encoded) [batch_size, num_classes]
 * @param logits Raw logits [batch_size, num_classes]
 * @return Scalar tensor containing the softmax cross-entropy loss
 */
Tensor nn_softmax_crossentropy(Tensor y_true, Tensor logits);

/**
 * @brief Mean Squared Error loss function
 * @param y_true True values
 * @param y_pred Predicted values
 * @return Scalar tensor containing MSE loss
 */
Tensor nn_mse_loss(Tensor y_true, Tensor y_pred);

/**
 * @brief Mean Absolute Error loss function
 * @param y_true True values
 * @param y_pred Predicted values
 * @return Scalar tensor containing MAE loss
 */
Tensor nn_mae_loss(Tensor y_true, Tensor y_pred);

/**
 * @brief Huber loss function (smooth L1 loss)
 * @param y_true True values
 * @param y_pred Predicted values
 * @param delta Threshold parameter
 * @return Scalar tensor containing Huber loss
 */
Tensor nn_huber_loss(Tensor y_true, Tensor y_pred, float delta);

/* Memory Management */

/** @brief Pool identifier type for memory management */
typedef int64_t PoolId;

/**
 * @brief Begin memory allocation in a specific pool
 * @param id Pool identifier
 * @details All subsequent tensor allocations will be assigned to this pool
 */
void cten_begin_malloc(PoolId id);

/**
 * @brief End the current memory allocation pool
 * @details Returns to the previous pool in the stack
 */
void cten_end_malloc();

/**
 * @brief Free all tensors allocated in a specific pool
 * @param id Pool identifier to free
 */
void cten_free(PoolId id);

/* Optimizer */

/** @brief SGD optimizer structure */
typedef struct optim_sgd optim_sgd;
/** @brief AdaGrad optimizer structure */
typedef struct optim_adagrad optim_adagrad;
/** @brief RMSprop optimizer structure */
typedef struct optim_rmsprop optim_rmsprop;
/** @brief Adam optimizer structure */
typedef struct optim_adam optim_adam;

// SGD

/**
 * @brief Create new SGD optimizer
 * @param n_params Number of parameter tensors
 * @param params Array of parameter tensors to optimize
 * @param weight_decay L2 regularization coefficient
 * @return Pointer to SGD optimizer instance
 */
optim_sgd* optim_sgd_new(int n_params, Tensor* params, float weight_decay);

/**
 * @brief Configure SGD optimizer parameters
 * @param self SGD optimizer instance
 * @param lr Learning rate
 * @param momentum Momentum coefficient
 */
void optim_sgd_config(optim_sgd* self, float lr, float momentum);

/**
 * @brief Zero out all gradients
 * @param self SGD optimizer instance
 */
void optim_sgd_zerograd(optim_sgd* self);

/**
 * @brief Perform one optimization step
 * @param self SGD optimizer instance
 */
void optim_sgd_step(optim_sgd* self);

// AdaGrad

/**
 * @brief Create new AdaGrad optimizer
 * @param n_params Number of parameter tensors
 * @param params Array of parameter tensors to optimize
 * @param lr Learning rate
 * @param ε Small constant for numerical stability
 * @param weight_decay L2 regularization coefficient
 * @return Pointer to AdaGrad optimizer instance
 */
optim_adagrad*
    optim_adagrad_new(int n_params, Tensor* params, float lr, float ε, float weight_decay);

/**
 * @brief Zero out all gradients
 * @param self AdaGrad optimizer instance
 */
void optim_adagrad_zerograd(optim_adagrad* self);

/**
 * @brief Perform one optimization step
 * @param self AdaGrad optimizer instance
 */
void optim_adagrad_step(optim_adagrad* self);

// RMSProp

/**
 * @brief Create new RMSProp optimizer
 * @param n_params Number of parameter tensors
 * @param params Array of parameter tensors to optimize
 * @param lr Learning rate
 * @param β Decay rate for moving average
 * @param ε Small constant for numerical stability
 * @param weight_decay L2 regularization coefficient
 * @return Pointer to RMSProp optimizer instance
 */
optim_rmsprop*
    optim_rmsprop_new(int n_params, Tensor* params, float lr, float β, float ε, float weight_decay);

/**
 * @brief Zero out all gradients
 * @param self RMSProp optimizer instance
 */
void optim_rmsprop_zerograd(optim_rmsprop* self);

/**
 * @brief Perform one optimization step
 * @param self RMSProp optimizer instance
 */
void optim_rmsprop_step(optim_rmsprop* self);

// Adam

/**
 * @brief Create new Adam optimizer
 * @param n_params Number of parameter tensors
 * @param params Array of parameter tensors to optimize
 * @param lr Learning rate
 * @param β1 Exponential decay rate for first moment estimates
 * @param β2 Exponential decay rate for second moment estimates
 * @param ε Small constant for numerical stability
 * @param weight_decay L2 regularization coefficient
 * @return Pointer to Adam optimizer instance
 */
optim_adam* optim_adam_new(int n_params,
                           Tensor* params,
                           float lr,
                           float β1,
                           float β2,
                           float ε,
                           float weight_decay);

/**
 * @brief Zero out all gradients
 * @param self Adam optimizer instance
 */
void optim_adam_zerograd(optim_adam* self);

/**
 * @brief Perform one optimization step
 * @param self Adam optimizer instance
 */
void optim_adam_step(optim_adam* self);

/* Gradient Clipping */

/**
 * @brief Clip gradients by global norm
 * @param params Array of parameter tensors
 * @param n_params Number of parameters
 * @param max_norm Maximum allowed gradient norm
 */
void cten_clip_grad_norm(Tensor* params, int n_params, float max_norm);

/**
 * @brief Clip gradients by absolute value
 * @param params Array of parameter tensors
 * @param n_params Number of parameters
 * @param max_value Maximum absolute value for gradients
 */
void cten_clip_grad_value(Tensor* params, int n_params, float max_value);

/**
 * @brief Clip gradients to a value range
 * @param params Array of parameter tensors
 * @param n_params Number of parameters
 * @param min_value Minimum gradient value
 * @param max_value Maximum gradient value
 */
void cten_clip_grad_value_range(Tensor* params, int n_params, float min_value, float max_value);

/**
 * @brief Clip positive gradients to maximum value
 * @param params Array of parameter tensors
 * @param n_params Number of parameters
 * @param max_value Maximum value for positive gradients
 */
void cten_clip_grad_positive(Tensor* params, int n_params, float max_value);

/**
 * @brief Clip negative gradients to minimum value
 * @param params Array of parameter tensors
 * @param n_params Number of parameters
 * @param min_value Minimum value for negative gradients
 */
void cten_clip_grad_negative(Tensor* params, int n_params, float min_value);

/* Misc */

/**
 * @brief Enter evaluation mode (disables gradient computation)
 * @details Gradients will not be computed for operations in eval mode
 */
void cten_begin_eval();

/**
 * @brief Check if currently in evaluation mode
 * @return true if in evaluation mode, false otherwise
 */
bool cten_is_eval();

/**
 * @brief Exit evaluation mode (re-enables gradient computation)
 */
void cten_end_eval();

/**
 * @brief Check if variadic argument is present (utility function)
 * @param args Variadic argument list
 * @return Always returns false (placeholder implementation)
 */
bool va_arg_is_present(va_list args);

/* Utils */

/**
 * @brief Normalize dataset using training statistics
 * @param X Input dataset [n_samples][n_features]
 * @param X_norm Output normalized dataset [n_samples][n_features]
 * @param n_samples Total number of samples
 * @param n_train_samples Number of training samples (used for computing stats)
 * @param n_features Number of features
 * @details Computes mean and std from training samples, applies to all samples
 */
void Tensor_normalize_dataset(const float (*X)[4],
                              float (*X_norm)[4],
                              int n_samples,
                              int n_train_samples,
                              int n_features);

/**
 * @brief Detach tensor from computation graph
 * @param self The tensor to detach
 * @return New tensor with same data but no gradient tracking
 * @details Creates a copy that doesn't participate in backpropagation
 */
Tensor Tensor_detach(Tensor self);

/**
 * @brief Shuffle dataset randomly
 * @param X Input features [n_samples][n_features]
 * @param y Input labels [n_samples]
 * @param X_shuffled Output shuffled features [n_samples][n_features]
 * @param y_shuffled Output shuffled labels [n_samples]
 * @param n_samples Number of samples
 * @param n_features Number of features
 */
void Tensor_shuffle_dataset(const float (*X)[4],
                            const int* y,
                            float (*X_shuffled)[4],
                            int* y_shuffled,
                            int n_samples,
                            int n_features);

/**
 * @brief Assert condition with formatted message
 * @param cond Condition to check
 * @param fmt Format string for error message
 * @param ... Format arguments
 */
void cten_assert(bool cond, const char* fmt, ...);

/**
 * @brief Assert that two tensor shapes are equal
 * @param title Description for assertion
 * @param a First tensor shape
 * @param b Second tensor shape
 */
void cten_assert_shape(const char* title, TensorShape a, TensorShape b);

/**
 * @brief Assert that two dimensions are equal
 * @param title Description for assertion
 * @param a First dimension
 * @param b Second dimension
 */
void cten_assert_dim(const char* title, int a, int b);

/**
 * @brief Perform element-wise broadcasting on two tensors
 * @param a Pointer to first tensor (modified in-place if broadcasting needed)
 * @param b Pointer to second tensor (modified in-place if broadcasting needed)
 * @return true if broadcasting successful, false if incompatible shapes
 * @details Modifies tensors in-place to have compatible shapes for element-wise ops
 */
bool cten_elemwise_broadcast(Tensor* a, Tensor* b);

/**
 * @brief Load the Iris dataset
 * @param X Pointer to receive features array [150][4]
 * @param y Pointer to receive labels array [150]
 * @return Number of samples loaded (150 for Iris)
 */
int load_iris_dataset(const float (**X)[4], const int** y);

/**
 * @brief Reduce tensor along a dimension using specified operation
 * @param self The input tensor
 * @param dim The dimension to reduce
 * @param operation Operation name ("sum", "mean", etc.)
 * @return Tensor with reduced dimension
 * @details Internal function used by reduction operations
 */
Tensor Tensor_reduce_dim(Tensor self, int dim, const char* operation);

/**
 * @brief Reduce gradient tensor to match original shape after broadcasting
 * @param grad The gradient tensor
 * @param original_shape The original tensor shape before broadcasting
 * @param broadcasted_shape The shape after broadcasting
 * @return Reduced gradient tensor matching original shape
 */
Tensor reduce_gradient_for_broadcasting(Tensor grad,
                                        TensorShape original_shape,
                                        TensorShape broadcasted_shape);

/**
 * @brief Add a singleton dimension at specified position
 * @param self The input tensor
 * @param dim Position to insert new dimension
 * @return Tensor with added dimension of size 1
 */
Tensor Tensor_unsqueeze(Tensor self, int dim);