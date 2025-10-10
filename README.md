# cTensor

A lightweight neural network library written in C11 for embedded systems.

## Overview

cTensor is a compact tensor computation library designed for small client-side devices, such as mobile phones and microcontrollers. The library implements automatic differentiation and dynamic compute graph functionality, allowing for efficient training and deployment of neural networks on resource-constrained devices.

This library was developed as part of GSoC 2025 and has been successfully validated on ARM Cortex-M3 microcontrollers, achieving 90% classification accuracy on the Iris dataset in a bare-metal environment.

## Features

### Core Infrastructure
- **Lightweight C11 Implementation:** Minimal dependencies for wide compatibility
- **Automatic Differentiation Framework:** Complete gradient computation with backward pass
- **Dynamic Compute Graph:** Efficient computation flow with gradient tracking
- **Pool-based Memory Management:** Efficient memory allocation system for embedded devices

### Tensor Operations
- **Basic Arithmetic:** add, subtract, multiply, divide, power (both tensor-tensor and tensor-scalar)
- **Unary Operations:** negation, absolute value, square, reciprocal
- **Matrix Operations:** matrix multiplication, transpose
- **Mathematical Functions:** logarithm, exponential, sine, cosine, tangent
- **Shape Operations:** unsqueeze, detach
- **Broadcasting:** Element-wise broadcasting for operations on tensors with different shapes

### Reduction Operations
- **Sum:** All elements or along specific dimension
- **Mean:** All elements or along specific dimension
- **Max/Min:** All elements or along dimension with indices
- **Argmax:** Find indices of maximum values

### Neural Network Components
- **Layers:** Linear (fully connected) layer
- **Activation Functions:** ReLU, Sigmoid, Tanh, ELU, SELU, Softmax
- **Loss Functions:** Cross-entropy, Softmax Cross-entropy, MSE, MAE, Huber Loss
- **Weight Initialization:** Glorot/Xavier initialization

### Optimizers
- **SGD:** Stochastic Gradient Descent with momentum
- **Adam:** Adaptive moment estimation
- **RMSProp:** Root Mean Square Propagation
- **AdaGrad:** Adaptive Gradient Algorithm
- **Features:** Weight decay support for all optimizers

### Training Utilities
- **Gradient Clipping:** By norm, value, range, positive/negative values
- **Evaluation Mode:** Disable gradient computation for inference
- **Dataset Utilities:** Normalization, shuffling

## Validation

cTensor has been successfully deployed and tested on:
- **ARM Cortex-M3 (STM32F103ZE)** using Keil MDK simulation
- **Task:** Neural network classification on Iris dataset
- **Result:** 90% accuracy matching desktop performance
- **Complete validation project:** [cTensor_Cortex_SIM](https://github.com/PrimedErwin/cTensor_Cortex_SIM)

## Getting Started

### Prerequisites

- C Compiler with C11 support (GCC, Clang)
- CMake (3.10+) for build configuration
- Math library (automatically linked on non-Windows systems)

### Building with CMake

**On Windows:**
```batch
build.bat
```

**On Linux/macOS:**
```bash
mkdir -p build && cd build
cmake ..
cmake --build .
cd ..
```

### Building with Direct Compilation

**On Linux/macOS:**
```bash
./build_g.sh
```

**On Windows with GCC:**
```batch
gcc -std=c11 -Iinclude -O0 -Wfatal-errors -g -DDEBUG -lm src/nn.c src/operator.c src/basic.c src/iris_dataset.c src/context.c src/pool.c src/utils.c src/common/vector.c src/optimizer/sgd.c src2/main.c -o main
```
and run `main.exe` from root directory

## Testing the Library

cTensor uses a custom test framework. To run the tests:

```bash
# Build the test executable with CMake
mkdir -p build && cd build
cmake ..
cmake --build .

# Run the tests
./cten_exe
```

For detailed testing information, refer to [Testing Documentation](tests/README.md).

## Usage Example

Here's a complete example of training a neural network to predict sine wave values with noise:

```c
#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define memory pools
enum MemoryPoolIds {
    PoolId_Default = 0,
    PoolId_Model = 1,
    PoolId_Optimizer = 2,
};

// Define the model structure
typedef struct {
    Tensor w1, b1;
    Tensor w2, b2;
    Tensor w3, b3;
} Model;

// Forward pass for the model
Tensor Model_forward(Model* model, Tensor x) {
    x = nn_linear(x, model->w1, model->b1);
    x = nn_elu(x, 1.0f);
    x = nn_linear(x, model->w2, model->b2);
    x = nn_elu(x, 1.0f);
    x = nn_linear(x, model->w3, model->b3);
    return x;
}

int main() {
    cten_initilize();

    // Generate sine wave data
    int n_samples = 2048;
    float* x_data = malloc(n_samples * sizeof(float));
    float* y_data = malloc(n_samples * sizeof(float));
    // ... (data generation logic) ...

    // Create model and allocate in its own memory pool
    Model model;
    cten_begin_malloc(PoolId_Model);
    model.w1 = Glorot_init((TensorShape){1, 64}, true);
    model.b1 = Tensor_zeros((TensorShape){1, 64}, true);
    model.w2 = Glorot_init((TensorShape){64, 32}, true);
    model.b2 = Tensor_zeros((TensorShape){1, 32}, true);
    model.w3 = Glorot_init((TensorShape){32, 1}, true);
    model.b3 = Tensor_zeros((TensorShape){1, 1}, true);
    cten_end_malloc();

    // Create optimizer
    float learning_rate = 0.01f;
    cten_begin_malloc(PoolId_Optimizer);
    optim_adam* optimizer = optim_adam_new(6, (Tensor*)&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f);
    cten_end_malloc();

    // Training loop
    int batch_size = 64;
    for (int epoch = 0; epoch < 200; epoch++) {
        // ... (training logic with batching, loss calculation, backpropagation) ...
        
        cten_begin_malloc(PoolId_Default); // for temporary tensors in each step

        // ... create input and y_true tensors ...

        optim_adam_zerograd(optimizer);
        Tensor y_pred = Model_forward(&model, input);
        
        // Combined Loss
        Tensor huber = nn_huber_loss(y_true, y_pred, 1.0f);
        Tensor mae = nn_mae_loss(y_true, y_pred);
        Tensor loss = Tensor_add(huber, Tensor_mulf(mae, 0.3f));

        Tensor_backward(loss, Tensor_ones((TensorShape){1}, false));
        
        // Gradient Clipping
        cten_clip_grad_norm((Tensor*)&model, 6, 5.0f);

        optim_adam_step(optimizer);
        
        cten_end_malloc();
        cten_free(PoolId_Default); // free temporary tensors
    }

    // Evaluate model
    cten_begin_eval();
    // ... (evaluation logic) ...
    cten_end_eval();

    // Free memory pools
    cten_free(PoolId_Optimizer);
    cten_free(PoolId_Model); 

    cten_finalize();
    return 0;
}
```

## API Overview

### Tensor Creation and Management

```c
// Basic tensor creation
Tensor Tensor_new(TensorShape shape, bool requires_grad);
Tensor Tensor_zeros(TensorShape shape, bool requires_grad);
Tensor Tensor_ones(TensorShape shape, bool requires_grad);

// Tensor manipulation
Tensor Tensor_transpose(Tensor self);
Tensor Tensor_detach(Tensor self);
Tensor Tensor_unsqueeze(Tensor self, int dim);

// Element access
float Tensor_get(Tensor self, int i, int j, int k, int l);
void Tensor_set(Tensor self, int i, int j, int k, int l, float value);

// Backpropagation
void Tensor_backward(Tensor self, Tensor grad);
```

### Basic Operations

```c
// Element-wise operations with tensors
Tensor Tensor_add(Tensor self, Tensor other);
Tensor Tensor_sub(Tensor self, Tensor other);
Tensor Tensor_mul(Tensor self, Tensor other);
Tensor Tensor_div(Tensor self, Tensor other);
Tensor Tensor_pow(Tensor self, Tensor other);

// Element-wise operations with scalars
Tensor Tensor_addf(Tensor self, float other);
Tensor Tensor_subf(Tensor self, float other);
Tensor Tensor_mulf(Tensor self, float other);
Tensor Tensor_divf(Tensor self, float other);
Tensor Tensor_powf(Tensor self, float other);

// Matrix operations
Tensor Tensor_matmul(Tensor self, Tensor other);

// Unary operations
Tensor Tensor_neg(Tensor self);
Tensor Tensor_abs(Tensor self);
Tensor Tensor_square(Tensor self);
Tensor Tensor_reciprocal(Tensor self);
```

### Mathematical Functions

```c
// Logarithmic and exponential
Tensor nn_log(Tensor self);
Tensor nn_exp(Tensor self);

// Trigonometric functions
Tensor nn_sin(Tensor self);
Tensor nn_cos(Tensor self);
Tensor nn_tan(Tensor self);
```

### Reduction Operations

```c
// Reduction operations (with macro dispatch)
Tensor Tensor_sum(Tensor self);           // Sum all elements
Tensor Tensor_sum(Tensor self, int dim);  // Sum along dimension

Tensor Tensor_mean(Tensor self);          // Mean of all elements
Tensor Tensor_mean(Tensor self, int dim); // Mean along dimension

Tensor Tensor_max(Tensor self);           // Max of all elements
TensorMaxMinResult Tensor_max(Tensor self, int dim); // Max along dimension

Tensor Tensor_min(Tensor self);           // Min of all elements
TensorMaxMinResult Tensor_min(Tensor self, int dim); // Min along dimension

// Argmax operation
void Tensor_argmax(Tensor self, int* out);
```

### Neural Network Functions

```c
// Neural network layers
Tensor nn_linear(Tensor input, Tensor weight, Tensor bias);

// Activation functions
Tensor nn_relu(Tensor input);
Tensor nn_sigmoid(Tensor input);
Tensor nn_tanh(Tensor input);
Tensor nn_elu(Tensor self, float alpha);
Tensor nn_selu(Tensor self);
Tensor nn_softmax(Tensor input, int dim);

// Loss functions
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);
Tensor nn_softmax_crossentropy(Tensor y_true, Tensor logits);
Tensor nn_mse_loss(Tensor y_true, Tensor y_pred);
Tensor nn_mae_loss(Tensor y_true, Tensor y_pred);
Tensor nn_huber_loss(Tensor y_true, Tensor y_pred, float delta);

// Weight initialization
Tensor Glorot_init(TensorShape shape, bool requires_grad);
```

### Optimizers

```c
// SGD Optimizer
optim_sgd* optim_sgd_new(int n_params, Tensor* params, float weight_decay);
void optim_sgd_config(optim_sgd* self, float lr, float momentum);
void optim_sgd_zerograd(optim_sgd* self);
void optim_sgd_step(optim_sgd* self);

// Adam Optimizer
optim_adam* optim_adam_new(int n_params, Tensor* params, float lr, 
                          float β1, float β2, float ε, float weight_decay);
void optim_adam_zerograd(optim_adam* self);
void optim_adam_step(optim_adam* self);

// RMSProp Optimizer
optim_rmsprop* optim_rmsprop_new(int n_params, Tensor* params, float lr, 
                                float β, float ε, float weight_decay);
void optim_rmsprop_zerograd(optim_rmsprop* self);
void optim_rmsprop_step(optim_rmsprop* self);

// AdaGrad Optimizer
optim_adagrad* optim_adagrad_new(int n_params, Tensor* params, float lr, 
                                float ε, float weight_decay);
void optim_adagrad_zerograd(optim_adagrad* self);
void optim_adagrad_step(optim_adagrad* self);
```

### Gradient Clipping

```c
// Gradient clipping functions
void cten_clip_grad_norm(Tensor* params, int n_params, float max_norm);
void cten_clip_grad_value(Tensor* params, int n_params, float max_value);
void cten_clip_grad_value_range(Tensor* params, int n_params, float min_value, float max_value);
void cten_clip_grad_positive(Tensor* params, int n_params, float max_value);
void cten_clip_grad_negative(Tensor* params, int n_params, float min_value);
```

### Utility Functions

```c
// TensorShape utilities
int TensorShape_numel(TensorShape shape);
int TensorShape_dim(TensorShape shape);
int TensorShape_asdim(TensorShape shape, int dim);
int TensorShape_tostring(TensorShape shape, char* buf, int size);

// Dataset utilities
int load_iris_dataset(const float (**X)[4], const int** y);
void Tensor_normalize_dataset(const float (*X)[4], float (*X_norm)[4], int n_samples, int n_train_samples, int n_features);
void Tensor_shuffle_dataset(const float (*X)[4], const int *y, float (*X_shuffled)[4], int *y_shuffled, int n_samples, int n_features);

// Evaluation mode
void cten_begin_eval();
bool cten_is_eval();
void cten_end_eval();

// Broadcasting
bool cten_elemwise_broadcast(Tensor* a, Tensor* b);
Tensor reduce_gradient_for_broadcasting(Tensor grad, TensorShape original_shape, 
                                       TensorShape broadcasted_shape);
```

## Memory Management

cTensor uses a pool-based memory management system to efficiently handle tensor allocations:

```c
void cten_initilize();
void cten_finalize();
void cten_begin_malloc(PoolId id);
void cten_end_malloc();
void cten_free(PoolId id);
```

## Project Structure

```
cTensor/
├── include/          # Header files defining the API
│   └── cten.h       # Complete API header
├── src/             # Core implementation files
│   ├── basic.c      # Basic tensor operations
│   ├── nn.c         # Neural network primitives
│   ├── operator.c   # Mathematical operators
│   ├── context.c    # Memory management
│   ├── utils.c      # Utility functions
│   ├── optimizer/   # Optimizer implementations
│   └── ...
├── src2/            # Example applications
│   └── main.c       # Sine regression example
└── tests/           # Test suite
```

## Implemented Features Summary

| Category | Components | Status |
|----------|------------|--------|
| **Core Structs** | `Tensor`, `GradNode`, `TensorMaxMinResult` | ✅ |
| **Autograd** | `Tensor_backward`, `requires_grad`, `detach` | ✅ |
| **Tensor Creation** | `Tensor_new`, `zeros`, `ones`, `Glorot_init` | ✅ |
| **Binary Operations** | `add`, `sub`, `mul`, `div`, `pow`, `matmul` | ✅ |
| **Unary Operations** | `neg`, `abs`, `square`, `reciprocal` | ✅ |
| **Math Functions** | `log`, `exp`, `sin`, `cos`, `tan` | ✅ |
| **Aggregations** | `sum`, `mean`, `max`, `min` (with indices) | ✅ |
| **Search/Sort** | `argmax` | ✅ |
| **Shape Operations** | `transpose`, `unsqueeze` | ✅ |
| **NN Layers** | `nn_linear` | ✅ |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `ELU`, `SELU`, `Softmax` | ✅ |
| **Loss Functions** | `CrossEntropy`, `MSE`, `MAE`, `Huber` | ✅ |
| **Optimizers** | `SGD`, `Adam`, `RMSProp`, `AdaGrad` | ✅ |
| **Training Utils** | `Gradient Clipping`, `Evaluation Mode`, `Weight Decay` | ✅ |

## Contributing

Contributions to cTensor are welcome! Key areas for contribution include:

1. **Performance Optimization:** Benchmarking and SIMD implementations
2. **Advanced Layers:** Convolutional and recurrent neural network layers
3. **Documentation:** Examples, tutorials, and API documentation improvements
4. **Testing:** Expanding test coverage and validation on different platforms

## GSoC 2025 Acknowledgments

This project was developed during Google Summer of Code 2025 by [Advait Gaur](https://github.com/Advaitgaur004) under the mentorship of [PrimedErwin](https://github.com/PrimedErwin), [Anurag Bhat](https://github.com/faze-geek), and [blueloveTH](https://github.com/blueloveTH). The project successfully transformed cTensor from a basic prototype into a functional deep learning framework suitable for embedded applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.