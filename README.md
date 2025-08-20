# cTensor

A lightweight neural network library written in C11 for embedded systems.

## Overview

cTensor is a compact tensor computation library designed for small client-side devices, such as mobile phones, microcontrollers. The library implements automatic differentiation and dynamic compute graph functionality, allowing for efficient training and deployment of neural networks on resource-constrained devices.

## Current Status

This project is under active development. The prototype demonstrates basic tensor operations and neural network functionality using the Iris dataset as an example. Many core mathematical operators and features are still being implemented.

## Features

### Currently Implemented

- **Lightweight C11 Implementation:** Minimal dependencies for wide compatibility
- **Automatic Differentiation Framework:** Basic gradient computation infrastructure
- **Dynamic Compute Graph:** Groundwork for efficient computation flow
- **Basic Tensor Operations:** 
  - Basic arithmetic: add, subtract, multiply, divide, power
  - Element-wise operations: square, reciprocal
  - Matrix multiplication
  - Tensor transpose
- **Reduction Operations:**
  - Sum (all elements or along dimension)
  - Mean (all elements or along dimension)
  - Max (all elements or along dimension with indices)
  - Min (all elements or along dimension with indices)
  - Argmax function
- **Neural Network Components:**
  - Linear layer
  - Activation functions: ReLU, Sigmoid, Softmax
  - Cross-entropy loss
  - Softmax cross-entropy (combined operation)
  - Glorot weight initialization
- **SGD Optimizer:** Stochastic gradient descent implementation
- **Memory Management:** Pool-based memory allocation system
- **Tensor Utilities:**
  - Element access and manipulation
  - Tensor detachment
  - Tensor unsqueeze operation
  - Broadcasting support for element-wise operations
  - Dataset normalization and shuffling utilities

### Development Roadmap

The following features are planned for implementation:

#### Math Operators
- **Unary Operations:**
  - Negative (Tensor_neg)
  - Absolute value (Tensor_abs)
- **Mathematical Functions:**
  - Logarithm (nn_log)
  - Exponential (nn_exp)
  - Trigonometric functions (nn_sin, nn_cos, nn_tan)

#### Broadcasting System Enhancements
- Broadcasting for Matmul

#### Activation Functions
- ELU (Exponential Linear Unit)
- SELU (Scaled Exponential Linear Unit)
- Additional activation functions

#### Loss Functions
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss
- Enhanced multi-class classification losses

#### Advanced Optimizers
- Adam optimizer
- RMSProp optimizer
- AdaGrad optimizer
- Weight decay implementation
- Gradient clipping

#### Performance Enhancements
- Profiling and benchmarking infrastructure
- Loop unrolling and SIMD optimizations where applicable

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

For a more detailed guide, refer to [Testing Documentation](tests/README.md).

```bash
# Build the test executable with CMake
mkdir -p build && cd build
cmake ..
cmake --build .

# Run the tests
./cten_exe
```

## Usage Example

The repository includes a simple example in `src2/main.c` that demonstrates how to train a neural network on the Iris dataset:

```c
#include "cten.h"
#include <stdio.h>

int main() {
    // Initialize cTensor library
    cten_initilize();
    
    // Load the Iris dataset
    const float (*X)[4];
    const int* y;
    int num_samples = load_iris_dataset(&X, &y);
    
    // Create a simple neural network
    TensorShape input_shape = {1, 4, 0, 0};  // 4 features
    TensorShape hidden_shape = {4, 10, 0, 0}; // 10 hidden units
    TensorShape output_shape = {10, 3, 0, 0}; // 3 classes (iris species)
    
    // Initialize network parameters with Glorot initialization
    Tensor W1 = Glorot_init(hidden_shape, true);
    Tensor b1 = Tensor_zeros((TensorShape){1, 10, 0, 0}, true);
    Tensor W2 = Glorot_init(output_shape, true);
    Tensor b2 = Tensor_zeros((TensorShape){1, 3, 0, 0}, true);
    
    // Setup optimizer
    Tensor params[4] = {W1, b1, W2, b2};
    optim_sgd* optimizer = optim_sgd_new(4, params);
    optim_sgd_config(optimizer, 0.01f, 0.9f);
    
    // Training loop
    // ...
    
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
Tensor Tensor_square(Tensor self);
Tensor Tensor_reciprocal(Tensor self);
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
Tensor nn_softmax(Tensor input);

// Loss functions
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);
Tensor nn_softmax_crossentropy(Tensor y_true, Tensor logits);

// Weight initialization
Tensor Glorot_init(TensorShape shape, bool requires_grad);
```

### Optimizer

```c
// SGD Optimizer
optim_sgd* optim_sgd_new(int n_params, Tensor* params);
void optim_sgd_config(optim_sgd* self, float lr, float momentum);
void optim_sgd_zerograd(optim_sgd* self);
void optim_sgd_step(optim_sgd* self);
void optim_sgd_delete(optim_sgd* self);
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
```

## Memory Management

cTensor uses a pool-based memory management system to efficiently handle tensor allocations:

```c
void cten_begin_malloc(PoolId id);
void cten_end_malloc();
void cten_free(PoolId id);
```

## Project Structure

```
cTensor/
├── include/          # Header files defining the API
├── src/              # Core implementation files
│   ├── basic.c       # Basic tensor operations
│   ├── nn.c          # Neural network primitives
│   ├── operator.c    # Mathematical operators
│   └── ...
├── src2/             # Example applications
│   └── main.c        # Iris dataset example
└── tests/            # Test suite
```
## API Reference

For a detailed API reference, refer to [API Documentation](API.md).

## Contributing

Contributions to cTensor are welcome! The project needs implementation of various components as outlined in the Development Roadmap section. Key areas for contribution include:

1. **Activation Functions:** Implementing additional activation functions (ELU, SELU) with gradient support
2. **Loss Functions:** Adding more loss functions (MSE, MAE, Huber) with gradient support
3. **Advanced Optimizers:** Creating additional optimizers beyond SGD (Adam, RMSProp, AdaGrad)
4. **Performance Optimization:** Enhancing computational efficiency through benchmarking and optimizations
5. **Documentation:** Improving examples, tutorials, and API documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.