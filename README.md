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
  - Basic arithmetic: add, subtract, multiply, matrix multiplication
  - Reduction: sum, mean
  - Element access and manipulation
- **Basic Neural Network Components:**
  - Linear layer
  - ReLU activation
  - Softmax function
  - Cross-entropy loss
  - Basic weight initialization (Glorot)
- **SGD Optimizer:** Basic stochastic gradient descent framework
- **Memory Management:** Pool-based memory allocation system

### Development Roadmap

The following features are planned for implementation:

#### Math Operators
- **Forward Operators:**
  - Division (Tensor_div, Tensor_divf)
  - Power (Tensor_pow, Tensor_powf)
  - Square (Tensor_square)
  - Reciprocal (Tensor_reciprocal)
- **Backward Operators:**
  - Complete gradient functions for all element-wise operations
  - Enhanced broadcasting support

#### Reduction Operations
- **Forward and Backward:**
  - Max (Tensor_max)
  - Min (Tensor_min)
  - Enhanced broadcasting and dimension-specific reductions

#### Broadcasting System
- Stride-based access for optimized tensor operations
- Full-featured broadcasting mechanism for all tensor shapes
- Optimization for memory efficiency and computational performance

#### Activation Functions
- ELU (Exponential Linear Unit)
- SELU (Scaled Exponential Linear Unit)
- Logarithm (nn_log)
- Exponential (nn_exp)
- Trigonometric functions (nn_sin, nn_cos, nn_tan)

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
- Memory usage optimization
- Enhanced memory pooling strategies

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
Tensor Tensor_new(TensorShape shape, bool requires_grad);
Tensor Tensor_zeros(TensorShape shape, bool requires_grad);
Tensor Tensor_ones(TensorShape shape, bool requires_grad);
```

### Basic Operations

```c
Tensor Tensor_add(Tensor self, Tensor other);
Tensor Tensor_sub(Tensor self, Tensor other);
Tensor Tensor_mul(Tensor self, Tensor other);
Tensor Tensor_div(Tensor self, Tensor other);
Tensor Tensor_matmul(Tensor self, Tensor other);
```

### Neural Network Functions

```c
Tensor nn_linear(Tensor input, Tensor weight, Tensor bias);
Tensor nn_relu(Tensor input);
Tensor nn_sigmoid(Tensor input);
Tensor nn_softmax(Tensor input);
Tensor nn_crossentropy(Tensor y_true, Tensor y_pred);
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

## Contributing

Contributions to cTensor are welcome! The project needs implementation of various components as outlined in the Development Roadmap section. Key areas for contribution include:

1. **Math Operators:** Implementing missing math operations (div, pow, square, reciprocal) with both forward and backward passes
2. **Reduction Operations:** Adding max/min reduction operations with proper gradient handling
3. **Broadcasting System:** Enhancing the tensor broadcasting mechanism for better flexibility and performance
4. **Activation Functions:** Implementing additional activation functions (ELU, SELU, log, exp, trigonometric functions)
5. **Loss Functions:** Adding more loss functions (MSE, MAE, Huber) with gradient support
6. **Advanced Optimizers:** Creating additional optimizers beyond SGD (Adam, RMSProp, AdaGrad)
7. **Performance Optimization:** Enhancing computational efficiency through benchmarking and optimizations
8. **Documentation:** Improving examples, tutorials, and API documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.