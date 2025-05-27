# cTensor Tests

This directory contains tests for the cTensor library, comparing operations against PyTorch reference values.

## Running Tests

### Linux
```bash
# Build with GCC
cmake -B build -DCMAKE_C_COMPILER=gcc
cmake --build build
./build/bin/cten_tests

# Build with Clang
cmake -B build -DCMAKE_C_COMPILER=clang
cmake --build build
./build/bin/cten_tests
```

### Windows
```powershell
# Using Visual Studio Developer Command Prompt
cmake -B build
cmake --build build --config Debug
.\build\bin\Debug\cten_tests.exe
```

### macOS
```bash
cmake -B build
cmake --build build
./build/bin/cten_tests
```

## Adding New Operator Tests

1. **Create Reference Values in PyTorch**:
   ```python
   import torch
   
   # Example for a new operator (e.g., subtract)
   a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
   b = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
   result = a - b  # For subtraction
   print(result)  # Copy these values for your test
   ```

2. **Add Test Function**:
   - Open `test_pytorch_ref.c`
   - Add a new test function following this pattern:
   
   ```c
   void test_tensor_sub() {
       // Create input tensors
       float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
       float b_data[] = {0.5f, 1.0f, 1.5f, 2.0f};
       Tensor a = create_tensor(a_data, {2, 2, 0, 0}, false);
       Tensor b = create_tensor(b_data, {2, 2, 0, 0}, false);
       
       // Expected output from PyTorch
       float expected_data[] = {0.5f, 1.0f, 1.5f, 2.0f}; // Values from PyTorch
       Tensor expected_output = create_tensor(expected_data, {2, 2, 0, 0}, false);
       
       // Run cTensor operation
       Tensor result = Tensor_sub(a, b);
       
       // Compare with PyTorch reference
       float tolerance = 1e-6f;
       bool pass = compare_tensors(result, expected_output, tolerance);
       
       if (pass) {
           printf("Test test_tensor_sub: PASS\n");
       } else {
           printf("Test test_tensor_sub: FAIL\n");
           printf("Expected:\n");
           print_tensor(expected_output);
           printf("Got:\n");
           print_tensor(result);
       }
       
       // Clean up
       Tensor_free(a);
       Tensor_free(b);
       Tensor_free(result);
       Tensor_free(expected_output);
   }
   ```

3. **Add Function Call in Main**:
   - Add your test function to `main()` in `test_pytorch_ref.c`:
   
   ```c
   int main() {
       // Initialize cTensor
       cten_initilize();
       cten_begin_malloc();
       
       // Run tests
       test_tensor_add();
       test_tensor_matmul();
       test_tensor_sub(); // Add your new test here
       
       // Clean up
       cten_end_malloc();
       return 0;
   }
   ```

4. **Run Tests**:
   - Rebuild and run tests on all platforms to verify cross-platform compatibility

## CI Testing

The GitHub Actions workflow automatically tests on:
- Linux (GCC and Clang)
- Windows (MSVC)
- macOS (Clang)

Push to the `test` branch to trigger CI testing.
