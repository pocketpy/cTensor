#include "test_utils.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * PyTorch Reference Tests
 * 
 * These tests compare cTensor operations against hardcoded reference values from PyTorch.
 * Each test ensures that cTensor produces the same results as PyTorch for the same inputs.
 */

// Test tensor addition against PyTorch reference
void test_add_pytorch_ref() {
    printf("Running Test on Tensor_add Operator\n");

    // Input tensors
    float data_a_arr[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorShape shape_a = {2, 3, 0, 0}; 
    Tensor a = create_tensor(data_a_arr, shape_a, false);

    float data_b_arr[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    TensorShape shape_b = {2, 3, 0, 0}; 
    Tensor b = create_tensor(data_b_arr, shape_b, false);

    // PyTorch reference output
    float expected_data_arr[] = {8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f};
    TensorShape expected_shape = {2, 3, 0, 0};
    Tensor expected_output = create_tensor(expected_data_arr, expected_shape, false);

    // Perform cTensor operation
    Tensor result = Tensor_add(a, b);

    // Compare with PyTorch reference
    float tolerance = 1e-6f; 
    bool pass = compare_tensors(result, expected_output, tolerance);

    if (pass) {
        printf("  Test on Tensor_add Operator: PASS\n");
    } else {
        printf("  Test on Tensor_add Operator: FAIL\n");
        printf("    cTensor Result:\n");
        print_tensor_data(result);
        printf("    PyTorch Reference:\n");
        print_tensor_data(expected_output);
    }
}

// Test matrix multiplication against PyTorch reference
void test_matmul_pytorch_ref() {
    printf("Running Test on Tensor_matmul Operator\n");

    // Input tensors
    float data_a_arr[] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorShape shape_a = {2, 2, 0, 0}; 
    Tensor a = create_tensor(data_a_arr, shape_a, false);

    float data_b_arr[] = {5.0f, 6.0f, 7.0f, 8.0f};
    TensorShape shape_b = {2, 2, 0, 0}; 
    Tensor b = create_tensor(data_b_arr, shape_b, false);

    // PyTorch reference output
    float expected_data_arr[] = {19.0f, 22.0f, 43.0f, 50.0f};
    TensorShape expected_shape = {2, 2, 0, 0};
    Tensor expected_output = create_tensor(expected_data_arr, expected_shape, false);

    // Perform cTensor operation
    Tensor result = Tensor_matmul(a, b);

    // Compare with PyTorch reference
    float tolerance = 1e-6f; 
    bool pass = compare_tensors(result, expected_output, tolerance);

    if (pass) {
        printf("  Test on Tensor_matmul Operator: PASS\n");
    } else {
        printf("  Test on Tensor_matmul Operator: FAIL\n");
        printf("    cTensor Result:\n");
        print_tensor_data(result);
        printf("    PyTorch Reference:\n");
        print_tensor_data(expected_output);
    }
}

int main() {
    printf("Starting PyTorch Reference Tests\n\n");
    
    cten_initilize();

    PoolId pool_id = (PoolId)1; 
    cten_begin_malloc(pool_id);

    // Run PyTorch reference tests
    test_add_pytorch_ref();
    test_matmul_pytorch_ref();
    
    // Add more tests for other operators here:

    cten_end_malloc();
    cten_free(pool_id);

    cten_finalize();
    
    printf("\nPyTorch Reference Tests Finished.\n");
    return 0;
}
