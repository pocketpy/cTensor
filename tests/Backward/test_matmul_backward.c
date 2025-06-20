#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_matmul_backward() {
    const char* op_name = "matmul_backward";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Basic matrix multiplication backward (sum to scalar)
    {
        const char* tc_name = "matmul_basic_sum_backward";
        TensorShape a_shape = {2, 3, 0, 0};  // 2x3 matrix
        TensorShape b_shape = {3, 2, 0, 0};  // 3x2 matrix
        
        // A = [[1, 2, 3], [4, 5, 6]]
        float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        // B = [[1, 2], [3, 4], [5, 6]]
        float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        
        // Expected gradients (computed manually):
        // For sum(A @ B), gradients are sum of all partial derivatives
        // dC/dA = ones(2,2) @ B^T = [[1,1],[1,1]] @ [[1,3,5],[2,4,6]] = [[3,7,11],[3,7,11]]
        float exp_grad_a[] = {3.0f, 7.0f, 11.0f, 3.0f, 7.0f, 11.0f};
        // dC/dB = A^T @ ones(2,2) = [[1,4],[2,5],[3,6]] @ [[1,1],[1,1]] = [[5,5],[7,7],[9,9]]
        float exp_grad_b[] = {5.0f, 5.0f, 7.0f, 7.0f, 9.0f, 9.0f};
        
        Tensor A = create_test_tensor(a_shape, a_data, true);
        Tensor B = create_test_tensor(b_shape, b_data, true);
        Tensor C = Tensor_matmul(A, B);  // C = A @ B (2x2 result)
        Tensor C_sum = Tensor_sum(C);  // sum to scalar for backward
        
        Tensor grad_dummy = {0};
        Tensor_backward(C_sum, grad_dummy);
        
        Tensor expected_grad_a = create_test_tensor(a_shape, exp_grad_a, false);
        Tensor expected_grad_b = create_test_tensor(b_shape, exp_grad_b, false);

        compare_tensors(&A.node->grad, &expected_grad_a, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&B.node->grad, &expected_grad_b, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Square matrix multiplication backward (sum to scalar)
    {
        const char* tc_name = "matmul_square_sum_backward";
        TensorShape shape = {2, 2, 0, 0};  // 2x2 matrices
        
        // A = [[1, 2], [3, 4]]
        float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        // B = [[2, 0], [1, 3]]
        float b_data[] = {2.0f, 0.0f, 1.0f, 3.0f};
        
        // Expected gradients for sum(A @ B):
        // dC/dA = ones(2,2) @ B^T = [[1,1],[1,1]] @ [[2,1],[0,3]] = [[2,4],[2,4]]
        float exp_grad_a[] = {2.0f, 4.0f, 2.0f, 4.0f};
        // dC/dB = A^T @ ones(2,2) = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        float exp_grad_b[] = {4.0f, 4.0f, 6.0f, 6.0f};
        
        Tensor A = create_test_tensor(shape, a_data, true);
        Tensor B = create_test_tensor(shape, b_data, true);
        Tensor C = Tensor_matmul(A, B);
        Tensor C_sum = Tensor_sum(C);  // sum to scalar for backward
        
        Tensor grad_dummy = {0};
        Tensor_backward(C_sum, grad_dummy);
        
        Tensor expected_grad_a = create_test_tensor(shape, exp_grad_a, false);
        Tensor expected_grad_b = create_test_tensor(shape, exp_grad_b, false);

        compare_tensors(&A.node->grad, &expected_grad_a, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&B.node->grad, &expected_grad_b, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Rectangular matrix multiplication backward (sum to scalar)
    {
        const char* tc_name = "matmul_rectangular_sum_backward";
        TensorShape a_shape = {3, 2, 0, 0};  // 3x2 matrix
        TensorShape b_shape = {2, 4, 0, 0};  // 2x4 matrix
        
        // A = [[1, 2], [3, 4], [5, 6]]
        float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        // B = [[1, 2, 3, 4], [5, 6, 7, 8]]
        float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        
        // Expected gradients for sum(A @ B):
        // dC/dA = ones(3,4) @ B^T = ones(3,4) @ [[1,5],[2,6],[3,7],[4,8]] = [[10,26],[10,26],[10,26]]
        float exp_grad_a[] = {10.0f, 26.0f, 10.0f, 26.0f, 10.0f, 26.0f};
        // dC/dB = A^T @ ones(3,4) = [[1,3,5],[2,4,6]] @ ones(3,4) = [[9,9,9,9],[12,12,12,12]]
        float exp_grad_b[] = {9.0f, 9.0f, 9.0f, 9.0f, 12.0f, 12.0f, 12.0f, 12.0f};
        
        Tensor A = create_test_tensor(a_shape, a_data, true);
        Tensor B = create_test_tensor(b_shape, b_data, true);
        Tensor C = Tensor_matmul(A, B);  // C = A @ B (3x4 result)
        Tensor C_sum = Tensor_sum(C);  // sum to scalar for backward
        
        Tensor grad_dummy = {0};
        Tensor_backward(C_sum, grad_dummy);
        
        Tensor expected_grad_a = create_test_tensor(a_shape, exp_grad_a, false);
        Tensor expected_grad_b = create_test_tensor(b_shape, exp_grad_b, false);

        compare_tensors(&A.node->grad, &expected_grad_a, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&B.node->grad, &expected_grad_b, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Complex computation graph with matmul
    {
        const char* tc_name = "matmul_complex_graph_backward";
        TensorShape a_shape = {2, 2, 0, 0};
        TensorShape b_shape = {2, 2, 0, 0};
        TensorShape w_shape = {2, 2, 0, 0};
        
        // A = [[1, 2], [3, 4]]
        float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        // B = [[1, 0], [0, 1]] (identity)
        float b_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
        // W = [[2, 1], [1, 2]] (weights)
        float w_data[] = {2.0f, 1.0f, 1.0f, 2.0f};
        
        // Expected gradients for z = sum((A @ B) * W) = sum(A * W) since B is identity
        float exp_grad_a[] = {2.0f, 1.0f, 1.0f, 2.0f};  // dz/dA = W
        float exp_grad_b[] = {6.0f, 6.0f, 10.0f, 10.0f}; // dz/dB = A^T @ W
        float exp_grad_w[] = {1.0f, 2.0f, 3.0f, 4.0f};   // dz/dW = A @ B = A
        
        Tensor A = create_test_tensor(a_shape, a_data, true);
        Tensor B = create_test_tensor(b_shape, b_data, true);
        Tensor W = create_test_tensor(w_shape, w_data, true);
        
        Tensor AB = Tensor_matmul(A, B);  // AB = A (since B is identity)
        Tensor prod = Tensor_mul(AB, W);  // prod = A * W
        Tensor z = Tensor_sum(prod);  // z = sum(A * W) (scalar)
        
        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);
        
        Tensor expected_grad_a = create_test_tensor(a_shape, exp_grad_a, false);
        Tensor expected_grad_b = create_test_tensor(b_shape, exp_grad_b, false);
        Tensor expected_grad_w = create_test_tensor(w_shape, exp_grad_w, false);

        compare_tensors(&A.node->grad, &expected_grad_a, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&B.node->grad, &expected_grad_b, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&W.node->grad, &expected_grad_w, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}