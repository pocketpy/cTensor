#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_max_backward() {
    const char* op_name = "max_backward";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Vector with a unique maximum value
    {
        const char* tc_name = "max_vector_unique_backward";
        TensorShape v_shape = {3};
        float data[] = {2.0f, 8.0f, 5.0f};
        float exp_grad[] = {0.0f, 1.0f, 0.0f};
        
        Tensor t = create_test_tensor(v_shape, data, true);
        Tensor z = Tensor_max(t);
        
        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);
        
        Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);
        compare_tensors(&t.node->grad, &expected_grad, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Vector with duplicate maximum values
    {
        const char* tc_name = "max_vector_duplicate_backward";
        TensorShape v_shape = {4};
        float data[] = {9.0f, 3.0f, 9.0f, 1.0f};
        float exp_grad[] = {0.5f, 0.0f, 0.5f, 0.0f};
        
        Tensor t = create_test_tensor(v_shape, data, true);
        Tensor z = Tensor_max(t);
        
        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);
        
        Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);
        compare_tensors(&t.node->grad, &expected_grad, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix with a unique maximum value
    {
        const char* tc_name = "max_matrix_unique_backward";
        TensorShape m_shape = {2, 2};
        float data[] = {1.0f, 2.0f, 10.0f, 4.0f};
        float exp_grad[] = {0.0f, 0.0f, 1.0f, 0.0f};
        
        Tensor t = create_test_tensor(m_shape, data, true);
        Tensor z = Tensor_max(t);
        
        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);
        
        Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);
        compare_tensors(&t.node->grad, &expected_grad, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Complex computation graph (z = max(x) * y)
    {
        const char* tc_name = "max_complex_graph_backward";
        TensorShape v_shape = {3};
        TensorShape s_shape = {1};
        float x_data[] = {1.0f, 5.0f, 2.0f};
        float y_data[] = {4.0f};
        
        // Let m = max(x). z = m * y.
        // dz/dx = dz/dm * dm/dx
        // dz/dm = y = 4.0
        // dm/dx = [0, 1, 0]
        // dz/dx = 4.0 * [0, 1, 0] = [0, 4.0, 0]
        float exp_grad_x[] = {0.0f, 4.0f, 0.0f};
        // dz/dy = m = 5.0
        float exp_grad_y[] = {5.0f};
        
        Tensor x = create_test_tensor(v_shape, x_data, true);
        Tensor y = create_test_tensor(s_shape, y_data, true);
        
        Tensor m = Tensor_max(x);      // m = 5.0
        Tensor z = Tensor_mul(m, y);   // z = 20.0
        
        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);
        
        Tensor expected_grad_x_tensor = create_test_tensor(v_shape, exp_grad_x, false);
        Tensor expected_grad_y_tensor = create_test_tensor(s_shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x_tensor, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y_tensor, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}