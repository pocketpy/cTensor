#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_add_backward() {
    const char* op_name = "add_backward";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar backward (1x1 tensors)
    {
        const char* tc_name = "add_scalar_backward";
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1: Basic scalar backward
        {
            float d1[] = {2.0f};
            float d2[] = {3.0f};
            float exp_grad1[] = {1.0f};  // dz/dx = 1
            float exp_grad2[] = {1.0f};  // dz/dy = 1
            
            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor t2 = create_test_tensor(s_shape, d2, true);
            Tensor z = Tensor_add(t1, t2);  // z = 5.0
            
            // Scalar backward
            Tensor grad_dummy = {0};
            Tensor_backward(z, grad_dummy);
            
            Tensor expected_grad1 = create_test_tensor(s_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(s_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Different scalar values
        {
            float d1[] = {4.0f}; 
            float d2[] = {5.0f};
            float exp_grad1[] = {1.0f};  // dz/dx = 1
            float exp_grad2[] = {1.0f};  // dz/dy = 1
            
            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor t2 = create_test_tensor(s_shape, d2, true);
            Tensor z = Tensor_add(t1, t2);  // z = 9.0
            
            Tensor grad_dummy = {0};
            Tensor_backward(z, grad_dummy);
            
            Tensor expected_grad1 = create_test_tensor(s_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(s_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector sum backward (reduces to scalar)
    {
        const char* tc_name = "add_vector_sum_backward";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float d2[] = {4.0f, 5.0f, 6.0f};
        float exp_grad1[] = {1.0f, 1.0f, 1.0f};  // dz/dx = 1 for each element
        float exp_grad2[] = {1.0f, 1.0f, 1.0f};  // dz/dy = 1 for each element
        
        Tensor t1 = create_test_tensor(v_shape, d1, true);
        Tensor t2 = create_test_tensor(v_shape, d2, true);
        Tensor z = Tensor_add(t1, t2);  // z = [5, 7, 9]
        Tensor z_sum = Tensor_sum_all(z);  // sum to scalar for backward
        
        // Scalar backward
        Tensor grad_dummy = {0};
        Tensor_backward(z_sum, grad_dummy);
        
        Tensor expected_grad1 = create_test_tensor(v_shape, exp_grad1, false);
        Tensor expected_grad2 = create_test_tensor(v_shape, exp_grad2, false);

        compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix sum backward (reduces to scalar)
    {
        const char* tc_name = "add_matrix_sum_backward";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float exp_grad1[] = {1.0f, 1.0f, 1.0f, 1.0f};  // dz/dx = 1 for each element
        float exp_grad2[] = {1.0f, 1.0f, 1.0f, 1.0f};  // dz/dy = 1 for each element
        
        Tensor t1 = create_test_tensor(m_shape, d1, true);
        Tensor t2 = create_test_tensor(m_shape, d2, true);
        Tensor z = Tensor_add(t1, t2);  // z = [[6, 8], [10, 12]]
        Tensor z_sum = Tensor_sum_all(z);  // sum to scalar for backward
        
        Tensor grad_dummy = {0};
        Tensor_backward(z_sum, grad_dummy);
        
        Tensor expected_grad1 = create_test_tensor(m_shape, exp_grad1, false);
        Tensor expected_grad2 = create_test_tensor(m_shape, exp_grad2, false);

        compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Broadcasting sum backward (vector + scalar, reduces to scalar)
    {
        const char* tc_name = "add_broadcast_sum_backward";
        TensorShape vec_shape = {2, 0, 0, 0};
        TensorShape scalar_shape = {1, 0, 0, 0};
        float vec_data[] = {1.0f, 2.0f};
        float scalar_data[] = {3.0f};
        float exp_grad_vec[] = {1.0f, 1.0f};  // dz/dx = 1 for each element
        float exp_grad_scalar[] = {2.0f};     // dz/dy = sum over broadcasted dimensions = 2.0
        
        Tensor t_vec = create_test_tensor(vec_shape, vec_data, true);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, true);
        Tensor z = Tensor_add(t_vec, t_scalar);  // z = [4, 5]
        Tensor z_sum = Tensor_sum_all(z);  // sum to scalar for backward
        
        Tensor grad_dummy = {0};
        Tensor_backward(z_sum, grad_dummy);
        
        Tensor expected_grad_vec = create_test_tensor(vec_shape, exp_grad_vec, false);
        Tensor expected_grad_scalar = create_test_tensor(scalar_shape, exp_grad_scalar, false);

        compare_tensors(&t_vec.node->grad, &expected_grad_vec, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&t_scalar.node->grad, &expected_grad_scalar, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Complex computation graph (chained operations)
    {
        const char* tc_name = "add_complex_graph_backward";
        TensorShape v_shape = {2, 0, 0, 0};
        TensorShape s_shape = {1, 0, 0, 0};
        float x_data[] = {1.0f, 2.0f};
        float y_data[] = {3.0f};
        float w_data[] = {2.0f, 3.0f};
        
        // Expected gradients for z = sum((x + y) * w)
        float exp_grad_x[] = {2.0f, 3.0f};  // dz/dx = w = [2, 3]
        float exp_grad_y[] = {5.0f};        // dz/dy = sum(w) = 2 + 3 = 5
        float exp_grad_w[] = {4.0f, 5.0f};  // dz/dw = (x + y) = [4, 5]
        
        Tensor x = create_test_tensor(v_shape, x_data, true);
        Tensor y = create_test_tensor(s_shape, y_data, true);
        Tensor w = create_test_tensor(v_shape, w_data, true);
        
        Tensor sum = Tensor_add(x, y);  // sum = [4, 5]
        Tensor prod = Tensor_mul(sum, w);  // prod = [8, 15]
        Tensor z = Tensor_sum_all(prod);  // z = 23 (scalar)
        
        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);
        
        Tensor expected_grad_x = create_test_tensor(v_shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(s_shape, exp_grad_y, false);
        Tensor expected_grad_w = create_test_tensor(v_shape, exp_grad_w, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&w.node->grad, &expected_grad_w, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}