#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_sub_backward() {
    const char* op_name = "sub_backward";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple backward (1x1 tensors)
    {
        const char* tc_name = "Simple_backward";
        // Sub-test 1: Scalar backward
        {
            TensorShape s_shape = {1};
            float d1[] = {5.0f};
            float d2[] = {3.0f};
            float exp_grad1[] = {1.0f};  // dz/dx = 1
            float exp_grad2[] = {-1.0f}; // dz/dy = -1
            
            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor t2 = create_test_tensor(s_shape, d2, true);
            Tensor z = Tensor_sub(t1, t2);  // z = 2.0
            
            Tensor_backward(z, (Tensor){0});
            
            Tensor expected_grad1 = create_test_tensor(s_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(s_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Vector sub backward
        {
            TensorShape v_shape = {3};
            float d1[] = {5.0f, 7.0f, 9.0f};
            float d2[] = {2.0f, 3.0f, 4.0f};
            float exp_grad1[] = {1.0f, 1.0f, 1.0f};
            float exp_grad2[] = {-1.0f, -1.0f, -1.0f};
            
            Tensor t1 = create_test_tensor(v_shape, d1, true);
            Tensor t2 = create_test_tensor(v_shape, d2, true);
            Tensor z = Tensor_sub(t1, t2);
            Tensor l = Tensor_sum(z);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad1 = create_test_tensor(v_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(v_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Matrix sub backward
        {
            TensorShape m_shape = {2, 2};
            float d1[] = {10.0f, 20.0f, 30.0f, 40.0f};
            float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
            float exp_grad1[] = {1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad2[] = {-1.0f, -1.0f, -1.0f, -1.0f};
            
            Tensor t1 = create_test_tensor(m_shape, d1, true);
            Tensor t2 = create_test_tensor(m_shape, d2, true);
            Tensor z = Tensor_sub(t1, t2);
            Tensor l = Tensor_sum(z);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad1 = create_test_tensor(m_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(m_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Broadcasting backward
    {
        const char* tc_name = "Broadcasting_backward";
        // Sub-test 1: Vector - Scalar
        {
            TensorShape vec_shape = {2};
            TensorShape scalar_shape = {1};
            float vec_data[] = {5.0f, 10.0f};
            float scalar_data[] = {3.0f};
            float exp_grad_vec[] = {1.0f, 1.0f};
            float exp_grad_scalar[] = {-2.0f};
            
            Tensor t_vec = create_test_tensor(vec_shape, vec_data, true);
            Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, true);
            Tensor z = Tensor_sub(t_vec, t_scalar);
            Tensor l = Tensor_sum(z);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad_vec = create_test_tensor(vec_shape, exp_grad_vec, false);
            Tensor expected_grad_scalar = create_test_tensor(scalar_shape, exp_grad_scalar, false);

            compare_tensors(&t_vec.node->grad, &expected_grad_vec, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_scalar.node->grad, &expected_grad_scalar, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Matrix - Row Vector
        {
            TensorShape mat_shape = {2, 3};
            TensorShape row_shape = {1, 3};
            float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float row_data[] = {0.1f, 0.2f, 0.3f};
            float exp_grad_mat[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_row[] = {-2.0f, -2.0f, -2.0f};

            Tensor t_mat = create_test_tensor(mat_shape, mat_data, true);
            Tensor t_row = create_test_tensor(row_shape, row_data, true);
            Tensor z = Tensor_sub(t_mat, t_row);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_mat = create_test_tensor(mat_shape, exp_grad_mat, false);
            Tensor expected_grad_row = create_test_tensor(row_shape, exp_grad_row, false);

            compare_tensors(&t_mat.node->grad, &expected_grad_mat, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_row.node->grad, &expected_grad_row, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Matrix - Column Vector
        {
            TensorShape mat_shape = {2, 3};
            TensorShape col_shape = {2, 1};
            float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float col_data[] = {10.0f, 20.0f};
            float exp_grad_mat[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_col[] = {-3.0f, -3.0f};

            Tensor t_mat = create_test_tensor(mat_shape, mat_data, true);
            Tensor t_col = create_test_tensor(col_shape, col_data, true);
            Tensor z = Tensor_sub(t_mat, t_col);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_mat = create_test_tensor(mat_shape, exp_grad_mat, false);
            Tensor expected_grad_col = create_test_tensor(col_shape, exp_grad_col, false);

            compare_tensors(&t_mat.node->grad, &expected_grad_mat, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_col.node->grad, &expected_grad_col, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 3: Random input backward
    {
        const char* tc_name = "Random_input_backward";
        // Sub-test 1: Random vector subtraction
        {
            TensorShape v_shape = {4};
            float d1[] = {2.5f, 3.7f, 1.2f, 8.9f};
            float d2[] = {1.1f, 2.2f, 0.5f, 3.3f};
            float exp_grad1[] = {1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad2[] = {-1.0f, -1.0f, -1.0f, -1.0f};
            
            Tensor t1 = create_test_tensor(v_shape, d1, true);
            Tensor t2 = create_test_tensor(v_shape, d2, true);
            Tensor z = Tensor_sub(t1, t2);
            Tensor l = Tensor_sum(z);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad1 = create_test_tensor(v_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(v_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Random 3D tensor subtraction
        {
            TensorShape tensor3d_shape = {2, 2, 2};
            float data1[] = {1.5f, 2.7f, 3.1f, 4.2f, 5.3f, 6.8f, 7.4f, 8.0f};
            float data2[] = {0.5f, 0.7f, 0.9f, 1.2f, 1.3f, 1.8f, 1.4f, 2.0f};
            float exp_grad1[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad2[] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

            Tensor t1 = create_test_tensor(tensor3d_shape, data1, true);
            Tensor t2 = create_test_tensor(tensor3d_shape, data2, true);
            Tensor z = Tensor_sub(t1, t2);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad1 = create_test_tensor(tensor3d_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(tensor3d_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad, &expected_grad1, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad, &expected_grad2, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 4: Chained operations
    {
        const char* tc_name = "Chained_operations_backward";
        // Sub-test 1: (a-b)*c
        {
            TensorShape v_shape = {2};
            float a_data[] = {3.0f, 4.0f};
            float b_data[] = {1.0f, 2.0f};
            float c_data[] = {2.0f, 3.0f};
            float exp_grad_a[] = {2.0f, 3.0f};
            float exp_grad_b[] = {-2.0f, -3.0f};
            float exp_grad_c[] = {2.0f, 2.0f};
            
            Tensor a = create_test_tensor(v_shape, a_data, true);
            Tensor b = create_test_tensor(v_shape, b_data, true);
            Tensor c = create_test_tensor(v_shape, c_data, true);
            
            Tensor diff = Tensor_sub(a, b);
            Tensor prod = Tensor_mul(diff, c);
            Tensor l = Tensor_sum(prod);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad_a = create_test_tensor(v_shape, exp_grad_a, false);
            Tensor expected_grad_b = create_test_tensor(v_shape, exp_grad_b, false);
            Tensor expected_grad_c = create_test_tensor(v_shape, exp_grad_c, false);

            compare_tensors(&a.node->grad, &expected_grad_a, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&b.node->grad, &expected_grad_b, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&c.node->grad, &expected_grad_c, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}