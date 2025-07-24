#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_abs_backward() {
    const char* op_name = "abs_backward";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple backward
    {
        const char* tc_name = "Simple_backward";
        // Sub-test 1: Scalar backward
        {
            TensorShape s_shape = {1};
            float d1[] = {-5.0f};
            float exp_grad[] = {-1.0f};
            
            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor z = Tensor_abs(t1);
            
            Tensor_backward(z, (Tensor){0});
            
            Tensor expected_grad = create_test_tensor(s_shape, exp_grad, false);
            compare_tensors(&t1.node->grad, &expected_grad, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Vector with mixed values
        {
            TensorShape v_shape = {5};
            float d1[] = {10.0f, -2.0f, 0.0f, 5.5f, -0.1f};
            float exp_grad[] = {1.0f, -1.0f, 0.0f, 1.0f, -1.0f};
            
            Tensor t1 = create_test_tensor(v_shape, d1, true);
            Tensor z = Tensor_abs(t1);
            Tensor l = Tensor_sum(z);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);
            compare_tensors(&t1.node->grad, &expected_grad, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Matrix backward
        {
            TensorShape m_shape = {2, 2};
            float d1[] = {1.0f, -2.0f, 0.0f, -4.0f};
            float exp_grad[] = {1.0f, -1.0f, 0.0f, -1.0f};
            
            Tensor t1 = create_test_tensor(m_shape, d1, true);
            Tensor z = Tensor_abs(t1);
            Tensor l = Tensor_sum(z);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);
            compare_tensors(&t1.node->grad, &expected_grad, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Multi-dimensional Tensor backward
    {
        const char* tc_name = "Multidim_backward";
        TensorShape shape_3d = {2, 2, 2};
        float data[] = {1.5f, -2.5f, 0.0f, -4.5f, 5.5f, 6.5f, -7.5f, -8.5f};
        float exp_grad[] = {1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f};

        Tensor t = create_test_tensor(shape_3d, data, true);
        Tensor z = Tensor_abs(t);
        Tensor l = Tensor_sum(z);
        Tensor_backward(l, (Tensor){0});

        Tensor expected_grad = create_test_tensor(shape_3d, exp_grad, false);
        compare_tensors(&t.node->grad, &expected_grad, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }


    // Test Case 3: Chained and Complex Graphs
    {   
        const char* tc_name = "Chained_and_Complex_Graphs_backward";
        // Sub-test 1: z = abs(x) * w
        {
            TensorShape shape = {2};
            float x_data[] = {-2.0f, 3.0f};
            float w_data[] = {5.0f, 10.0f};
            // z = abs(x)*w = {2, 3} * {5, 10} = {10, 30}
            // l = sum(z) = 40
            // dl/dx = (dl/dz) * (dz/dx) = {1, 1} * (w * sign(x)) = {5*(-1), 10*1} = {-5, 10}
            // dl/dw = (dl/dz) * (dz/dw) = {1, 1} * abs(x) = {2, 3}
            float exp_grad_x[] = {-5.0f, 10.0f};
            float exp_grad_w[] = {2.0f, 3.0f};
            
            Tensor x = create_test_tensor(shape, x_data, true);
            Tensor w = create_test_tensor(shape, w_data, true);
            
            Tensor abs_x = Tensor_abs(x);
            Tensor prod = Tensor_mul(abs_x, w);
            Tensor l = Tensor_sum(prod);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad_x = create_test_tensor(shape, exp_grad_x, false);
            Tensor expected_grad_w = create_test_tensor(shape, exp_grad_w, false);

            compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
            compare_tensors(&w.node->grad, &expected_grad_w, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: z = abs(x * w)
        {
            TensorShape shape = {2};
            float x_data[] = {-2.0f, 3.0f};
            float w_data[] = {5.0f, -1.0f};
            // y = x*w = {-10, -3}
            // z = abs(y) = {10, 3}
            // l = sum(z) = 13
            // dl/dx = (dl/dz)(dz/dy)(dy/dx) = {1,1} * sign(y) * w = {-1,-1} * {5,-1} = {-5, 1}
            // dl/dw = (dl/dz)(dz/dy)(dy/dw) = {1,1} * sign(y) * x = {-1,-1} * {-2,3} = {2, -3}
            float exp_grad_x[] = {-5.0f, 1.0f};
            float exp_grad_w[] = {2.0f, -3.0f};
            
            Tensor x = create_test_tensor(shape, x_data, true);
            Tensor w = create_test_tensor(shape, w_data, true);
            
            Tensor prod = Tensor_mul(x, w);
            Tensor abs_prod = Tensor_abs(prod);
            Tensor l = Tensor_sum(abs_prod);
            
            Tensor_backward(l, (Tensor){0});
            
            Tensor expected_grad_x = create_test_tensor(shape, exp_grad_x, false);
            Tensor expected_grad_w = create_test_tensor(shape, exp_grad_w, false);

            compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
            compare_tensors(&w.node->grad, &expected_grad_w, op_name, tc_name, 4, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}