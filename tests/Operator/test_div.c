#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void test_div_operator() {
    const char* op_name = "div";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar division (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1: Basic division
        {
            const char* tc_name = "div_scalar_basic";
            float d1[] = {10.0f};
            float d2[] = {2.0f};
            float exp_d[] = {5.0f}; // 10 / 2 = 5
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_div(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Division resulting in fractional value
        {
            const char* tc_name = "div_scalar_fractional";
            float d1[] = {5.0f};
            float d2[] = {2.0f};
            float exp_d[] = {2.5f}; // 5 / 2 = 2.5
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_div(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Division by 1
        {
            const char* tc_name = "div_scalar_by_one";
            float d1[] = {7.0f};
            float d2[] = {1.0f};
            float exp_d[] = {7.0f}; // 7 / 1 = 7
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_div(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector division operations
    {
        const char* tc_name = "div_vector";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {10.0f, 8.0f, 6.0f};
        float d2[] = {2.0f, 4.0f, 3.0f};
        float exp_d[] = {5.0f, 2.0f, 2.0f}; // [10/2, 8/4, 6/3]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix division operations
    {
        const char* tc_name = "div_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {10.0f, 8.0f, 6.0f, 4.0f};
        float d2[] = {2.0f, 4.0f, 3.0f, 2.0f};
        float exp_d[] = {5.0f, 2.0f, 2.0f, 2.0f}; // [10/2, 8/4, 6/3, 4/2]
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor division operations
    {
        const char* tc_name = "div_3d_tensor";
        TensorShape t_shape = {2, 2, 2, 0};
        float d1[] = {10.0f, 8.0f, 6.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.5f};
        float d2[] = {2.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.25f};
        float exp_d[] = {5.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}; // [10/2, 8/4, 6/3, 4/2, 3/1.5, 2/1, 1/0.5, 0.5/0.25]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor t2 = create_test_tensor(t_shape, d2, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Broadcasting (vector divided by scalar)
    {
        const char* tc_name = "div_broadcast_vector_scalar";
        TensorShape vec_shape = {3, 0, 0, 0}; 
        float vec_data[] = {10.0f, 20.0f, 30.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; 
        float scalar_data[] = {2.0f};
        
        // Expected: broadcast scalar {2} to {2, 2, 2} then apply division
        TensorShape expected_shape = {3, 0, 0, 0}; 
        float exp_data[] = {5.0f, 10.0f, 15.0f}; // [10/2, 20/2, 30/2]

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, false);
        
        Tensor actual_res = Tensor_div(t_vec, t_scalar);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Division by zero
    {
        const char* tc_name = "div_by_zero";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {10.0f};
        float d2[] = {0.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);

        Tensor actual_res = Tensor_div(t1, t2);
        // Check if the result is very large (greater than 1e10 in absolute value)
        if (fabs(actual_res.data->flex[0]) < 1e10) {
            fprintf(stderr, "Test %s:%d failed: expected a very large number, got %f\n", tc_name, 1, actual_res.data->flex[0]);
            abort();
        }
    }

    cten_free(pool_id);
}
