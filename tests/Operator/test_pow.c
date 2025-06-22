#include "../../include/cten.h"
#include "../test_utils.h"
#include "../test_config.h"
#include <stdio.h>
#include <math.h>

void test_pow_operator() {
    const char* op_name = "pow";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar power (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1: Basic power
        {
            const char* tc_name = "pow_scalar_basic";
            float d1[] = {2.0f};
            float d2[] = {3.0f};
            float exp_d[] = {8.0f}; // 2^3 = 8
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_pow(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Square root
        {
            const char* tc_name = "pow_scalar_sqrt";
            float d1[] = {9.0f};
            float d2[] = {0.5f};
            float exp_d[] = {3.0f}; // 9^0.5 = 3
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_pow(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Power of 0
        {
            const char* tc_name = "pow_scalar_zero_power";
            float d1[] = {5.0f};
            float d2[] = {0.0f};
            float exp_d[] = {1.0f}; // 5^0 = 1
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_pow(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector power operations
    {
        const char* tc_name = "pow_vector_elements";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float d2[] = {2.0f, 2.0f, 2.0f};
        float exp_d[] = {1.0f, 4.0f, 9.0f}; // [1^2, 2^2, 3^2]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix power operations
    {
        const char* tc_name = "pow_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float d2[] = {3.0f, 2.0f, 1.0f, 0.5f};
        float exp_d[] = {1.0f, 4.0f, 3.0f, 2.0f}; // [1^3, 2^2, 3^1, 4^0.5]
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor power operations
    {
        const char* tc_name = "pow_3d_tensor";
        TensorShape t_shape = {2, 2, 2, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float d2[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        float exp_d[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f}; // [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 8^2]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor t2 = create_test_tensor(t_shape, d2, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Broadcasting (scalar power applied to vector)
    {
        const char* tc_name = "pow_broadcast_vector_scalar";
        TensorShape vec_shape = {3, 0, 0, 0}; 
        float vec_data[] = {2.0f, 3.0f, 4.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; 
        float scalar_data[] = {2.0f}; // power of 2
        
        // Expected: broadcast scalar {2} to {2, 2, 2} then apply power
        TensorShape expected_shape = {3, 0, 0, 0}; 
        float exp_data[] = {4.0f, 9.0f, 16.0f}; // [2^2, 3^2, 4^2]

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, false);
        
        Tensor actual_res = Tensor_pow(t_vec, t_scalar);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Negative base
    {
        const char* tc_name = "pow_negative_base";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {-2.0f};
        float d2[] = {3.0f};
        float exp_d[] = {-8.0f}; // (-2)^3 = -8
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
