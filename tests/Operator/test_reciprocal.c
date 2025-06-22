#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_reciprocal_operator() {
    const char* op_name = "reciprocal";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar reciprocal (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1: Basic reciprocal
        {
            const char* tc_name = "reciprocal_scalar_basic";
            float d1[] = {2.0f};
            float exp_d[] = {0.5f}; // 1/2 = 0.5
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_reciprocal(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Reciprocal of a large number
        {
            const char* tc_name = "reciprocal_scalar_large";
            float d1[] = {100.0f};
            float exp_d[] = {0.01f}; // 1/100 = 0.01
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_reciprocal(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Reciprocal of 1
        {
            const char* tc_name = "reciprocal_scalar_one";
            float d1[] = {1.0f};
            float exp_d[] = {1.0f}; // 1/1 = 1
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_reciprocal(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector reciprocal operations
    {
        const char* tc_name = "reciprocal_vector";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 4.0f};
        float exp_d[] = {1.0f, 0.5f, 0.25f}; // [1/1, 1/2, 1/4]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix reciprocal operations
    {
        const char* tc_name = "reciprocal_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 4.0f, 5.0f};
        float exp_d[] = {1.0f, 0.5f, 0.25f, 0.2f}; // [1/1, 1/2, 1/4, 1/5]
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor reciprocal operations
    {
        const char* tc_name = "reciprocal_3d_tensor";
        TensorShape t_shape = {2, 2, 2, 0};
        float d1[] = {1.0f, 2.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        float exp_d[] = {1.0f, 0.5f, 0.25f, 0.2f, 0.1667f, 0.1429f, 0.125f, 0.1111f}; // [1/1, 1/2, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Reciprocal of near-zero value (numerical stability test)
    {
        const char* tc_name = "reciprocal_near_zero";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {1e-6f}; // Very small number
        float exp_d[] = {1e6f}; // 1 / (1e-6) = 1e6
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, 1e-1f); // Using a larger tolerance due to floating point imprecision
    }

    // Test Case 6: Reciprocal of negative numbers
    {
        const char* tc_name = "reciprocal_negative";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {-2.0f};
        float exp_d[] = {-0.5f}; // 1 / (-2) = -0.5
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 7: 4D tensor reciprocal operations
    {
        const char* tc_name = "reciprocal_4d_tensor";
        TensorShape t_shape = {2, 1, 2, 1}; // 2x1x2x1 tensor
        float d1[] = {2.0f, 4.0f, 5.0f, 10.0f};
        float exp_d[] = {0.5f, 0.25f, 0.2f, 0.1f}; // [1/2, 1/4, 1/5, 1/10]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 8: Mixed positive and negative values
    {
        const char* tc_name = "reciprocal_mixed_signs";
        TensorShape v_shape = {4, 0, 0, 0};
        float d1[] = {2.0f, -4.0f, 10.0f, -20.0f};
        float exp_d[] = {0.5f, -0.25f, 0.1f, -0.05f}; // [1/2, 1/(-4), 1/10, 1/(-20)]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 9: Reciprocal of fractional numbers
    {
        const char* tc_name = "reciprocal_fractional";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {0.5f, 0.25f, 0.125f};
        float exp_d[] = {2.0f, 4.0f, 8.0f}; // [1/0.5, 1/0.25, 1/0.125]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 10: Reciprocal of very large numbers (testing for underflow)
    {
        const char* tc_name = "reciprocal_large_numbers";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {1e6f}; // Large number
        float exp_d[] = {1e-6f}; // 1 / (1e6) = 1e-6
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
