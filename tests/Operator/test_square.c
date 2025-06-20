#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_square_operator() {
    const char* op_name = "square";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar square (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1: Basic square
        {
            const char* tc_name = "square_scalar_basic";
            float d1[] = {3.0f};
            float exp_d[] = {9.0f}; // 3^2 = 9
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_square(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Square of a negative number
        {
            const char* tc_name = "square_scalar_negative";
            float d1[] = {-4.0f};
            float exp_d[] = {16.0f}; // (-4)^2 = 16
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_square(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Square of zero
        {
            const char* tc_name = "square_scalar_zero";
            float d1[] = {0.0f};
            float exp_d[] = {0.0f}; // 0^2 = 0
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_square(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 4: Square of a fractional number
        {
            const char* tc_name = "square_scalar_fractional";
            float d1[] = {0.5f};
            float exp_d[] = {0.25f}; // 0.5^2 = 0.25
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_square(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 4, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector square operations
    {
        const char* tc_name = "square_vector";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float exp_d[] = {1.0f, 4.0f, 9.0f}; // [1^2, 2^2, 3^2]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix square operations
    {
        const char* tc_name = "square_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.5f, -2.0f, 3.0f, 0.0f};
        float exp_d[] = {2.25f, 4.0f, 9.0f, 0.0f}; // [1.5^2, (-2)^2, 3^2, 0^2]
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor square operations
    {
        const char* tc_name = "square_3d_tensor";
        TensorShape t_shape = {2, 2, 2, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float exp_d[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f}; // [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 8^2]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
