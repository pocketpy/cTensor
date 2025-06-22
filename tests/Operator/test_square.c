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

    // Test Case 5: 4D tensor square operations
    {
        const char* tc_name = "square_4d_tensor";
        TensorShape t_shape = {2, 1, 2, 1}; // 2x1x2x1 tensor
        float d1[] = {0.1f, -0.2f, 0.3f, -0.4f};
        float exp_d[] = {0.01f, 0.04f, 0.09f, 0.16f}; // [0.1^2, (-0.2)^2, 0.3^2, (-0.4)^2]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Very large numbers
    {
        const char* tc_name = "square_large_numbers";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {1000.0f};
        float exp_d[] = {1000000.0f}; // 1000^2 = 1,000,000
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, 0.1f); // Using larger tolerance due to potential floating point imprecision
    }

    // Test Case 7: Very small numbers
    {
        const char* tc_name = "square_small_numbers";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {1e-4f};
        float exp_d[] = {1e-8f}; // (1e-4)^2 = 1e-8
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 8: Mixed positive and negative values
    {
        const char* tc_name = "square_mixed_sign_vector";
        TensorShape v_shape = {5, 0, 0, 0};
        float d1[] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
        float exp_d[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f}; // [1^2, (-2)^2, 3^2, (-4)^2, 5^2]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 9: Larger matrix
    {
        const char* tc_name = "square_3x3_matrix";
        TensorShape m_shape = {3, 3, 0, 0};
        float d1[] = {
            1.0f, 2.0f, 3.0f,
            -4.0f, -5.0f, -6.0f,
            0.1f, 0.2f, 0.3f
        };
        float exp_d[] = {
            1.0f, 4.0f, 9.0f,
            16.0f, 25.0f, 36.0f,
            0.01f, 0.04f, 0.09f
        }; // Square of each element
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 10: Tensor with zeros and ones
    {
        const char* tc_name = "square_zeros_and_ones";
        TensorShape v_shape = {4, 0, 0, 0};
        float d1[] = {0.0f, 1.0f, 0.0f, 1.0f};
        float exp_d[] = {0.0f, 1.0f, 0.0f, 1.0f}; // [0^2, 1^2, 0^2, 1^2]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_square(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
