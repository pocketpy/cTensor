#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>
#include <math.h>

void test_abs_operator() {
    const char* op_name = "abs";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Basic test with mixed positive, negative and zero values
    {
        const char* tc_name = "abs_basic_mixed_values";
        TensorShape shape = {6};
        float d1[] = {-2.5f, -1.0f, 0.0f, 1.0f, 2.5f, -3.0f};
        float exp_d[] = {2.5f, 1.0f, 0.0f, 1.0f, 2.5f, 3.0f};

        Tensor t1 = create_test_tensor(shape, d1, false);
        Tensor expected_res = create_test_tensor(shape, exp_d, false);
        Tensor actual_res = Tensor_abs(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: All negative values
    {
        const char* tc_name = "abs_all_negative";
        TensorShape shape = {4};
        float d1[] = {-1.5f, -2.0f, -0.5f, -10.0f};
        float exp_d[] = {1.5f, 2.0f, 0.5f, 10.0f};

        Tensor t1 = create_test_tensor(shape, d1, false);
        Tensor expected_res = create_test_tensor(shape, exp_d, false);
        Tensor actual_res = Tensor_abs(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: All positive values (should be unchanged)
    {
        const char* tc_name = "abs_all_positive";
        TensorShape shape = {3};
        float d1[] = {1.0f, 5.5f, 0.1f};
        float exp_d[] = {1.0f, 5.5f, 0.1f};

        Tensor t1 = create_test_tensor(shape, d1, false);
        Tensor expected_res = create_test_tensor(shape, exp_d, false);
        Tensor actual_res = Tensor_abs(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 2D Tensor (Matrix)
    {
        const char* tc_name = "abs_matrix_2x3";
        TensorShape m_shape = {2, 3};
        float d1[] = {1.0f, -2.0f, 0.0f, -4.0f, 5.0f, -6.0f};
        float exp_d[] = {1.0f, 2.0f, 0.0f, 4.0f, 5.0f, 6.0f};

        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_abs(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Large and near-zero values
    {
        const char* tc_name = "abs_large_and_near_zero";
        TensorShape shape = {5};
        float d1[] = {-1e8f, 1e-8f, 0.0f, 1e8f, -1e-8f};
        float exp_d[] = {1e8f, 1e-8f, 0.0f, 1e8f, 1e-8f};

        Tensor t1 = create_test_tensor(shape, d1, false);
        Tensor expected_res = create_test_tensor(shape, exp_d, false);
        Tensor actual_res = Tensor_abs(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}