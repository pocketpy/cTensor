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

    cten_free(pool_id);
}
