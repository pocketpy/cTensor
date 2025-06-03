#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_sum_operator() {
    const char* op_name = "sum";
    PoolId pool_id = 6;

    cten_begin_malloc(pool_id);

    // Test Case 1: Sum of a scalar tensor
    {
        const char* tc_name = "sum_scalar";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {5.0f};
        float exp_d[] = {5.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_sum(t1);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 2: Sum of a vector tensor
    {
        const char* tc_name = "sum_vector";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float exp_d[] = {6.0f}; // Sum is 1+2+3 = 6
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_sum(t1);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 3: Sum of a matrix tensor
    {
        const char* tc_name = "sum_matrix";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float exp_d[] = {10.0f}; // Sum is 1+2+3+4 = 10
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_sum(t1);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 4: Sum of a tensor with negative numbers
    {
        const char* tc_name = "sum_vector_negative";
        TensorShape v_shape = {4, 0, 0, 0};
        float d1[] = {-1.0f, 2.0f, -3.0f, 0.5f};
        float exp_d[] = {-1.5f}; // Sum is -1+2-3+0.5 = -1.5
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_sum(t1);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    cten_free(pool_id);
}
