#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_max_operator() {
    const char* op_name = "max";
    PoolId pool_id = 0;
    
    cten_begin_malloc(pool_id);

    // Test Case 1: Max of a scalar tensor
    {
        const char* tc_name = "max_scalar";
        TensorShape s_shape = {1};
        float d1[] = {5.0f};
        float exp_d[] = {5.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Max of a vector tensor
    {
        const char* tc_name = "max_vector";
        TensorShape v_shape = {5};
        float d1[] = {1.0f, 7.0f, 3.0f, 5.0f, 2.0f};
        float exp_d[] = {7.0f}; // Max is 7
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Max of a matrix tensor
    {
        const char* tc_name = "max_matrix";
        TensorShape m_shape = {2, 3};
        float d1[] = {1.0f, 2.0f, 8.0f, 3.0f, 4.0f, 6.0f};
        float exp_d[] = {8.0f}; // Max is 8
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Max of a tensor with negative numbers
    {
        const char* tc_name = "max_vector_negative";
        TensorShape v_shape = {4};
        float d1[] = {-1.0f, -2.0f, -3.0f, -0.5f};
        float exp_d[] = {-0.5f}; // Max is -0.5
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Max with duplicate maximum values
    {
        const char* tc_name = "max_duplicate";
        TensorShape v_shape = {5};
        float d1[] = {1.0f, 9.0f, 3.0f, 9.0f, 2.0f};
        float exp_d[] = {9.0f}; // Max is 9, appears twice
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Max of a 3D tensor
    {
        const char* tc_name = "max_3d_tensor";
        TensorShape t_shape = {2, 2, 2};
        float d1[] = {1.0f, 2.0f, 3.0f, 12.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float exp_d[] = {12.0f}; // Max is 12
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
