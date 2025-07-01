#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_min_operator() {
    const char* op_name = "min";
    PoolId pool_id = 0;

    cten_begin_malloc(pool_id);

    // Test Case 1: Min of a scalar tensor
    {
        const char* tc_name = "min_scalar";
        TensorShape s_shape = {1};
        float d1[] = {5.0f};
        float exp_d[] = {5.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Min of a vector tensor
    {
        const char* tc_name = "min_vector";
        TensorShape v_shape = {5};
        float d1[] = {8.0f, 3.0f, 7.0f, 5.0f, 9.0f};
        float exp_d[] = {3.0f}; // Min is 3
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Min of a matrix tensor
    {
        const char* tc_name = "min_matrix";
        TensorShape m_shape = {2, 3};
        float d1[] = {5.0f, 2.0f, 8.0f, 3.0f, 4.0f, 6.0f};
        float exp_d[] = {2.0f}; // Min is 2
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Min of a tensor with negative numbers
    {
        const char* tc_name = "min_vector_negative";
        TensorShape v_shape = {4};
        float d1[] = {-1.0f, -2.0f, -3.0f, -0.5f};
        float exp_d[] = {-3.0f}; // Min is -3.0
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Min with duplicate minimum values
    {
        const char* tc_name = "min_duplicate";
        TensorShape v_shape = {5};
        float d1[] = {5.0f, 2.0f, 8.0f, 2.0f, 7.0f};
        float exp_d[] = {2.0f}; // Min is 2, appears twice
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Min of a 3D tensor
    {
        const char* tc_name = "min_3d_tensor";
        TensorShape t_shape = {2, 2, 2};
        float d1[] = {10.0f, 20.0f, 5.0f, 12.0f, 15.0f, 16.0f, 7.0f, 18.0f};
        float exp_d[] = {5.0f}; // Min is 5
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    cten_free(pool_id);
}
