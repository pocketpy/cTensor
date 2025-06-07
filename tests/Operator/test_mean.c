#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_mean_operator() {
    const char* op_name = "mean";
    PoolId pool_id = 3; 
    cten_begin_malloc(pool_id);

    TensorShape exp_shape_scalar = {1, 0, 0, 0}; 

    // Test Case 1: Mean of a scalar tensor
    {
        const char* tc_name = "mean_scalar";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {5.0f}; 
        float exp_d[] = {5.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Mean of a vector
    {
        const char* tc_name = "mean_vector_3el";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f}; // Sum = 6, Count = 3, Mean = 2
        float exp_d[] = {2.0f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Mean of a matrix
    {
        const char* tc_name = "mean_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f}; // Sum = 10, Count = 4, Mean = 2.5
        float exp_d[] = {2.5f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    
    // Test Case 4: Mean of a matrix with negative numbers
    {
        const char* tc_name = "mean_matrix_2x2_negative";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {-1.0f, 2.0f, -3.0f, 4.0f}; // Sum = 2, Count = 4, Mean = 0.5
        float exp_d[] = {0.5f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Mean of a tensor with all zeros
    {
        const char* tc_name = "mean_vector_all_zeros";
        TensorShape v_shape = {4, 0, 0, 0};
        float d1[] = {0.0f, 0.0f, 0.0f, 0.0f}; // Sum = 0, Count = 4, Mean = 0
        float exp_d[] = {0.0f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
        Tensor actual_res = Tensor_mean(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
