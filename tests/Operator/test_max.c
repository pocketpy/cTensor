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
        float d1[] = {2.7885f};
        float exp_d[] = {2.7885f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Max of a vector tensor
    {
        const char* tc_name = "max_vector";
        TensorShape v_shape = {5};
        float d1[] = {8.7458f, 4.147f, 0.9326f, 7.1226f, 2.5115f}; 
        float exp_d[] = {8.7458f};
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
        float d1[] = {7.6507f, -6.481f, 2.9918f, -6.1952f, -9.0693f, 4.4308f}; 
        float exp_d[] = {7.6507f};
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
        float d1[] = {-8.687f, -0.9767f, -9.2835f, -6.0498f}; 
        float exp_d[] = {-0.9767f};
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
        float d1[] = {6.1886f, -9.87f, 5.8818f, 5.8818f, 6.1886f}; 
        float exp_d[] = {6.1886f};
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
        float d1[] = {-6.8904f, 9.1443f, -3.2681f, -8.1451f, -8.0657f, 6.9499f, 2.0745f, 6.1426f}; 
        float exp_d[] = {9.1443f};
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_max(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
