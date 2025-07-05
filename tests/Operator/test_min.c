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
        float d1[] = {2.7885f};
        float exp_d[] = {2.7885f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Min of a vector tensor
    {
        const char* tc_name = "min_vector";
        TensorShape v_shape = {5};
        float d1[] = {-9.4998f, -4.4994f, -5.5358f, 4.7294f, 3.534f};
        float exp_d[] = {-9.4998f};
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
        float d1[] = {7.8436f, -8.2612f, -1.5616f, -9.4041f, -5.6272f, 0.1071f};
        float exp_d[] = {-9.4041f}; 
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
        float d1[] = {-9.7373f, -8.0315f, -3.5661f, -4.6051f};
        float exp_d[] = {-9.7373f};
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
        float d1[] = {1.7853f, -9.87f, -2.7956f, -2.7956f, -9.87f};
        float exp_d[] = {-9.87f}; 
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
        float d1[] = {-6.8904f, 9.1443f, -3.2681f, -8.1451f, -8.0657f, 6.9499f, 2.0745f, 6.1426f};
        float exp_d[] = {-8.1451f};
        TensorShape exp_shape = {1, 0, 0, 0};
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_min(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    cten_free(pool_id);
}
