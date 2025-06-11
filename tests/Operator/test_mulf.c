#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_mulf_operator() {
    const char* op_name = "mulf";
    PoolId pool_id = 5;

    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar tensor multiplied by float
    {
        const char* tc_name = "mulf_scalar_x_float";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {2.0f};
        float scalar_val = 3.0f;
        float exp_d[] = {6.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Vector tensor multiplied by float
    {
        const char* tc_name = "mulf_vector_x_float";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float scalar_val = 0.5f;
        float exp_d[] = {0.5f, 1.0f, 1.5f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix tensor multiplied by float
    {
        const char* tc_name = "mulf_matrix_x_float";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scalar_val = -2.0f;
        float exp_d[] = {-2.0f, -4.0f, -6.0f, -8.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Tensor multiplied by zero
    {
        const char* tc_name = "mulf_matrix_x_zero";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scalar_val = 0.0f;
        float exp_d[] = {0.0f, 0.0f, 0.0f, 0.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Special Scalar Values
    {
        const char* tc_name = "mulf_special_scalar_values";
        
        // Sub-test 1: Multiplication by zero
        {
            TensorShape m_shape = {2, 3, 0, 0};
            float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float scalar_val = 0.0f;
            float exp_d[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            
            Tensor t1 = create_test_tensor(m_shape, d1, false);
            Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Very small scalar multiplication
        {
            TensorShape v_shape = {4, 0, 0, 0};
            float d1[] = {1000.0f, 2000.0f, 3000.0f, 4000.0f};
            float scalar_val = 1e-6f;
            float exp_d[] = {1e-3f, 2e-3f, 3e-3f, 4e-3f};
            
            Tensor t1 = create_test_tensor(v_shape, d1, false);
            Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Higher Dimensional Tensors
    {
        const char* tc_name = "mulf_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor scalar multiplication
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24];
            for(int i = 0; i < 24; i++) d1[i] = (float)(i + 1);
            
            float scalar_val = 1.5f;
            float exp_d[24];
            for(int i = 0; i < 24; i++) exp_d[i] = d1[i] * scalar_val;
            
            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor scalar multiplication
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[120];
            for(int i = 0; i < 120; i++) d1[i] = (float)(i + 1);
            
            float scalar_val = -0.5f;
            float exp_d[120];
            for(int i = 0; i < 120; i++) exp_d[i] = d1[i] * scalar_val;
            
            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
