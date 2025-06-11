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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Large Tensor Reductions
    {
        const char* tc_name = "sum_large_tensor_reductions";
        
        // Sub-test 1: Large tensor sum (1,000 elements)
        {
            TensorShape large_shape = {1000, 0, 0, 0};
            float large_data[1000];
            for(int i = 0; i < 1000; i++) large_data[i] = 1.0f;
            
            float exp_d[] = {1000.0f}; // Sum of 1000 ones
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(large_shape, large_data, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sum(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Larger tensor sum (stress test - 5,000 elements)
        {
            TensorShape stress_shape = {5000, 0, 0, 0};
            float stress_data[5000];
            for(int i = 0; i < 5000; i++) stress_data[i] = 1.0f;
            
            float exp_d[] = {5000.0f}; // Sum of 5000 ones
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(stress_shape, stress_data, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sum(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Higher Dimensional Tensors
    {
        const char* tc_name = "sum_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor sum
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24];
            float sum = 0.0f;
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 1);
                sum += d1[i];
            }
            
            float exp_d[] = {sum}; // Sum of 1+2+...+24 = 300
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sum(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor sum
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[120];
            float sum = 0.0f;
            for(int i = 0; i < 120; i++) {
                d1[i] = (float)(i + 1);
                sum += d1[i];
            }
            
            float exp_d[] = {sum}; // Sum of 1+2+...+120 = 7260
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sum(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
