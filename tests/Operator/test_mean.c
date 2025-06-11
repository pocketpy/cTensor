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
        const char* tc_name = "mean_vector_1D";
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

    // Test Case 6: Large Tensor Reductions
    {
        const char* tc_name = "mean_large_tensor_reductions";
        
        // Sub-test 1: Large tensor mean (1,000 elements)
        {
            TensorShape large_shape = {1000, 0, 0, 0};
            float large_data[1000];
            for(int i = 0; i < 1000; i++) large_data[i] = 1.0f;
            
            float exp_d[] = {1.0f}; // Mean of 1000 ones = 1.0
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(large_shape, large_data, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Large count division (stress test - 5,000 elements)
        {
            TensorShape stress_shape = {5000, 0, 0, 0};
            float stress_data[5000];
            for(int i = 0; i < 5000; i++) stress_data[i] = 2.0f;
            
            float exp_d[] = {2.0f}; // Mean of 5000 twos = 2.0
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(stress_shape, stress_data, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Higher Dimensional Tensors
    {
        const char* tc_name = "mean_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor mean
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24];
            float sum = 0.0f;
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 1);
                sum += d1[i];
            }
            
            float exp_d[] = {sum / 24.0f}; // Mean of 1+2+...+24 = 300/24 = 12.5
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor mean
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[120];
            float sum = 0.0f;
            for(int i = 0; i < 120; i++) {
                d1[i] = (float)(i + 1);
                sum += d1[i];
            }
            
            float exp_d[] = {sum / 120.0f}; // Mean of 1+2+...+120 = 7260/120 = 60.5
            TensorShape exp_shape = {1, 0, 0, 0};
            
            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 8: Edge Cases
    {
        const char* tc_name = "mean_edge_cases";
        
        // Sub-test 1: Single element mean (division by 1)
        {
            TensorShape single_shape = {1, 0, 0, 0};
            float d1[] = {42.5f};
            float exp_d[] = {42.5f}; // Mean of single element is itself
            
            Tensor t1 = create_test_tensor(single_shape, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape_scalar, exp_d, false);
            Tensor actual_res = Tensor_mean(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
