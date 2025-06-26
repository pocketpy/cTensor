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
        TensorShape s_shape = {1};
        const char* tc_name = "reciprocal_scalar_basic";
        // Sub-test 1: Basic reciprocal
        {
            float d[] = {6.754841f};
            float exp_d[] = {0.148042f}; // 1 / 6.754841 = 0.148042
            Tensor t1 = create_test_tensor(s_shape, d, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_reciprocal(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Reciprocal of a large number
        {
            float d[] = {188.0f};
            float exp_d[] = {0.00535f}; // 1/188 = 0.00535
            Tensor t1 = create_test_tensor(s_shape, d, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_reciprocal(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

    }

    // Test Case 2: Vector reciprocal operations
    {
        const char* tc_name = "reciprocal_vector_elements";
        TensorShape v_shape = {3};
        float d[] = {4.370861f, 9.556429f, 7.587945f};
        float exp_d[] = {0.228788f, 0.104642f, 0.131788f}; // [1/4.370861, 1/9.556429, 1/7.587945]
        Tensor t1 = create_test_tensor(v_shape, d, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix reciprocal operations
    {
        const char* tc_name = "reciprocal_matrix_2x2";
        TensorShape m_shape = {2, 2};
        float d[] = {6.387926f, 2.404168f, 2.403951f, 1.522753f};
        float exp_d[] = {0.156545f, 0.415944f, 0.415982f, 0.656706f}; // [1/6.387926, 1/2.404168, 1/2.403951, 1/1.522753]
        Tensor t1 = create_test_tensor(m_shape, d, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor reciprocal operations
    {
        const char* tc_name = "reciprocal_3d_tensor";
        TensorShape t_shape = {2, 2, 2};
        float d[] = {8.795585f, 6.410035f, 7.372653f, 1.185260f, 9.729189f, 8.491984f, 2.911052f, 2.636425f};
        float exp_d[] = {0.113693f, 0.156005f, 0.135636f, 0.843696f, 0.102783f, 0.117758f, 0.343518f, 0.379302f};
        // exp_d = [1/8.795585, 1/6.410035, 1/7.372653, 1/1.185260, 1/9.729189, 1/8.491984, 1/2.911052, 1/2.636425], 1/9
        Tensor t1 = create_test_tensor(t_shape, d, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Reciprocal of near-zero value (numerical stability test)
    {
        const char* tc_name = "reciprocal_near_zero";
        TensorShape s_shape = {1};
        float d1[] = {1e-6f}; // Very small number
        float exp_d[] = {1e6f}; // 1 / (1e-6) = 1e6
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_reciprocal(t1);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, 1e-1f); // Using a larger tolerance due to floating point imprecision
    }

   // Test Case 6: Reciprocal of negative numbers
   {
    const char* tc_name = "reciprocal_negative";
    TensorShape s_shape = {1};
    float d1[] = {-19.352466f};
    float exp_d[] = {-0.051673f}; // 1 / (-19.3525) = -0.0517
    Tensor t1 = create_test_tensor(s_shape, d1, false);
    Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
    Tensor actual_res = Tensor_reciprocal(t1);

    compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

// Test Case 7: 4D tensor reciprocal operations
{
    const char* tc_name = "reciprocal_4d_tensor";
    TensorShape t_shape = {2, 1, 2, 1}; // 2x1x2x1 tensor
    float d1[] = {19.063572f, 14.907885f, 12.374511f, 3.964354f};
    float exp_d[] = {0.052456f, 0.067079f, 0.080811f, 0.252248f};// Expected: [1/d1[0], 1/d1[1], ...]
    Tensor t1 = create_test_tensor(t_shape, d1, false);
    Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
    Tensor actual_res = Tensor_reciprocal(t1);

    compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
}


// Test Case 8: Mixed positive and negative values
{
    const char* tc_name = "reciprocal_mixed_signs";
    TensorShape v_shape = {4};
    float d1[] = {-17.200274f, -22.095819f, 18.308807f, 5.055750f};
    float exp_d[] = {-0.058139f, -0.045257f, 0.054619f, 0.197795f};
    Tensor t1 = create_test_tensor(v_shape, d1, false);
    Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
    Tensor actual_res = Tensor_reciprocal(t1);

    compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
}

// Test Case 9: Reciprocal of fractional numbers
{
    const char* tc_name = "reciprocal_fractional";
    TensorShape v_shape = {3};
    float d1[] = {0.737265f, 0.118526f, 0.972919f};
    float exp_d[] = {1.356364f, 8.436964f, 1.027835f}; // Reciprocal of fractional numbers results in values > 1.0
    Tensor t1 = create_test_tensor(v_shape, d1, false);
    Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
    Tensor actual_res = Tensor_reciprocal(t1);

    compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
}

// Test Case 10: Reciprocal of very large numbers (testing for underflow)
{
    const char* tc_name = "reciprocal_large_numbers";
    TensorShape s_shape = {1};
    float d1[] = {8.341182e+06f};
    float exp_d[] = {1.198871e-07f}; // 1 / (8.34e+06) = 1.20e-07
    Tensor t1 = create_test_tensor(s_shape, d1, false);
    Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
    Tensor actual_res = Tensor_reciprocal(t1);

    compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
}

    cten_free(pool_id);
}
