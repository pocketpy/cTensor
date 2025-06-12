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

    // Test Case 4: Higher Dimensional Sum
    {
        const char* tc_name = "sum_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor sum along axis 1 (3x4x5 -> 3x5)
        {
            TensorShape shape_3d = {3, 4, 5, 0};
            float d1[] = {0.3831f, 0.5189f, 0.0470f, 0.1663f, 0.7380f, 0.0828f, 0.6032f, 0.2453f, 0.3893f, 0.2887f, 0.3557f, 0.7190f, 0.2971f, 0.5664f, 0.4761f, 0.6637f, 0.9368f, 0.7326f, 0.2149f, 0.0312f, 0.2623f, 0.5951f, 0.0514f, 0.4964f, 0.5968f, 0.3342f, 0.7709f, 0.1066f, 0.0751f, 0.7282f, 0.4955f, 0.6884f, 0.4348f, 0.2464f, 0.8191f, 0.7994f, 0.6947f, 0.2721f, 0.5902f, 0.3610f, 0.0916f, 0.9173f, 0.1368f, 0.9502f, 0.4460f, 0.1851f, 0.5419f, 0.8729f, 0.7322f, 0.8066f, 0.6588f, 0.6923f, 0.8492f, 0.2497f, 0.4894f, 0.2212f, 0.9877f, 0.9441f, 0.0394f, 0.7056f};
            TensorShape exp_shape = {3, 5, 0, 0};
            float exp_d[] = {1.4853f, 2.7779f, 1.3220f, 1.3369f, 1.5340f, 1.8914f, 2.7491f, 0.8650f, 1.4081f, 2.5051f, 1.1567f, 3.1392f, 2.8030f, 1.9716f, 2.4476f};
            
            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sum(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor sum along axis 2 (2x3x4x5 -> 2x3x5)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[] = {0.9252f, 0.1806f, 0.5679f, 0.9155f, 0.0339f, 0.6974f, 0.2973f, 0.9244f, 0.9711f, 0.9443f, 0.4742f, 0.8620f, 0.8445f, 0.3191f, 0.8289f, 0.0370f, 0.5963f, 0.2300f, 0.1206f, 0.0770f, 0.6963f, 0.3399f, 0.7248f, 0.0654f, 0.3153f, 0.5395f, 0.7907f, 0.3188f, 0.6259f, 0.8860f, 0.6159f, 0.2330f, 0.0244f, 0.8701f, 0.0213f, 0.8747f, 0.5289f, 0.9391f, 0.7988f, 0.9979f, 0.3507f, 0.7672f, 0.4019f, 0.4799f, 0.6275f, 0.8737f, 0.9841f, 0.7683f, 0.4178f, 0.4214f, 0.7376f, 0.2388f, 0.1105f, 0.3546f, 0.2872f, 0.2963f, 0.2336f, 0.0421f, 0.0179f, 0.9877f, 0.4278f, 0.3843f, 0.6796f, 0.2183f, 0.9500f, 0.7863f, 0.0894f, 0.4176f, 0.8791f, 0.9447f, 0.4674f, 0.6134f, 0.1670f, 0.9912f, 0.2317f, 0.9427f, 0.6496f, 0.6077f, 0.5127f, 0.2307f, 0.1765f, 0.2205f, 0.1864f, 0.7796f, 0.3501f, 0.0578f, 0.9691f, 0.8838f, 0.9278f, 0.9949f, 0.1739f, 0.3962f, 0.7582f, 0.6960f, 0.1539f, 0.8158f, 0.2244f, 0.2238f, 0.5370f, 0.5929f, 0.5801f, 0.0915f, 0.8775f, 0.2656f, 0.1295f, 0.8887f, 0.9557f, 0.8621f, 0.8095f, 0.6552f, 0.5509f, 0.0870f, 0.4085f, 0.3727f, 0.2598f, 0.7234f, 0.4959f, 0.0810f, 0.2202f, 0.6833f};
            TensorShape exp_shape = {2, 3, 5, 0};
            float exp_d[] = {2.1339f, 1.9362f, 2.5669f, 2.3262f, 1.8841f, 2.7263f, 1.8925f, 2.0070f, 2.3601f, 2.2205f, 2.2583f, 2.2237f, 1.3228f, 1.2701f, 2.3238f, 2.6243f, 1.7368f, 1.8720f, 2.6012f, 2.3570f, 1.2241f, 1.8103f, 2.0523f, 2.9403f, 2.0919f, 2.7431f, 1.6300f, 2.2291f, 1.6680f, 1.7278f};
            
            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sum(t1);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
