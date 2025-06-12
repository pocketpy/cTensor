#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_mul_operator() {
    const char* op_name = "mul";
    PoolId pool_id = 1;

    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar multiplication (1x1 tensors)
    {
        const char* tc_name = "mul_scalar";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {2.0f}; 
        float d2[] = {3.0f};
        float exp_d[] = {6.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_mul(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Vector element-wise multiplication
    {
        const char* tc_name = "mul_vector_1D";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float d2[] = {4.0f, 5.0f, 0.5f};
        float exp_d[] = {4.0f, 10.0f, 1.5f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_mul(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix element-wise multiplication
    {
        const char* tc_name = "mul_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float d2[] = {0.5f, 2.0f, -1.0f, 0.25f};
        float exp_d[] = {0.5f, 4.0f, -3.0f, 1.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_mul(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Broadcasting (matrix * scalar-like tensor)
    // Example: [[1,2],[3,4]] * [2] (shape {1}) -> PyTorch result: [[2,4],[6,8]]
    {
        const char* tc_name = "mul_broadcast_matrix_by_scalar_tensor";
        TensorShape mat_shape = {2, 2, 0, 0}; float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; float scalar_data[] = {2.0f};
        
        TensorShape expected_shape = {2, 2, 0, 0}; float exp_data[] = {2.0f, 4.0f, 6.0f, 8.0f};

        Tensor t_mat = create_test_tensor(mat_shape, mat_data, false);
        Tensor t_scalar_original = create_test_tensor(scalar_shape, scalar_data, false);
        
        Tensor actual_res = Tensor_mul(t_mat, t_scalar_original); 
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    
    // TODO: Problem in Broadcasting

    // // Test Case 5: Advanced Broadcasting
    // {
    //     const char* tc_name = "mul_advanced_broadcasting";
        
    //     // Sub-test 1: Multi-dimensional broadcasting {3,1} * {1,4} -> {3,4}
    //     {
    //         TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {2.0f, 3.0f, 4.0f};
    //         TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //         TensorShape exp_shape = {3, 4, 0, 0}; 
    //         float exp_d[] = {2.0f, 4.0f, 6.0f, 8.0f,    // 2*[1,2,3,4]
    //                          3.0f, 6.0f, 9.0f, 12.0f,   // 3*[1,2,3,4]
    //                          4.0f, 8.0f, 12.0f, 16.0f}; // 4*[1,2,3,4]

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_mul(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 4D broadcasting {1,2,3,4} * {5,1,1,1} -> {5,2,3,4}
    //     {
    //         TensorShape s1_shape = {1, 2, 3, 4}; 
    //         float d1[] = {
    //             0.1254f, 0.7612f, 0.3476f, 0.8791f, 
    //             0.2415f, 0.5832f, 0.6903f, 0.9234f, 
    //             0.1327f, 0.4651f, 0.7561f, 0.5872f, 
    //             0.9135f, 0.2783f, 0.3491f, 0.7392f, 
    //             0.5517f, 0.8253f, 0.6023f, 0.1937f, 
    //             0.4936f, 0.2341f, 0.8745f, 0.5291f
    //         };
            
    //         TensorShape s2_shape = {5, 1, 1, 1}; 
    //         float d2[] = {0.8365f, 0.2471f, 0.9382f, 0.5713f, 0.1648f};
            
    //         TensorShape exp_shape = {5, 2, 3, 4};
            
    //         float exp_d[] = {
    //             // Batch 0
    //             0.1049f, 0.6367f, 0.2908f, 0.7354f,
    //             0.2020f, 0.4878f, 0.5774f, 0.7724f,
    //             0.1110f, 0.3891f, 0.6325f, 0.4912f,
    //             0.7641f, 0.2328f, 0.2920f, 0.6183f,
    //             0.4615f, 0.6904f, 0.5038f, 0.1620f,
    //             0.4129f, 0.1958f, 0.7315f, 0.4426f,
    //             // Batch 1
    //             0.0310f, 0.1881f, 0.0859f, 0.2172f,
    //             0.0597f, 0.1441f, 0.1706f, 0.2282f,
    //             0.0328f, 0.1149f, 0.1868f, 0.1451f,
    //             0.2257f, 0.0688f, 0.0863f, 0.1827f,
    //             0.1363f, 0.2039f, 0.1488f, 0.0479f,
    //             0.1220f, 0.0578f, 0.2161f, 0.1307f,
    //             // Batch 2
    //             0.1177f, 0.7142f, 0.3261f, 0.8248f,
    //             0.2266f, 0.5472f, 0.6476f, 0.8663f,
    //             0.1245f, 0.4364f, 0.7094f, 0.5509f,
    //             0.8570f, 0.2611f, 0.3275f, 0.6935f,
    //             0.5176f, 0.7743f, 0.5651f, 0.1817f,
    //             0.4631f, 0.2196f, 0.8205f, 0.4964f,
    //             // Batch 3
    //             0.0716f, 0.4349f, 0.1986f, 0.5022f,
    //             0.1380f, 0.3332f, 0.3944f, 0.5275f,
    //             0.0758f, 0.2657f, 0.4320f, 0.3355f,
    //             0.5219f, 0.1590f, 0.1994f, 0.4223f,
    //             0.3152f, 0.4715f, 0.3441f, 0.1107f,
    //             0.2820f, 0.1337f, 0.4996f, 0.3023f,
    //             // Batch 4
    //             0.0207f, 0.1254f, 0.0573f, 0.1449f,
    //             0.0398f, 0.0961f, 0.1138f, 0.1522f,
    //             0.0219f, 0.0766f, 0.1246f, 0.0968f,
    //             0.1505f, 0.0459f, 0.0575f, 0.1218f,
    //             0.0909f, 0.1360f, 0.0993f, 0.0319f,
    //             0.0813f, 0.0386f, 0.1441f, 0.0872f
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_mul(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }
    // }

    // Test Case 6: Sign Preservation
    {
        const char* tc_name = "mul_sign_preservation";
        
        // Sub-test 1: Negative number multiplication
        {
            TensorShape v_shape = {2, 0, 0, 0};
            float d1[] = {-1.0f, 1.0f};
            float d2[] = {-2.0f, -3.0f};
            float exp_d[] = {2.0f, -3.0f}; // (-1)*(-2)=2, (1)*(-3)=-3

            Tensor t1 = create_test_tensor(v_shape, d1, false);
            Tensor t2 = create_test_tensor(v_shape, d2, false);
            Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Higher Dimensional Tensors
    {
        const char* tc_name = "mul_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor multiplication (same shape)
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24], d2[24], exp_d[24];
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 1);
                d2[i] = 2.0f;
                exp_d[i] = d1[i] * d2[i];
            }

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
