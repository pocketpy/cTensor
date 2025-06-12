#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_sub_operator() {
    const char* op_name = "sub";
    PoolId pool_id = 2;

    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar subtraction (1x1 tensors)
    {
        const char* tc_name = "sub_scalar";
        TensorShape s_shape = {1, 0, 0, 0};
        float d1[] = {5.0f}; 
        float d2[] = {3.0f};
        float exp_d[] = {2.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_sub(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Vector element-wise subtraction
    {
        const char* tc_name = "sub_vector_1D";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {10.0f, 20.0f, 30.0f};
        float d2[] = {1.0f, 5.0f, 2.5f};
        float exp_d[] = {9.0f, 15.0f, 27.5f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_sub(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix element-wise subtraction
    {
        const char* tc_name = "sub_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float d2[] = {0.5f, 2.0f, -1.0f, 5.0f};
        float exp_d[] = {0.5f, 0.0f, 4.0f, -1.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_sub(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Broadcasting (matrix - scalar-like tensor)
    // Example: [[1,2],[3,4]] - [1] (shape {1}) -> PyTorch result: [[0,1],[2,3]]
    {
        const char* tc_name = "sub_broadcast_matrix_minus_scalar_tensor";
        TensorShape mat_shape = {2, 2, 0, 0}; float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; float scalar_data[] = {1.0f};
        
        TensorShape expected_shape = {2, 2, 0, 0}; float exp_data[] = {0.0f, 1.0f, 2.0f, 3.0f};

        Tensor t_mat = create_test_tensor(mat_shape, mat_data, false);
        Tensor t_scalar_original = create_test_tensor(scalar_shape, scalar_data, false);
        
        Tensor actual_res = Tensor_sub(t_mat, t_scalar_original); 
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // TODO: Problem in Broadcasting
    
    // // Test Case 5: Advanced Broadcasting
    // {
    //     const char* tc_name = "sub_advanced_broadcasting";
        
    //     // Sub-test 1: Multi-dimensional broadcasting {3,1} - {1,4} -> {3,4}
    //     {
    //         TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {10.0f, 20.0f, 30.0f};
    //         TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //         TensorShape exp_shape = {3, 4, 0, 0}; 
    //         float exp_d[] = {9.0f, 8.0f, 7.0f, 6.0f,    // 10-[1,2,3,4]
    //                          19.0f, 18.0f, 17.0f, 16.0f, // 20-[1,2,3,4]
    //                          29.0f, 28.0f, 27.0f, 26.0f}; // 30-[1,2,3,4]

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_sub(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 3D broadcasting {2,3,1} - {1,1,4} -> {2,3,4}
    //     {
    //         TensorShape s1_shape = {2, 3, 1, 0}; 
    //         float d1[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    //         TensorShape s2_shape = {1, 1, 4, 0}; 
    //         float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //         TensorShape exp_shape = {2, 3, 4, 0};
    //         float exp_d[] = {
    //             // First 2x3 slice
    //             9.0f, 8.0f, 7.0f, 6.0f,    // 10-[1,2,3,4]
    //             19.0f, 18.0f, 17.0f, 16.0f, // 20-[1,2,3,4]
    //             29.0f, 28.0f, 27.0f, 26.0f, // 30-[1,2,3,4]
    //             // Second 2x3 slice
    //             39.0f, 38.0f, 37.0f, 36.0f, // 40-[1,2,3,4]
    //             49.0f, 48.0f, 47.0f, 46.0f, // 50-[1,2,3,4]
    //             59.0f, 58.0f, 57.0f, 56.0f  // 60-[1,2,3,4]
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_sub(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }
    //
    //     // Sub-test 3: 4D broadcasting {1,3,1,5} - {2,1,4,1} -> {2,3,4,5}
    //     {
    //         TensorShape s1_shape = {1, 3, 1, 5};
    //         float d1[] = {
    //             0.3745f, 0.9507f, 0.7320f, 0.5987f, 0.1560f,  // [0,0,0,:]
    //             0.1576f, 0.0721f, 0.8381f, 0.5801f, 0.5153f,  // [0,1,0,:]
    //             0.0206f, 0.9699f, 0.8324f, 0.2123f, 0.1818f,  // [0,2,0,:]
    //         };
    //         
    //         TensorShape s2_shape = {2, 1, 4, 1};
    //         float d2[] = {
    //             0.1834f,  // [0,0,0,0]
    //             0.3042f,  // [0,0,1,0]
    //             0.5248f,  // [0,0,2,0]
    //             0.4319f,  // [0,0,3,0]
    //             0.2912f,  // [1,0,0,0]
    //             0.6119f,  // [1,0,1,0]
    //             0.1395f,  // [1,0,2,0]
    //             0.2921f,  // [1,0,3,0]
    //         };
    //         
    //         TensorShape exp_shape = {2, 3, 4, 5};
    //         
    //         float exp_d[] = {
    //             // Batch 0
    //             0.1911f, 0.7673f, 0.5486f, 0.4153f, -0.0274f,
    //             0.0703f, 0.6465f, 0.4278f, 0.2944f, -0.1482f,
    //             -0.1502f, 0.4260f, 0.2072f, 0.0739f, -0.3687f,
    //             -0.0574f, 0.5188f, 0.3000f, 0.1667f, -0.2759f,
    //             
    //             -0.0274f, -0.1253f, 0.6828f, 0.4177f, 0.5247f,
    //             -0.1482f, -0.2462f, 0.5619f, 0.2969f, 0.4038f,
    //             -0.3688f, -0.4667f, 0.3414f, 0.0764f, 0.1833f,
    //             -0.2760f, -0.3739f, 0.4342f, 0.1692f, 0.2761f,
    //             
    //             -0.1628f, 0.7865f, 0.6490f, 0.0289f, -0.0016f,
    //             -0.2837f, 0.6657f, 0.5282f, -0.0919f, -0.1224f,
    //             -0.5042f, 0.4452f, 0.3077f, -0.3124f, -0.3429f,
    //             -0.4114f, 0.5380f, 0.4005f, -0.2196f, -0.2501f,
    //             
    //             // Batch 1
    //             0.0833f, 0.6595f, 0.4408f, 0.3074f, -0.1352f,
    //             -0.2373f, 0.3389f, 0.1201f, -0.0132f, -0.4558f,
    //             0.2350f, 0.8112f, 0.5925f, 0.4592f, 0.0165f,
    //             0.0824f, 0.6586f, 0.4398f, 0.3065f, -0.1361f,
    //             
    //             -0.1352f, -0.2331f, 0.5749f, 0.3099f, 0.4168f,
    //             -0.4559f, -0.5538f, 0.2543f, -0.0107f, 0.0962f,
    //             0.0165f, -0.0814f, 0.7267f, 0.4616f, 0.5686f,
    //             -0.1362f, -0.2341f, 0.5740f, 0.3090f, 0.4159f,
    //             
    //             -0.2706f, 0.6787f, 0.5412f, -0.0789f, -0.1094f,
    //             -0.5913f, 0.3581f, 0.2206f, -0.3995f, -0.4300f,
    //             -0.1189f, 0.8304f, 0.6929f, 0.0728f, 0.0423f,
    //             -0.2716f, 0.6778f, 0.5403f, -0.0798f, -0.1103f,
    //         };
    //
    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_sub(t1, t2);
    //
    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
    //     }
    // }

    // Test Case 6: Order Dependency
    {
        const char* tc_name = "sub_order_dependency";
        
        // Sub-test 1: a - b ≠ b - a verification
        {
            TensorShape v_shape = {2, 0, 0, 0};
            
            // First: [5.0, 3.0] - [2.0, 1.0] = [3.0, 2.0]
            float d1[] = {5.0f, 3.0f};
            float d2[] = {2.0f, 1.0f};
            float exp_d1[] = {3.0f, 2.0f};
            
            Tensor t1 = create_test_tensor(v_shape, d1, false);
            Tensor t2 = create_test_tensor(v_shape, d2, false);
            Tensor expected_res1 = create_test_tensor(v_shape, exp_d1, false);
            Tensor actual_res1 = Tensor_sub(t1, t2);

            compare_tensors(&actual_res1, &expected_res1, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);

            // Second: [2.0, 1.0] - [5.0, 3.0] = [-3.0, -2.0]
            float exp_d2[] = {-3.0f, -2.0f};
            Tensor expected_res2 = create_test_tensor(v_shape, exp_d2, false);
            Tensor actual_res2 = Tensor_sub(t2, t1);

            compare_tensors(&actual_res2, &expected_res2, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Higher Dimensional Tensors
    {
        const char* tc_name = "sub_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor subtraction (same shape)
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24], d2[24], exp_d[24];
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 10);
                d2[i] = (float)(i + 1);
                exp_d[i] = d1[i] - d2[i]; // Should be 9.0 for all elements
            }

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
