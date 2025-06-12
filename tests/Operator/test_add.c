#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_add_operator() {
    const char* op_name = "add";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar addition (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1
        {
            const char* tc_name = "add_scalar";
            float d1[] = {2.5f};
            float d2[] = {3.5f};
            float exp_d[] = {6.0f};
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2
        {
            const char* tc_name = "add_scalar";  
            float d1[] = {10.0f}; 
            float d2[] = {5.0f};
            float exp_d[] = {15.0f}; 
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector addition (1D tensors)
    {
        const char* tc_name = "add_vector_1D";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float d2[] = {4.0f, 5.0f, 6.0f};
        float exp_d[] = {5.0f, 7.0f, 9.0f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_add(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix addition (2D tensors)
    {
        const char* tc_name = "add_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float exp_d[] = {6.0f, 8.0f, 10.0f, 12.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_add(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    
    // Test Case 4: Broadcasting (vector + scalar-like tensor)
    // cTensor's Tensor_add handles broadcasting if cten_elemwise_broadcast modifies inputs.
    // Let's assume for now Tensor_add expects compatible shapes after potential broadcast by cten_elemwise_broadcast.
    // We will test a simple case where one tensor is {1,1} and other is {2,1} (or similar that cten_elemwise_broadcast handles)
    // For this example, let's test adding a {1,0,0,0} tensor to a {2,0,0,0} tensor, expecting failure if not broadcastable or specific error.
    // Given the current structure of Tensor_add, it calls cten_elemwise_broadcast. If that fails, it asserts.
    // A proper test would mock or control cten_elemwise_broadcast, or test shapes it's known to handle.
    // For now, we'll test addition of compatible shapes that might arise from broadcasting.
    // Example: Add [10] (shape {1}) to [[1,2],[3,4]] (shape {2,2}). PyTorch result: [[11,12],[13,14]]
    // cten_elemwise_broadcast might expand [10] to [[10,10],[10,10]] internally before element-wise add.
    // Let's test a compatible case first, then consider how to test broadcast failure or specific broadcast scenarios.
    {
        const char* tc_name = "add_broadcast_vector_plus_scalar_tensor";
        TensorShape vec_shape = {2, 0, 0, 0}; float vec_data[] = {1.0f, 2.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; float scalar_data[] = {10.0f};
        // Expected: PyTorch would broadcast scalar {10} to {10, 10} then add to get {11, 12}
        // We need to know how cten_elemwise_broadcast modifies shapes to set the correct expected_res_shape.
        // Assuming cten_elemwise_broadcast makes 'scalar' compatible with 'vec' for Tensor_add.
        // If cten_elemwise_broadcast makes them both {2}, then expected shape is {2}.
        TensorShape expected_shape = {2, 0, 0, 0}; float exp_data[] = {11.0f, 12.0f};

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar_original = create_test_tensor(scalar_shape, scalar_data, false);
        
        // Tensor_add internally calls cten_elemwise_broadcast(&t_vec, &t_scalar_original)
        // This might modify t_vec.shape or t_scalar_original.shape if they are broadcastable.
        // The result tensor 'actual_res' will have the broadcasted shape.
        Tensor actual_res = Tensor_add(t_vec, t_scalar_original); 
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        // We compare actual_res against an expected_res that has the final broadcasted shape and values.
        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // TODO: Problem in Broadcasting

    // // Test Case 5: Advanced Broadcasting
    // {
    //     const char* tc_name = "add_advanced_broadcasting";
        
    //     // Sub-test 1: Multi-dimensional broadcasting {3,1} + {1,4} -> {3,4}
    //     {
    //         TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f};
    //         TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {10.0f, 20.0f, 30.0f, 40.0f};
    //         TensorShape exp_shape = {3, 4, 0, 0}; 
    //         float exp_d[] = {11.0f, 21.0f, 31.0f, 41.0f,  // 1+[10,20,30,40]
    //                          12.0f, 22.0f, 32.0f, 42.0f,  // 2+[10,20,30,40]
    //                          13.0f, 23.0f, 33.0f, 43.0f}; // 3+[10,20,30,40]

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 3D broadcasting {2,3,1} + {1,1,4} -> {2,3,4}
    //     {
    //         TensorShape s1_shape = {2, 3, 1, 0}; 
    //         float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    //         TensorShape s2_shape = {1, 1, 4, 0}; 
    //         float d2[] = {10.0f, 20.0f, 30.0f, 40.0f};
    //         TensorShape exp_shape = {2, 3, 4, 0};
    //         float exp_d[] = {
    //             // First 2x3 slice
    //             11.0f, 21.0f, 31.0f, 41.0f,  // 1+[10,20,30,40]
    //             12.0f, 22.0f, 32.0f, 42.0f,  // 2+[10,20,30,40]
    //             13.0f, 23.0f, 33.0f, 43.0f,  // 3+[10,20,30,40]
    //             // Second 2x3 slice
    //             14.0f, 24.0f, 34.0f, 44.0f,  // 4+[10,20,30,40]
    //             15.0f, 25.0f, 35.0f, 45.0f,  // 5+[10,20,30,40]
    //             16.0f, 26.0f, 36.0f, 46.0f   // 6+[10,20,30,40]
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 3: 4D broadcasting with size-1 dimensions {1,1,1,1} + {5,4,3,2} -> {5,4,3,2}
    //     {
    //         TensorShape s1_shape = {1, 1, 1, 1}; float d1[] = {5.0f};
    //         TensorShape s2_shape = {5, 4, 3, 2}; 
    //         float d2[120]; // 5*4*3*2 = 120 elements
    //         for(int i = 0; i < 120; i++) d2[i] = (float)(i + 1);
            
    //         TensorShape exp_shape = {5, 4, 3, 2};
    //         float exp_d[120];
    //         for(int i = 0; i < 120; i++) exp_d[i] = d2[i] + 5.0f;

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 4: Vector-to-matrix broadcasting {3} + {2,3} -> {2,3}
    //     {
    //         TensorShape vec_shape = {3, 0, 0, 0}; float vec_data[] = {1.0f, 2.0f, 3.0f};
    //         TensorShape mat_shape = {2, 3, 0, 0}; float mat_data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    //         TensorShape exp_shape = {2, 3, 0, 0}; 
    //         float exp_d[] = {11.0f, 22.0f, 33.0f, 41.0f, 52.0f, 63.0f};

    //         Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
    //         Tensor t_mat = create_test_tensor(mat_shape, mat_data, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t_vec, t_mat);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 4, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 5: Complex broadcasting {1,3,1,5} + {2,3,4,5} -> {2,3,4,5}
    //     {
    //         // Input 1: Shape [2,3,4,5] - first input tensor
    //         TensorShape s1_shape = {2, 3, 4, 5};
    //         float d1[] = {
    //             // First batch
    //             0.1164f, 0.9853f, 0.4897f, 0.3247f, 0.8866f,
    //             0.5937f, 0.1750f, 0.0133f, 0.3366f, 0.4444f,
    //             0.3795f, 0.3648f, 0.6116f, 0.8804f, 0.9854f,
    //             0.9208f, 0.7365f, 0.5698f, 0.8272f, 0.8305f,
                
    //             0.1750f, 0.2792f, 0.8255f, 0.4057f, 0.7042f,
    //             0.9157f, 0.7726f, 0.1742f, 0.0081f, 0.3020f,
    //             0.1216f, 0.6183f, 0.1343f, 0.4055f, 0.7938f,
    //             0.7681f, 0.4561f, 0.2486f, 0.0691f, 0.6973f,
                
    //             0.0169f, 0.7883f, 0.2586f, 0.9287f, 0.4551f,
    //             0.2588f, 0.3459f, 0.8105f, 0.8138f, 0.6058f,
    //             0.1654f, 0.0699f, 0.2755f, 0.8842f, 0.2298f,
    //             0.7816f, 0.0581f, 0.0986f, 0.1502f, 0.5969f,
                
    //             // Second batch
    //             0.5879f, 0.1269f, 0.2776f, 0.1289f, 0.4458f,
    //             0.6098f, 0.9936f, 0.8087f, 0.3615f, 0.1116f,
    //             0.3952f, 0.6604f, 0.0358f, 0.7517f, 0.5065f,
    //             0.3394f, 0.2174f, 0.9798f, 0.6479f, 0.3420f,
                
    //             0.7798f, 0.7025f, 0.5573f, 0.0537f, 0.0663f,
    //             0.6588f, 0.9430f, 0.1685f, 0.1777f, 0.7845f,
    //             0.4107f, 0.4294f, 0.3893f, 0.4155f, 0.7099f,
    //             0.0509f, 0.7379f, 0.8022f, 0.4010f, 0.4805f,
                
    //             0.8146f, 0.2539f, 0.6904f, 0.2082f, 0.5135f,
    //             0.4590f, 0.9003f, 0.7974f, 0.6023f, 0.5157f,
    //             0.2739f, 0.2265f, 0.6506f, 0.9085f, 0.4990f,
    //             0.8915f, 0.7279f, 0.5476f, 0.8812f, 0.8146f
    //         };

    //         // Input 2: Shape [1,3,1,5]
    //         TensorShape s2_shape = {1, 3, 1, 5};
    //         float d2[] = {
    //             0.9676f, 0.8207f, 0.2405f, 0.8951f, 0.8971f,  // [0,0,0,:]
    //             0.0216f, 0.1433f, 0.6033f, 0.8857f, 0.1187f,  // [0,1,0,:]
    //             0.0794f, 0.1209f, 0.5389f, 0.4889f, 0.6299f   // [0,2,0,:]
    //         };

    //         // Expected output shape: [2,3,4,5]
    //         TensorShape exp_shape = {2, 3, 4, 5};
    //         float exp_d[] = {
    //             // First batch
    //             1.0840f, 1.8060f, 0.7301f, 1.2198f, 1.7838f,
    //             1.5612f, 0.9957f, 0.2538f, 1.2317f, 1.3416f,
    //             1.3471f, 1.1854f, 0.8521f, 1.7755f, 1.8825f,
    //             1.8884f, 1.5571f, 0.8102f, 1.7223f, 1.7276f,
                
    //             0.1967f, 0.4225f, 1.4288f, 1.2914f, 0.8229f,
    //             0.9374f, 0.9159f, 0.7775f, 0.8939f, 0.4207f,
    //             0.1432f, 0.7616f, 0.7377f, 1.2912f, 0.9125f,
    //             0.7897f, 0.5994f, 0.8519f, 0.9549f, 0.8160f,
                
    //             0.0964f, 0.9092f, 0.7975f, 1.4176f, 1.0849f,
    //             0.3382f, 0.4668f, 1.3495f, 1.3027f, 1.2356f,
    //             0.2449f, 0.1908f, 0.8144f, 1.3731f, 0.8596f,
    //             0.8610f, 0.1790f, 0.6376f, 0.6390f, 1.2268f,
                
    //             // Second batch
    //             1.5555f, 0.9476f, 0.5181f, 1.0240f, 1.3429f,
    //             1.5774f, 1.8142f, 1.0492f, 1.2566f, 1.0087f,
    //             1.3628f, 1.4811f, 0.2762f, 1.6468f, 1.4036f,
    //             1.3070f, 1.0381f, 1.2203f, 1.5430f, 1.2392f,
                
    //             0.8014f, 0.8458f, 1.1606f, 0.9394f, 0.1850f,
    //             0.6805f, 1.0863f, 0.7718f, 1.0634f, 0.9031f,
    //             0.4323f, 0.5727f, 0.9926f, 1.3012f, 0.8286f,
    //             0.0726f, 0.8813f, 1.4055f, 1.2867f, 0.5991f,
                
    //             0.8940f, 0.3748f, 1.2293f, 0.6971f, 1.1433f,
    //             0.5385f, 1.0212f, 1.3363f, 1.0912f, 1.1455f,
    //             0.3533f, 0.3474f, 1.1895f, 1.3974f, 1.1288f,
    //             0.9710f, 0.8488f, 1.0865f, 1.3701f, 1.4445f
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 5, TEST_FLOAT_TOLERANCE);
    //     }
    // }

    // Test Case 6: Higher Dimensional Tensors
    {
        const char* tc_name = "add_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor addition (same shape)
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24], d2[24], exp_d[24];
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 1);
                d2[i] = (float)(i * 2);
                exp_d[i] = d1[i] + d2[i];
            }

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor addition (same shape)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[120], d2[120], exp_d[120];
            for(int i = 0; i < 120; i++) {
                d1[i] = (float)(i + 1);
                d2[i] = (float)(i + 10);
                exp_d[i] = d1[i] + d2[i];
            }

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor t2 = create_test_tensor(shape_4d, d2, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Gradient Propagation
    {
        const char* tc_name = "add_gradient_propagation";
        
        // Sub-test 1: requires_grad flag propagation
        {
            TensorShape shape = {2, 2, 0, 0};
            float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
            float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
            float exp_d[] = {6.0f, 8.0f, 10.0f, 12.0f};

            Tensor t1 = create_test_tensor(shape, d1, true);  // requires_grad = true
            Tensor t2 = create_test_tensor(shape, d2, false); // requires_grad = false
            Tensor expected_res = create_test_tensor(shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            // Check that result has gradient node when one input requires grad
            // Note: Gradient node creation depends on implementation

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
