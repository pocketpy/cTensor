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
            float d1[] = {0.3745f, 0.9507f, 0.7320f, 0.5987f, 0.1560f, 0.1560f, 0.0581f, 0.8662f, 0.6011f, 0.7081f, 0.0206f, 0.9699f, 0.8324f, 0.2123f, 0.1818f, 0.1834f, 0.3042f, 0.5248f, 0.4319f, 0.2912f, 0.6119f, 0.1395f, 0.2921f, 0.3664f};
            float d2[] = {0.4561f, 0.7852f, 0.1997f, 0.5142f, 0.5924f, 0.0465f, 0.6075f, 0.1705f, 0.0651f, 0.9489f, 0.9656f, 0.8084f, 0.3046f, 0.0977f, 0.6842f, 0.4402f, 0.1220f, 0.4952f, 0.0344f, 0.9093f, 0.2588f, 0.6625f, 0.3117f, 0.5201f};
            float exp_d[] = {0.8306f, 1.7359f, 0.9317f, 1.1129f, 0.7484f, 0.2024f, 0.6656f, 1.0367f, 0.6662f, 1.6570f, 0.9862f, 1.7783f, 1.1371f, 0.3100f, 0.8661f, 0.6236f, 0.4263f, 1.0199f, 0.4663f, 1.2005f, 0.8706f, 0.8020f, 0.6039f, 0.8864f};

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor addition (same shape)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[] = {0.5467f, 0.1849f, 0.9696f, 0.7751f, 0.9395f, 0.8948f, 0.5979f, 0.9219f, 0.0885f, 0.1960f, 0.0452f, 0.3253f, 0.3887f, 0.2713f, 0.8287f, 0.3568f, 0.2809f, 0.5427f, 0.1409f, 0.8022f, 0.0746f, 0.9869f, 0.7722f, 0.1987f, 0.0055f, 0.8155f, 0.7069f, 0.7290f, 0.7713f, 0.0740f, 0.3585f, 0.1159f, 0.8631f, 0.6233f, 0.3309f, 0.0636f, 0.3110f, 0.3252f, 0.7296f, 0.6376f, 0.8872f, 0.4722f, 0.1196f, 0.7132f, 0.7608f, 0.5613f, 0.7710f, 0.4938f, 0.5227f, 0.4275f, 0.0254f, 0.1079f, 0.0314f, 0.6364f, 0.3144f, 0.5086f, 0.9076f, 0.2493f, 0.4104f, 0.7556f, 0.2288f, 0.0770f, 0.2898f, 0.1612f, 0.9297f, 0.8081f, 0.6334f, 0.8715f, 0.8037f, 0.1866f, 0.8926f, 0.5393f, 0.8074f, 0.8961f, 0.3180f, 0.1101f, 0.2279f, 0.4271f, 0.8180f, 0.8607f, 0.0070f, 0.5107f, 0.4174f, 0.2221f, 0.1199f, 0.3376f, 0.9429f, 0.3232f, 0.5188f, 0.7030f, 0.3636f, 0.9718f, 0.9624f, 0.2518f, 0.4972f, 0.3009f, 0.2848f, 0.0369f, 0.6096f, 0.5027f, 0.0515f, 0.2786f, 0.9083f, 0.2396f, 0.1449f, 0.4895f, 0.9857f, 0.2421f, 0.6721f, 0.7616f, 0.2376f, 0.7282f, 0.3678f, 0.6323f, 0.6335f, 0.5358f, 0.0903f, 0.8353f, 0.3208f, 0.1865f};
            float d2[] = {0.0408f, 0.5909f, 0.6776f, 0.0166f, 0.5121f, 0.2265f, 0.6452f, 0.1744f, 0.6909f, 0.3867f, 0.9367f, 0.1375f, 0.3411f, 0.1135f, 0.9247f, 0.8773f, 0.2579f, 0.6600f, 0.8172f, 0.5552f, 0.5297f, 0.2419f, 0.0931f, 0.8972f, 0.9004f, 0.6331f, 0.3390f, 0.3492f, 0.7260f, 0.8971f, 0.8871f, 0.7799f, 0.6420f, 0.0841f, 0.1616f, 0.8986f, 0.6064f, 0.0092f, 0.1015f, 0.6635f, 0.0051f, 0.1608f, 0.5487f, 0.6919f, 0.6520f, 0.2243f, 0.7122f, 0.2372f, 0.3254f, 0.7465f, 0.6496f, 0.8492f, 0.6576f, 0.5683f, 0.0937f, 0.3677f, 0.2652f, 0.2440f, 0.9730f, 0.3931f, 0.8920f, 0.6311f, 0.7948f, 0.5026f, 0.5769f, 0.4925f, 0.1952f, 0.7225f, 0.2808f, 0.0243f, 0.6455f, 0.1771f, 0.9405f, 0.9539f, 0.9149f, 0.3702f, 0.0155f, 0.9283f, 0.4282f, 0.9667f, 0.9636f, 0.8530f, 0.2944f, 0.3851f, 0.8511f, 0.3169f, 0.1695f, 0.5568f, 0.9362f, 0.6960f, 0.5701f, 0.0972f, 0.6150f, 0.9901f, 0.1401f, 0.5183f, 0.8774f, 0.7408f, 0.6970f, 0.7025f, 0.3595f, 0.2936f, 0.8094f, 0.8101f, 0.8671f, 0.9132f, 0.5113f, 0.5015f, 0.7983f, 0.6500f, 0.7020f, 0.7958f, 0.8900f, 0.3380f, 0.3756f, 0.0940f, 0.5783f, 0.0359f, 0.4656f, 0.5426f};
            float exp_d[] = {0.5875f, 0.7757f, 1.6471f, 0.7917f, 1.4516f, 1.1213f, 1.2431f, 1.0962f, 0.7794f, 0.5827f, 0.9820f, 0.4629f, 0.7297f, 0.3848f, 1.7534f, 1.2341f, 0.5389f, 1.2027f, 0.9581f, 1.3574f, 0.6042f, 1.2287f, 0.8653f, 1.0959f, 0.9059f, 1.4486f, 1.0459f, 1.0782f, 1.4972f, 0.9712f, 1.2456f, 0.8957f, 1.5051f, 0.7074f, 0.4925f, 0.9621f, 0.9174f, 0.3344f, 0.8311f, 1.3011f, 0.8923f, 0.6330f, 0.6683f, 1.4051f, 1.4127f, 0.7855f, 1.4831f, 0.7310f, 0.8481f, 1.1740f, 0.6751f, 0.9571f, 0.6890f, 1.2047f, 0.4080f, 0.8763f, 1.1728f, 0.4933f, 1.3834f, 1.1486f, 1.1208f, 0.7081f, 1.0846f, 0.6639f, 1.5066f, 1.3006f, 0.8286f, 1.5939f, 1.0844f, 0.2109f, 1.5380f, 0.7165f, 1.7479f, 1.8500f, 1.2329f, 0.4802f, 0.2434f, 1.3554f, 1.2462f, 1.8274f, 0.9706f, 1.3638f, 0.7119f, 0.6072f, 0.9710f, 0.6545f, 1.1124f, 0.8800f, 1.4549f, 1.3990f, 0.9337f, 1.0690f, 1.5775f, 1.2418f, 0.6373f, 0.8192f, 1.1622f, 0.7777f, 1.3066f, 1.2052f, 0.4110f, 0.5722f, 1.7176f, 1.0497f, 1.0120f, 1.4027f, 1.4970f, 0.7436f, 1.4704f, 1.4116f, 0.9396f, 1.5240f, 1.2578f, 0.9703f, 1.0091f, 0.6298f, 0.6686f, 0.8712f, 0.7864f, 0.7292f};

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
