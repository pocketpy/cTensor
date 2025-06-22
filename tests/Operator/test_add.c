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

    // Test Case 5: Advanced Broadcasting
    {
        const char* tc_name = "add_advanced_broadcasting";
        
        // Sub-test 1: Multi-dimensional broadcasting {3,1} + {1,4} -> {3,4}
        {
            TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f};
            TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {10.0f, 20.0f, 30.0f, 40.0f};
            TensorShape exp_shape = {3, 4, 0, 0}; 
            float exp_d[] = {11.0f, 21.0f, 31.0f, 41.0f,  // 1+[10,20,30,40]
                             12.0f, 22.0f, 32.0f, 42.0f,  // 2+[10,20,30,40]
                             13.0f, 23.0f, 33.0f, 43.0f}; // 3+[10,20,30,40]

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 3D broadcasting {2,3,1} + {1,1,4} -> {2,3,4}
        {
            TensorShape s1_shape = {2, 3, 1, 0}; 
            float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            TensorShape s2_shape = {1, 1, 4, 0}; 
            float d2[] = {10.0f, 20.0f, 30.0f, 40.0f};
            TensorShape exp_shape = {2, 3, 4, 0};
            float exp_d[] = {
                // First 2x3 slice
                11.0f, 21.0f, 31.0f, 41.0f,  // 1+[10,20,30,40]
                12.0f, 22.0f, 32.0f, 42.0f,  // 2+[10,20,30,40]
                13.0f, 23.0f, 33.0f, 43.0f,  // 3+[10,20,30,40]
                // Second 2x3 slice
                14.0f, 24.0f, 34.0f, 44.0f,  // 4+[10,20,30,40]
                15.0f, 25.0f, 35.0f, 45.0f,  // 5+[10,20,30,40]
                16.0f, 26.0f, 36.0f, 46.0f   // 6+[10,20,30,40]
            };

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: 4D broadcasting with size-1 dimensions {1,1,1,1} + {5,4,3,2} -> {5,4,3,2}
        {
            TensorShape s1_shape = {1, 1, 1, 1}; float d1[] = {5.0f};
            TensorShape s2_shape = {5, 4, 3, 2}; 
            float d2[120]; // 5*4*3*2 = 120 elements
            for(int i = 0; i < 120; i++) d2[i] = (float)(i + 1);
            
            TensorShape exp_shape = {5, 4, 3, 2};
            float exp_d[120];
            for(int i = 0; i < 120; i++) exp_d[i] = d2[i] + 5.0f;

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 4: Complex broadcasting {1,3,1,5} + {2,1,4,1} -> {2,3,4,5}
        {
           // Input 1: Shape [2,1,4,1]
           TensorShape s1_shape = {2, 1, 4, 1};
           float d1[] = {
               0.2343f, 0.6135f, 0.1611f, 0.5962f,  // First batch
               0.5681f, 0.5235f, 0.1218f, 0.4864f   // Second batch
           };
            
           // Input 2: Shape [1,3,1,5]
           TensorShape s2_shape = {1, 3, 1, 5};
           float d2[] = {
               0.2277f, 0.9322f, 0.7883f, 0.1584f, 0.4751f,  // First channel
               0.8497f, 0.6706f, 0.5062f, 0.5672f, 0.8714f,  // Second channel
               0.0536f, 0.7535f, 0.6602f, 0.9550f, 0.0743f   // Third channel
           };
            
           // Expected output shape: [2,3,4,5]
           TensorShape exp_shape = {2, 3, 4, 5};
           float exp_d[] = {
               // Batch 0, Channel 0
               0.4620f, 1.1665f, 1.0226f, 0.3927f, 0.7094f,
               0.8412f, 1.5457f, 1.4018f, 0.7719f, 1.0886f,
               0.3888f, 1.0933f, 0.9494f, 0.3195f, 0.6362f,
               0.8239f, 1.5284f, 1.3845f, 0.7546f, 1.0713f,
                
               // Batch 0, Channel 1
               1.0840f, 0.9049f, 0.7405f, 0.8015f, 1.1057f,
               1.4632f, 1.2841f, 1.1197f, 1.1807f, 1.4849f,
               1.0108f, 0.8317f, 0.6672f, 0.7283f, 1.0325f,
               1.4459f, 1.2668f, 1.1024f, 1.1634f, 1.4676f,
                
               // Batch 0, Channel 2
               0.2879f, 0.9878f, 0.8945f, 1.1892f, 0.3086f,
               0.6671f, 1.3670f, 1.2737f, 1.5685f, 0.6878f,
               0.2147f, 0.9146f, 0.8213f, 1.1161f, 0.2354f,
               0.6498f, 1.3497f, 1.2564f, 1.5511f, 0.6705f,
                
               // Batch 1, Channel 0
               0.7958f, 1.5003f, 1.3564f, 0.7265f, 1.0432f,
               0.7512f, 1.4557f, 1.3118f, 0.6819f, 0.9986f,
               0.3496f, 1.0540f, 0.9101f, 0.2802f, 0.5969f,
               0.7141f, 1.4186f, 1.2747f, 0.6448f, 0.9615f,
                
               // Batch 1, Channel 1
               1.4178f, 1.2387f, 1.0743f, 1.1353f, 1.4395f,
               1.3732f, 1.1941f, 1.0297f, 1.0907f, 1.3949f,
               0.9715f, 0.7924f, 0.6281f, 0.6890f, 0.9932f,
               1.3361f, 1.1570f, 0.9926f, 1.0536f, 1.3578f,
                
               // Batch 1, Channel 2
               0.6217f, 1.3216f, 1.2283f, 1.5230f, 0.6424f,
               0.5771f, 1.2770f, 1.1837f, 1.4785f, 0.5978f,
               0.1754f, 0.8753f, 0.7820f, 1.0768f, 0.1961f,
               0.5400f, 1.2399f, 1.1466f, 1.4414f, 0.5607f
               };

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 4, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 5: Higher Dimensional Tensors
    {
        const char* tc_name = "add_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor addition (same shape)
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[] = {0.3745f, 0.9507f, 0.7320f, 0.5987f, 0.1560f, 0.1560f, 0.0581f, 0.8662f, 0.6011f, 0.7081f, 0.0206f, 0.9699f, 0.8324f, 0.2123f, 0.1818f, 0.1834f, 0.3042f, 0.5248f, 0.4319f, 0.2912f, 0.6119f, 0.1395f, 0.2921f, 0.3664f};
            float d2[] = {0.4561f, 0.7852f, 0.1997f, 0.5142f, 0.5924f, 0.0465f, 0.6075f, 0.1705f, 0.0651f, 0.9489f, 0.9656f, 0.8084f, 0.3046f, 0.0977f, 0.6842f, 0.4402f, 0.1220f, 0.4952f, 0.0344f, 0.9093f, 0.2588f, 0.6625f, 0.3117f, 0.5201f};
            float exp_d[] = {0.8306f, 1.7359f, 0.9317f, 1.1129f, 0.7484f, 0.2025f, 0.6656f, 1.0367f, 0.6662f, 1.6570f, 0.9862f, 1.7783f, 1.1370f, 0.3100f, 0.8660f, 0.6236f, 0.4262f, 1.0200f, 0.4663f, 1.2005f, 0.8707f, 0.8020f, 0.6038f, 0.8865f};

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
            float exp_d[] = {0.5875f, 0.7758f, 1.6472f, 0.7917f, 1.4516f, 1.1213f, 1.2431f, 1.0963f, 0.7794f,
                0.5827f, 0.9819f, 0.4628f, 0.7298f, 0.3848f, 1.7534f, 1.2341f, 0.5388f, 1.2027f,
                0.9581f, 1.3574f, 0.6043f, 1.2288f, 0.8653f, 1.0959f, 0.9059f, 1.4486f, 1.0459f,
                1.0782f, 1.4973f, 0.9711f, 1.2456f, 0.8958f, 1.5051f, 0.7074f, 0.4925f, 0.9622f,
                0.9174f, 0.3344f, 0.8311f, 1.3011f, 0.8923f, 0.6330f, 0.6683f, 1.4051f, 1.4128f,
                0.7856f, 1.4832f, 0.7310f, 0.8481f, 1.1740f, 0.6750f, 0.9571f, 0.6890f, 1.2047f,
                0.4081f, 0.8763f, 1.1728f, 0.4933f, 1.3834f, 1.1487f, 1.1208f, 0.7081f, 1.0846f,
                0.6638f, 1.5066f, 1.3006f, 0.8286f, 1.5940f, 1.0845f, 0.2109f, 1.5381f, 0.7164f,
                1.7479f, 1.8500f, 1.2329f, 0.4803f, 0.2434f, 1.3554f, 1.2462f, 1.8274f, 0.9706f,
                1.3637f, 0.7118f, 0.6072f, 0.9710f, 0.6545f, 1.1124f, 0.8800f, 1.4550f, 1.3990f,
                0.9337f, 1.0690f, 1.5774f, 1.2419f, 0.6373f, 0.8192f, 1.1622f, 0.7777f, 1.3066f,
                1.2052f, 0.4110f, 0.5722f, 1.7177f, 1.0497f, 1.0120f, 1.4027f, 1.4970f, 0.7436f,
                1.4704f, 1.4116f, 0.9396f, 1.5240f, 1.2578f, 0.9703f, 1.0091f, 0.6298f, 0.6686f,
                0.8712f, 0.7864f, 0.7291f};

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor t2 = create_test_tensor(shape_4d, d2, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Gradient Propagation
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
