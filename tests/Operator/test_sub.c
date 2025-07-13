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
        TensorShape s_shape = {1};
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
        TensorShape v_shape = {3};
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
        TensorShape m_shape = {2, 2};
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
        TensorShape mat_shape = {2, 2}; float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorShape scalar_shape = {1}; float scalar_data[] = {1.0f};
        
        TensorShape expected_shape = {2, 2}; float exp_data[] = {0.0f, 1.0f, 2.0f, 3.0f};

        Tensor t_mat = create_test_tensor(mat_shape, mat_data, false);
        Tensor t_scalar_original = create_test_tensor(scalar_shape, scalar_data, false);
        
        Tensor actual_res = Tensor_sub(t_mat, t_scalar_original); 
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    
    // Test Case 5: Advanced Broadcasting
    {
        const char* tc_name = "sub_advanced_broadcasting";
        
        // Sub-test 1: Multi-dimensional broadcasting {3,1} - {1,4} -> {3,4}
        {
            TensorShape s1_shape = {3, 1};
            float d1[] = {10.0f, 20.0f, 30.0f};
            TensorShape s2_shape = {1, 4};
            float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
            TensorShape exp_shape = {3, 4}; 
            float exp_d[] = {9.0f, 8.0f, 7.0f, 6.0f,    // 10-[1,2,3,4]
                             19.0f, 18.0f, 17.0f, 16.0f, // 20-[1,2,3,4]
                             29.0f, 28.0f, 27.0f, 26.0f}; // 30-[1,2,3,4]

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 3D broadcasting {2,3,1} - {1,1,4} -> {2,3,4}
        {
            TensorShape s1_shape = {2, 3, 1};
            float d1[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
            TensorShape s2_shape = {1, 1, 4};
            float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
            TensorShape exp_shape = {2, 3, 4};
            float exp_d[] = {
                // First 2x3 slice
                9.0f, 8.0f, 7.0f, 6.0f,    // 10-[1,2,3,4]
                19.0f, 18.0f, 17.0f, 16.0f, // 20-[1,2,3,4]
                29.0f, 28.0f, 27.0f, 26.0f, // 30-[1,2,3,4]
                // Second 2x3 slice
                39.0f, 38.0f, 37.0f, 36.0f, // 40-[1,2,3,4]
                49.0f, 48.0f, 47.0f, 46.0f, // 50-[1,2,3,4]
                59.0f, 58.0f, 57.0f, 56.0f  // 60-[1,2,3,4]
            };

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    
        // Sub-test 3: 4D broadcasting {1,3,1,5} - {2,1,4,1} -> {2,3,4,5}
        {
            TensorShape s1_shape = {1, 3, 1, 5};            
            TensorShape s2_shape = {2, 1, 4, 1};
            TensorShape exp_shape = {2, 3, 4, 5};
            
            float d1[] = {0.3745f, 0.9507f, 0.732f, 0.5987f, 0.156f, 0.1576f, 0.0721f, 0.8381f, 0.5801f, 0.5153f, 0.0206f, 0.9699f, 0.8324f, 0.2123f, 0.1818f};
            float d2[] = {0.1834f, 0.3042f, 0.5248f, 0.4319f, 0.2912f, 0.6119f, 0.1395f, 0.2921f};

            float exp_d[] = {
                // Batch 0
                0.1911f, 0.7673f, 0.5486f, 0.4153f, -0.0274f,
                0.0703f, 0.6465f, 0.4278f, 0.2945f, -0.1482f,
                -0.1503f, 0.4259f, 0.2072f, 0.0739f, -0.3688f,
                -0.0574f, 0.5188f, 0.3001f, 0.1668f, -0.2759f,
                
                -0.0258f, -0.1113f, 0.6547f, 0.3967f, 0.3319f,
                -0.1466f, -0.2321f, 0.5339f, 0.2759f, 0.2111f,
                -0.3672f, -0.4527f, 0.3133f, 0.0553f, -0.0095f,
                -0.2743f, -0.3598f, 0.4062f, 0.1482f, 0.0834f,
                
                -0.1628f, 0.7865f, 0.649f, 0.0289f, -0.0016f,
                -0.2836f, 0.6657f, 0.5282f, -0.0919f, -0.1224f,
                -0.5042f, 0.4451f, 0.3076f, -0.3125f, -0.343f,
                -0.4113f, 0.538f, 0.4005f, -0.2196f, -0.2501f,
                
                // Batch 1
                0.0833f, 0.6595f, 0.4408f, 0.3075f, -0.1352f,
                -0.2374f, 0.3388f, 0.1201f, -0.0132f, -0.4559f,
                0.235f, 0.8112f, 0.5925f, 0.4592f, 0.0165f,
                0.0824f, 0.6586f, 0.4399f, 0.3066f, -0.1361f,
                
                -0.1336f, -0.2191f, 0.5469f, 0.2889f, 0.2241f,
                -0.4543f, -0.5398f, 0.2262f, -0.0318f, -0.0966f,
                0.0181f, -0.0674f, 0.6986f, 0.4406f, 0.3758f,
                -0.1345f, -0.22f, 0.546f, 0.288f, 0.2232f,
                
                -0.2706f, 0.6787f, 0.5412f, -0.0789f, -0.1094f,
                -0.5913f, 0.358f, 0.2205f, -0.3996f, -0.4301f,
                -0.1189f, 0.8304f, 0.6929f, 0.0728f, 0.0423f,
                -0.2715f, 0.6778f, 0.5403f, -0.0798f, -0.1103f,
            };
    
            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);
    
            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Order Dependency
    {
        const char* tc_name = "sub_order_dependency";
        
        // Sub-test 1: a - b ≠ b - a verification
        {
            TensorShape v_shape = {2};
            
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
            TensorShape shape_3d = {2, 3, 4};
            float d1[] = {0.2865f, 0.5908f, 0.0305f, 0.0373f, 0.8226f, 0.3602f, 0.1271f, 0.5222f, 0.7700f, 0.2158f, 0.6229f, 0.0853f, 0.0517f, 0.5314f, 0.5406f, 0.6374f, 0.7261f, 0.9759f, 0.5163f, 0.3230f, 0.7952f, 0.2708f, 0.4390f, 0.0785f};
            float d2[] = {0.0254f, 0.9626f, 0.8360f, 0.6960f, 0.4090f, 0.1733f, 0.1564f, 0.2502f, 0.5492f, 0.7146f, 0.6602f, 0.2799f, 0.9549f, 0.7379f, 0.5544f, 0.6117f, 0.4196f, 0.2477f, 0.3560f, 0.7578f, 0.0144f, 0.1161f, 0.0460f, 0.0407f};
            float exp_d[] = {0.2611f, -0.3718f, -0.8055f, -0.6587f,  0.4136f,  0.1869f, -0.0293f,  0.2720f,
                0.2208f, -0.4988f, -0.0373f, -0.1946f, -0.9032f, -0.2065f, -0.0138f,  0.0257f,
                0.3065f,  0.7282f,  0.1603f, -0.4348f,  0.7808f,  0.1547f,  0.3930f,  0.0378f};

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor subtraction (same shape)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[] = {0.8555f, 0.7037f, 0.4742f, 0.0978f, 0.4916f, 0.4735f, 0.1732f, 0.4339f, 0.3985f, 0.6159f, 0.6351f, 0.0453f, 0.3746f, 0.6259f, 0.5031f, 0.8565f, 0.6587f, 0.1629f, 0.0706f, 0.6424f, 0.0265f, 0.5858f, 0.9402f, 0.5755f, 0.3882f, 0.6433f, 0.4583f, 0.5456f, 0.9415f, 0.3861f, 0.9612f, 0.9054f, 0.1958f, 0.0694f, 0.1008f, 0.0182f, 0.0944f, 0.6830f, 0.0712f, 0.3190f, 0.8449f, 0.0233f, 0.8145f, 0.2819f, 0.1182f, 0.6967f, 0.6289f, 0.8775f, 0.7351f, 0.8035f, 0.2820f, 0.1774f, 0.7506f, 0.8068f, 0.9905f, 0.4126f, 0.3720f, 0.7764f, 0.3408f, 0.9308f, 0.8584f, 0.4290f, 0.7509f, 0.7545f, 0.1031f, 0.9026f, 0.5053f, 0.8265f, 0.3200f, 0.8955f, 0.3892f, 0.0108f, 0.9054f, 0.0913f, 0.3193f, 0.9501f, 0.9506f, 0.5734f, 0.6318f, 0.4484f, 0.2932f, 0.3287f, 0.6725f, 0.7524f, 0.7916f, 0.7896f, 0.0912f, 0.4944f, 0.0576f, 0.5495f, 0.4415f, 0.8877f, 0.3509f, 0.1171f, 0.1430f, 0.7615f, 0.6182f, 0.1011f, 0.0841f, 0.7010f, 0.0728f, 0.8219f, 0.7062f, 0.0813f, 0.0848f, 0.9866f, 0.3743f, 0.3706f, 0.8128f, 0.9472f, 0.9860f, 0.7534f, 0.3763f, 0.0835f, 0.7771f, 0.5584f, 0.4242f, 0.9064f, 0.1112f, 0.4926f};
            float d2[] = {0.0114f, 0.4687f, 0.0563f, 0.1188f, 0.1175f, 0.6492f, 0.7460f, 0.5834f, 0.9622f, 0.3749f, 0.2857f, 0.8686f, 0.2236f, 0.9632f, 0.0122f, 0.9699f, 0.0432f, 0.8911f, 0.5277f, 0.9930f, 0.0738f, 0.5539f, 0.9693f, 0.5231f, 0.6294f, 0.6957f, 0.4545f, 0.6276f, 0.5843f, 0.9012f, 0.0454f, 0.2810f, 0.9504f, 0.8903f, 0.4557f, 0.6201f, 0.2774f, 0.1881f, 0.4637f, 0.3534f, 0.5837f, 0.0777f, 0.9744f, 0.9862f, 0.6982f, 0.5361f, 0.3095f, 0.8138f, 0.6847f, 0.1626f, 0.9109f, 0.8225f, 0.9498f, 0.7257f, 0.6134f, 0.4182f, 0.9327f, 0.8661f, 0.0452f, 0.0264f, 0.3765f, 0.8106f, 0.9873f, 0.1504f, 0.5941f, 0.3809f, 0.9699f, 0.8421f, 0.8383f, 0.4687f, 0.4148f, 0.2734f, 0.0564f, 0.8647f, 0.8129f, 0.9997f, 0.9966f, 0.5554f, 0.7690f, 0.9448f, 0.8496f, 0.2473f, 0.4505f, 0.1292f, 0.9541f, 0.6062f, 0.2286f, 0.6717f, 0.6181f, 0.3582f, 0.1136f, 0.6716f, 0.5203f, 0.7723f, 0.5202f, 0.8522f, 0.5519f, 0.5609f, 0.8767f, 0.4035f, 0.1340f, 0.0288f, 0.7551f, 0.6203f, 0.7041f, 0.2130f, 0.1364f, 0.0145f, 0.3506f, 0.5899f, 0.3922f, 0.4375f, 0.9042f, 0.3483f, 0.5140f, 0.7837f, 0.3965f, 0.6221f, 0.8624f, 0.9495f};
            float exp_d[] = {0.8441f,  0.2350f,  0.4179f, -0.0210f,  0.3741f, -0.1757f, -0.5728f, -0.1495f,
                -0.5637f,  0.2410f,  0.3494f, -0.8233f,  0.1510f, -0.3373f,  0.4909f, -0.1134f,
                 0.6155f, -0.7282f, -0.4571f, -0.3506f, -0.0473f,  0.0319f, -0.0291f,  0.0524f,
                -0.2412f, -0.0524f,  0.0038f, -0.0820f,  0.3572f, -0.5151f,  0.9158f,  0.6244f,
                -0.7546f, -0.8209f, -0.3549f, -0.6019f, -0.1830f,  0.4949f, -0.3925f, -0.0344f,
                 0.2612f, -0.0544f, -0.1599f, -0.7043f, -0.5800f,  0.1606f,  0.3194f,  0.0637f,
                 0.0504f,  0.6409f, -0.6289f, -0.6451f, -0.1992f,  0.0811f,  0.3771f, -0.0056f,
                -0.5607f, -0.0897f,  0.2956f,  0.9044f,  0.4819f, -0.3816f, -0.2364f,  0.6041f,
                -0.4910f,  0.5217f, -0.4646f, -0.0156f, -0.5183f,  0.4268f, -0.0256f, -0.2626f,
                 0.8490f, -0.7734f, -0.4936f, -0.0496f, -0.0460f,  0.0180f, -0.1372f, -0.4964f,
                -0.5564f,  0.0814f,  0.2220f,  0.6232f, -0.1625f,  0.1834f, -0.1374f, -0.1773f,
                -0.5605f,  0.1913f,  0.3279f,  0.2161f, -0.1694f, -0.6552f, -0.3772f, -0.0907f,
                 0.0663f, -0.4598f, -0.7926f,  0.2975f, -0.0612f,  0.7931f, -0.0489f, -0.5390f,
                -0.6193f,  0.7736f,  0.2379f,  0.3561f,  0.4622f,  0.3573f,  0.5938f,  0.3159f,
                -0.5279f, -0.2648f,  0.2631f, -0.2253f,  0.0277f,  0.2843f, -0.7512f, -0.4569f};

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor t2 = create_test_tensor(shape_4d, d2, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
