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
    
    // Test Case 5: Advanced Broadcasting
    {
        const char* tc_name = "mul_advanced_broadcasting";
        
        // Sub-test 1: Multi-dimensional broadcasting {3,1} * {1,4} -> {3,4}
        {
            TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {2.0f, 3.0f, 4.0f};
            TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
            TensorShape exp_shape = {3, 4, 0, 0}; 
            float exp_d[] = {2.0f, 4.0f, 6.0f, 8.0f,    // 2*[1,2,3,4]
                             3.0f, 6.0f, 9.0f, 12.0f,   // 3*[1,2,3,4]
                             4.0f, 8.0f, 12.0f, 16.0f}; // 4*[1,2,3,4]

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D broadcasting {1,2,3,4} * {5,1,1,1} -> {5,2,3,4}
        {
            TensorShape s1_shape = {1, 2, 3, 4}; 
            float d1[] = {
                0.1254f, 0.7612f, 0.3476f, 0.8791f, 
                0.2415f, 0.5832f, 0.6903f, 0.9234f, 
                0.1327f, 0.4651f, 0.7561f, 0.5872f, 
                0.9135f, 0.2783f, 0.3491f, 0.7392f, 
                0.5517f, 0.8253f, 0.6023f, 0.1937f, 
                0.4936f, 0.2341f, 0.8745f, 0.5291f
            };
            
            TensorShape s2_shape = {5, 1, 1, 1}; 
            float d2[] = {0.8365f, 0.2471f, 0.9382f, 0.5713f, 0.1648f};
            
            TensorShape exp_shape = {5, 2, 3, 4};
            
            float exp_d[] = {
                // Batch 0
                0.1049f, 0.6367f, 0.2908f, 0.7354f,
                0.2020f, 0.4878f, 0.5774f, 0.7724f,
                0.1110f, 0.3891f, 0.6325f, 0.4912f,
                0.7641f, 0.2328f, 0.2920f, 0.6183f,
                0.4615f, 0.6904f, 0.5038f, 0.1620f,
                0.4129f, 0.1958f, 0.7315f, 0.4426f,
                // Batch 1
                0.0310f, 0.1881f, 0.0859f, 0.2172f,
                0.0597f, 0.1441f, 0.1706f, 0.2282f,
                0.0328f, 0.1149f, 0.1868f, 0.1451f,
                0.2257f, 0.0688f, 0.0863f, 0.1827f,
                0.1363f, 0.2039f, 0.1488f, 0.0479f,
                0.1220f, 0.0578f, 0.2161f, 0.1307f,
                // Batch 2
                0.1177f, 0.7142f, 0.3261f, 0.8248f,
                0.2266f, 0.5472f, 0.6476f, 0.8663f,
                0.1245f, 0.4364f, 0.7094f, 0.5509f,
                0.8570f, 0.2611f, 0.3275f, 0.6935f,
                0.5176f, 0.7743f, 0.5651f, 0.1817f,
                0.4631f, 0.2196f, 0.8205f, 0.4964f,
                // Batch 3
                0.0716f, 0.4349f, 0.1986f, 0.5022f,
                0.1380f, 0.3332f, 0.3944f, 0.5275f,
                0.0758f, 0.2657f, 0.4320f, 0.3355f,
                0.5219f, 0.1590f, 0.1994f, 0.4223f,
                0.3152f, 0.4715f, 0.3441f, 0.1107f,
                0.2820f, 0.1337f, 0.4996f, 0.3023f,
                // Batch 4
                0.0207f, 0.1254f, 0.0573f, 0.1449f,
                0.0398f, 0.0961f, 0.1138f, 0.1522f,
                0.0219f, 0.0766f, 0.1246f, 0.0968f,
                0.1505f, 0.0459f, 0.0575f, 0.1218f,
                0.0909f, 0.1360f, 0.0993f, 0.0319f,
                0.0813f, 0.0386f, 0.1441f, 0.0872f
            };

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

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
            float d1[] = {0.1471f, 0.9266f, 0.4921f, 0.2582f, 0.4591f, 0.9800f, 0.4926f, 0.3288f, 0.6334f, 0.2401f, 0.0759f, 0.1289f, 0.1280f, 0.1519f, 0.1388f, 0.6409f, 0.1819f, 0.3457f, 0.8968f, 0.4740f, 0.6676f, 0.1723f, 0.1923f, 0.0409f};
            float d2[] = {0.1689f, 0.2786f, 0.1770f, 0.0887f, 0.1206f, 0.4608f, 0.2063f, 0.3643f, 0.5034f, 0.6904f, 0.0393f, 0.7994f, 0.6279f, 0.0818f, 0.8736f, 0.9209f, 0.0611f, 0.2769f, 0.8062f, 0.7483f, 0.1845f, 0.2093f, 0.3705f, 0.4845f};
            float exp_d[] = {0.0248f, 0.2582f, 0.0871f, 0.0229f, 0.0554f, 0.4516f, 0.1016f, 0.1198f, 0.3189f,
                0.1658f, 0.0030f, 0.1030f, 0.0804f, 0.0124f, 0.1213f, 0.5902f, 0.0111f, 0.0957f,
                0.7230f, 0.3547f, 0.1232f, 0.0361f, 0.0712f, 0.0198f};

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor multiplication (same shape)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[] = {0.6183f, 0.3689f, 0.4625f, 0.7475f, 0.0367f, 0.2524f, 0.7133f, 0.8952f, 0.5117f, 0.5321f, 0.1072f, 0.4474f, 0.5326f, 0.2425f, 0.2692f, 0.3773f, 0.0201f, 0.3221f, 0.2114f, 0.3275f, 0.1198f, 0.8905f, 0.5936f, 0.6791f, 0.7892f, 0.4984f, 0.0869f, 0.5371f, 0.5868f, 0.7454f, 0.4317f, 0.1276f, 0.2838f, 0.3631f, 0.6459f, 0.5708f, 0.3561f, 0.9865f, 0.6058f, 0.2372f, 0.1018f, 0.1529f, 0.2460f, 0.1607f, 0.1866f, 0.2851f, 0.1734f, 0.8968f, 0.0802f, 0.5245f, 0.4104f, 0.9824f, 0.1120f, 0.3979f, 0.9695f, 0.8655f, 0.8171f, 0.2579f, 0.1709f, 0.6686f, 0.9294f, 0.5568f, 0.5716f, 0.2800f, 0.7695f, 0.1870f, 0.3237f, 0.4254f, 0.5076f, 0.2424f, 0.1148f, 0.6106f, 0.2886f, 0.5812f, 0.1544f, 0.4811f, 0.5326f, 0.0518f, 0.3366f, 0.1344f, 0.0634f, 0.9900f, 0.3224f, 0.8099f, 0.2546f, 0.6815f, 0.7602f, 0.5956f, 0.4716f, 0.4118f, 0.3489f, 0.9295f, 0.8306f, 0.9650f, 0.1243f, 0.7309f, 0.9383f, 0.1812f, 0.0665f, 0.7411f, 0.5745f, 0.8418f, 0.1398f, 0.7953f, 0.2016f, 0.1637f, 0.1643f, 0.8146f, 0.6652f, 0.5231f, 0.3588f, 0.8772f, 0.3924f, 0.8166f, 0.4391f, 0.3769f, 0.4627f, 0.3014f, 0.7476f, 0.5027f};
            float d2[] = {0.2322f, 0.8996f, 0.3839f, 0.5436f, 0.9065f, 0.6242f, 0.1169f, 0.9398f, 0.6277f, 0.3349f, 0.1393f, 0.7940f, 0.6201f, 0.5335f, 0.8939f, 0.7886f, 0.1517f, 0.3117f, 0.2485f, 0.7439f, 0.0335f, 0.5699f, 0.7625f, 0.8768f, 0.3421f, 0.8213f, 0.1106f, 0.8465f, 0.1275f, 0.3973f, 0.7973f, 0.1499f, 0.2293f, 0.7223f, 0.7200f, 0.6411f, 0.6939f, 0.5427f, 0.2518f, 0.3457f, 0.1816f, 0.9085f, 0.5834f, 0.4009f, 0.4620f, 0.9473f, 0.1534f, 0.5862f, 0.5059f, 0.6115f, 0.0181f, 0.8721f, 0.9321f, 0.5651f, 0.6967f, 0.9225f, 0.7072f, 0.1525f, 0.5763f, 0.6067f, 0.4241f, 0.7364f, 0.9344f, 0.9256f, 0.4508f, 0.1132f, 0.9848f, 0.8389f, 0.1247f, 0.9208f, 0.8699f, 0.5188f, 0.5913f, 0.3990f, 0.0548f, 0.3352f, 0.8029f, 0.0046f, 0.3335f, 0.3982f, 0.5374f, 0.9199f, 0.3463f, 0.3470f, 0.7375f, 0.4522f, 0.2246f, 0.4524f, 0.1409f, 0.1764f, 0.4984f, 0.4189f, 0.9148f, 0.3624f, 0.5806f, 0.6323f, 0.0131f, 0.6635f, 0.1780f, 0.9611f, 0.1487f, 0.4146f, 0.0853f, 0.9969f, 0.5022f, 0.5954f, 0.0671f, 0.7500f, 0.2099f, 0.8981f, 0.2051f, 0.1907f, 0.0365f, 0.4721f, 0.5648f, 0.0657f, 0.7755f, 0.4533f, 0.5244f, 0.4408f};
            float exp_d[] = {0.1435693f, 0.3318624f, 0.1775537f, 0.4063410f, 0.03326855f,
                0.1575481f, 0.08338477f, 0.8413090f, 0.3211941f, 0.1782003f,
                0.01493296f, 0.3552356f, 0.3302653f, 0.1293738f, 0.2406379f,
                0.2975388f, 0.003049170f, 0.1003986f, 0.05253290f, 0.2436272f,
                0.004013300f, 0.5074959f, 0.4526200f, 0.5954348f, 0.2699853f,
                0.4093359f, 0.009611141f, 0.4546551f, 0.07481699f, 0.2961474f,
                0.3441944f, 0.01912724f, 0.06507535f, 0.2622671f, 0.4650480f,
                0.3659399f, 0.2470978f, 0.5353736f, 0.1525404f, 0.08200004f,
                0.01848688f, 0.1389097f, 0.1435164f, 0.06442463f, 0.08620920f,
                0.2700753f, 0.02659956f, 0.5257041f, 0.04057318f, 0.3207318f,
                0.007428240f, 0.8567510f, 0.1043952f, 0.2248533f, 0.6754506f,
                0.7984238f, 0.5778531f, 0.03932975f, 0.09848967f, 0.4056396f,
                0.3941586f, 0.4100275f, 0.5341031f, 0.2591680f, 0.3468906f,
                0.02116840f, 0.3187798f, 0.3568681f, 0.06329772f, 0.2232019f,
                0.09986452f, 0.3167793f, 0.1706492f, 0.2318988f, 0.008461121f,
                0.1612647f, 0.4276245f, 0.0002382800f, 0.1122561f, 0.05351808f,
                0.03407116f, 0.9107010f, 0.1116471f, 0.2810353f, 0.1877675f,
                0.3081743f, 0.1707409f, 0.2694494f, 0.06644844f, 0.07264152f,
                0.1738918f, 0.3893676f, 0.7598329f, 0.3497160f, 0.07216858f,
                0.4621481f, 0.01229173f, 0.1202262f, 0.01183700f, 0.7122712f,
                0.08542816f, 0.3490103f, 0.01192494f, 0.7928346f, 0.1012435f,
                0.09746698f, 0.01102453f, 0.6109500f, 0.1396255f, 0.4697962f,
                0.07358988f, 0.1672820f, 0.01432260f, 0.3855169f, 0.2480037f,
                0.02476233f, 0.3588239f, 0.1366246f, 0.3920414f, 0.2215901f};

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor t2 = create_test_tensor(shape_4d, d2, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
