#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_mulf_operator() {
    const char* op_name = "mulf";
    PoolId pool_id = 5;

    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar tensor multiplied by float
    {
        const char* tc_name = "mulf_scalar_x_float";
        TensorShape s_shape = {1};
        float d1[] = {2.0f};
        float scalar_val = 3.0f;
        float exp_d[] = {6.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Vector tensor multiplied by float
    {
        const char* tc_name = "mulf_vector_x_float";
        TensorShape v_shape = {3};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float scalar_val = 0.5f;
        float exp_d[] = {0.5f, 1.0f, 1.5f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix tensor multiplied by float
    {
        const char* tc_name = "mulf_matrix_x_float";
        TensorShape m_shape = {2, 2};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scalar_val = -2.0f;
        float exp_d[] = {-2.0f, -4.0f, -6.0f, -8.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Tensor multiplied by zero
    {
        const char* tc_name = "mulf_matrix_x_zero";
        TensorShape m_shape = {2, 2};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scalar_val = 0.0f;
        float exp_d[] = {0.0f, 0.0f, 0.0f, 0.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_mulf(t1, scalar_val);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Special Scalar Values
    {
        const char* tc_name = "mulf_special_scalar_values";

        // Sub-test 1: Multiplication by zero
        {
            TensorShape m_shape = {2, 3};
            float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float scalar_val = 0.0f;
            float exp_d[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            Tensor t1 = create_test_tensor(m_shape, d1, false);
            Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Very small scalar multiplication
        {
            TensorShape v_shape = {4};
            float d1[] = {1000.0f, 2000.0f, 3000.0f, 4000.0f};
            float scalar_val = 1e-6f;
            float exp_d[] = {1e-3f, 2e-3f, 3e-3f, 4e-3f};

            Tensor t1 = create_test_tensor(v_shape, d1, false);
            Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Higher Dimensional Tensors
    {
        const char* tc_name = "mulf_higher_dimensional_tensors";

        // Sub-test 1: 3D tensor scalar multiplication (2x3x4)
        {
            TensorShape shape_3d = {2, 3, 4};
            float d1[] = {0.6436f, 0.5264f, 0.7316f, 0.0816f, 0.0604f, 0.2471f, 0.1595f, 0.8718f,
                          0.2192f, 0.9759f, 0.3369f, 0.1821f, 0.7897f, 0.6587f, 0.4982f, 0.5554f,
                          0.7192f, 0.2285f, 0.9963f, 0.9748f, 0.6503f, 0.1995f, 0.6802f, 0.0722f};
            float scalar = 2.5f;
            float exp_d[] = {1.609000f, 1.316000f, 1.829000f, 0.204000f, 0.151000f, 0.617750f,
                             0.398750f, 2.179500f, 0.548000f, 2.439750f, 0.842250f, 0.455250f,
                             1.974250f, 1.646750f, 1.245500f, 1.388500f, 1.798000f, 0.571250f,
                             2.490750f, 2.437000f, 1.625750f, 0.498750f, 1.700500f, 0.180500f};

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor scalar multiplication (2x3x4x5)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[] = {0.0307f, 0.2577f, 0.4626f, 0.8683f, 0.7272f, 0.7427f, 0.4255f, 0.3459f,
                          0.3710f, 0.9876f, 0.0401f, 0.8670f, 0.5787f, 0.4386f, 0.7253f, 0.4867f,
                          0.8734f, 0.9007f, 0.4217f, 0.2768f, 0.5924f, 0.9124f, 0.2107f, 0.6230f,
                          0.6316f, 0.7331f, 0.1316f, 0.7158f, 0.9090f, 0.1797f, 0.2375f, 0.9714f,
                          0.1810f, 0.8544f, 0.4923f, 0.2472f, 0.8707f, 0.4453f, 0.5148f, 0.3592f,
                          0.5930f, 0.1635f, 0.3911f, 0.9694f, 0.2581f, 0.6567f, 0.3252f, 0.7735f,
                          0.1309f, 0.9698f, 0.4538f, 0.2361f, 0.0735f, 0.1698f, 0.5198f, 0.3370f,
                          0.8289f, 0.4309f, 0.2487f, 0.6171f, 0.7068f, 0.1670f, 0.1676f, 0.0367f,
                          0.7364f, 0.6638f, 0.4746f, 0.8442f, 0.8057f, 0.5854f, 0.8683f, 0.2058f,
                          0.1119f, 0.2697f, 0.0571f, 0.5312f, 0.9366f, 0.0393f, 0.1221f, 0.4522f,
                          0.9339f, 0.3162f, 0.5072f, 0.0416f, 0.1483f, 0.9866f, 0.9651f, 0.0049f,
                          0.9518f, 0.6391f, 0.8679f, 0.4547f, 0.5156f, 0.4888f, 0.6669f, 0.1397f,
                          0.0300f, 0.3079f, 0.7047f, 0.2019f, 0.6734f, 0.9699f, 0.0939f, 0.6726f,
                          0.4438f, 0.8681f, 0.1771f, 0.6926f, 0.8381f, 0.9446f, 0.6832f, 0.4972f,
                          0.6178f, 0.8689f, 0.5706f, 0.0304f, 0.9309f, 0.6895f, 0.6765f, 0.2157f};
            float scalar_val = 1.5f;
            float exp_d[] = {
                0.0461f, 0.3866f, 0.6939f, 1.3025f, 1.0908f, 1.1140f, 0.6382f, 0.5188f, 0.5565f,
                1.4814f, 0.0602f, 1.3005f, 0.8680f, 0.6579f, 1.0879f, 0.7300f, 1.3101f, 1.3510f,
                0.6326f, 0.4152f, 0.8886f, 1.3686f, 0.3160f, 0.9345f, 0.9474f, 1.0997f, 0.1974f,
                1.0737f, 1.3635f, 0.2695f, 0.3562f, 1.4571f, 0.2715f, 1.2816f, 0.7384f, 0.3708f,
                1.3061f, 0.6680f, 0.7722f, 0.5388f, 0.8895f, 0.2452f, 0.5867f, 1.4541f, 0.3871f,
                0.9851f, 0.4878f, 1.1603f, 0.1963f, 1.4547f, 0.6807f, 0.3541f, 0.1102f, 0.2547f,
                0.7797f, 0.5055f, 1.2434f, 0.6464f, 0.3730f, 0.9257f, 1.0602f, 0.2505f, 0.2514f,
                0.0551f, 1.1046f, 0.9957f, 0.7119f, 1.2663f, 1.2085f, 0.8781f, 1.3025f, 0.3087f,
                0.1679f, 0.4045f, 0.0857f, 0.7968f, 1.4049f, 0.0589f, 0.1832f, 0.6783f, 1.4009f,
                0.4743f, 0.7608f, 0.0624f, 0.2225f, 1.4799f, 1.4476f, 0.0074f, 1.4277f, 0.9586f,
                1.3019f, 0.6820f, 0.7734f, 0.7332f, 1.0003f, 0.2095f, 0.0450f, 0.4619f, 1.0570f,
                0.3029f, 1.0101f, 1.4548f, 0.1409f, 1.0089f, 0.6657f, 1.3022f, 0.2657f, 1.0389f,
                1.2572f, 1.4169f, 1.0248f, 0.7458f, 0.9267f, 1.3033f, 0.8559f, 0.0456f, 1.3963f,
                1.0343f, 1.0148f, 0.3235f};

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_mulf(t1, scalar_val);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
