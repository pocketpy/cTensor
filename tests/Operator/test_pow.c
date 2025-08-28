#include "../../include/cten.h"
#include "../test_utils.h"
#include "../test_config.h"
#include <stdio.h>
#include <math.h>

void test_pow_operator() {
    const char* op_name = "pow";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar power (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1};
        const char* tc_name = "pow_scalar_basic";

        // Sub-test 1: Basic power
        {
            float d1[] = {3.557707f};
            float d2[] = {2.050022f};
            float exp_d[] = {13.486858f};  // 3.557707^2.050022 = 13.486858
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_pow(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Square root
        {
            float d1[] = {7.300352f};
            float d2[] = {0.500000f};
            float exp_d[] = {2.701916f};  // 7.300352^0.5 = 2.701916
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_pow(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Power of 0
        {
            float d1[] = {3.008897f};
            float d2[] = {0.000000f};
            float exp_d[] = {1.000000f};  // 3.008897^0 = 1.000000
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_pow(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector power operations
    {
        const char* tc_name = "pow_vector_elements";
        TensorShape v_shape = {3};
        float d1[] = {2.498160f, 4.802857f, 3.927976f};
        float d2[] = {2.000000f, 2.000000f, 2.000000f};
        float exp_d[] = {6.240806f,
                         23.067438f,
                         15.428994f};  // [2.498160^2, 4.802857^2, 3.927976^2]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix power operations
    {
        const char* tc_name = "pow_matrix_2x2";
        TensorShape m_shape = {2, 2};
        float d1[] = {3.394634f, 1.624075f, 1.623978f, 1.232334f};
        float d2[] = {2.665440f, 2.002788f, 2.270181f, 0.551461f};
        float exp_d[] = {25.989442f,
                         2.641186f,
                         3.006458f,
                         1.122104f};  // [3.394634^2.665440, 1.624075^2.002788,
                                      // 1.623978^2.270181, 1.232334^0.551461]
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor power operations
    {
        const char* tc_name = "pow_3d_tensor";
        TensorShape t_shape = {2, 2, 2};
        float d1[] = {4.879639f,
                      4.329771f,
                      1.849356f,
                      1.727300f,
                      1.733618f,
                      2.216969f,
                      3.099026f,
                      2.727780f};
        float d2[] = {2.000000f,
                      2.000000f,
                      2.000000f,
                      2.000000f,
                      2.000000f,
                      2.000000f,
                      2.000000f,
                      2.000000f};
        float exp_d[] = {
            23.810881f,
            18.746913f,
            3.420119f,
            2.983565f,
            3.005432f,
            4.914951f,
            9.603960f,
            7.440784f};  // [4.879639^2, 4.329771^2, 1.849356^2, 1.727300^2, 1.733618^2,
                         // 2.216969^2, 3.099026^2, 2.727780^2]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor t2 = create_test_tensor(t_shape, d2, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Broadcasting (scalar power applied to vector)
    {
        const char* tc_name = "pow_broadcast_vector_scalar";
        TensorShape vec_shape = {3};
        float vec_data[] = {2.164917f, 3.447412f, 1.557975f};
        TensorShape scalar_shape = {1};
        float scalar_data[] = {2.000000f};  // power of 2

        // Expected: broadcast scalar {2} to the vector then apply power
        TensorShape expected_shape = {3};
        float exp_data[] = {4.686864f,
                            11.884647f,
                            2.427287f};  // [2.164917^2, 3.447412^2, 1.557975^2]

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, false);

        Tensor actual_res = Tensor_pow(t_vec, t_scalar);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Negative base
    {
        const char* tc_name = "pow_negative_base";
        TensorShape s_shape = {1};
        float d1[] = {-2.000000f};
        float d2[] = {3.000000f};
        float exp_d[] = {-8.000000f};  // (-2)^3 = -8
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 7: Negative exponent
    {
        const char* tc_name = "pow_negative_exponent";
        TensorShape s_shape = {1};
        float d1[] = {4.996321f};
        float d2[] = {-2.876786f};
        float exp_d[] = {0.009775f};  // 4.996321^(-2.876786) = 0.009775
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 8: Broadcasting (vector base, scalar exponent of negative value)
    {
        const char* tc_name = "pow_broadcast_negative_exponent";
        TensorShape vec_shape = {4};
        float vec_data[] = {11.515921f, 9.782560f, 4.028242f, 4.027929f};
        TensorShape scalar_shape = {1};
        float scalar_data[] = {-0.587125f};

        // Expected: broadcast scalar {-1} to {4} then apply power
        TensorShape expected_shape = {4};
        float exp_data[] = {0.238169f, 0.262108f, 0.441287f, 0.441307f};

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, false);

        Tensor actual_res = Tensor_pow(t_vec, t_scalar);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 9: Broadcasting with different dimensional tensors
    {
        const char* tc_name = "pow_broadcast_matrix_vector";
        TensorShape matrix_shape = {2, 3};  // 2x3 matrix
        float matrix_data[] = {2.1854f, 4.7782f, 3.7940f, 3.1940f, 1.2021f, 1.2020f};
        TensorShape vector_shape = {3};  // vector with 3 elements
        float vector_data[] = {0.6162f, 2.2324f, 1.7022f};

        // Expected: broadcast vector to shape [2,3] then apply power
        TensorShape expected_shape = {2, 3};
        float exp_data[] = {1.618896f, 32.838981f, 9.676972f, 2.045367f, 1.508202f, 1.367771f};
        Tensor t_matrix = create_test_tensor(matrix_shape, matrix_data, false);
        Tensor t_vector = create_test_tensor(vector_shape, vector_data, false);

        Tensor actual_res = Tensor_pow(t_matrix, t_vector);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 10: Power with very large exponent
    {
        const char* tc_name = "pow_large_exponent";
        TensorShape s_shape = {1};
        float d1[] = {2.799264f};
        float d2[] = {9.207805f};
        float exp_d[] = {13070.524894f};  // 2.799264^9.207805 = 13070.524894
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res,
                        &expected_res,
                        op_name,
                        tc_name,
                        1,
                        0.1f);  // Large tolerance for large values
    }

    // Test Case 11: Fractional exponent
    {
        const char* tc_name = "pow_fractional_exponent";
        TensorShape s_shape = {1};
        float d1[] = {70.149528f};
        float d2[] = {0.333333f};
        float exp_d[] = {4.124218f};  // 70.149528^(0.333333) = 4.124218
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 12: 4D tensor power operations
    {
        const char* tc_name = "pow_4d_tensor";
        TensorShape t_shape = {2, 2, 1, 2};  // 2x2x1x2 tensor
        float d1[] = {1.584617f,
                      2.582998f,
                      2.907829f,
                      1.501168f,
                      2.988317f,
                      2.426222f,
                      2.417480f,
                      1.510599f};
        float d2[] = {1.092250f,
                      3.099099f,
                      2.599444f,
                      1.186663f,
                      4.895022f,
                      1.931085f,
                      1.362426f,
                      3.473544f};
        float exp_d[] = {1.653360f,
                         18.932729f,
                         16.033239f,
                         1.619430f,
                         212.434116f,
                         5.537756f,
                         3.328920f,
                         4.190666f};  // Element-wise power operations
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor t2 = create_test_tensor(t_shape, d2, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res,
                        &expected_res,
                        op_name,
                        tc_name,
                        1,
                        0.1f);  // Large tolerance for large values
    }

    // Test Case 13: Power with zero base
    {
        const char* tc_name = "pow_zero_base";
        TensorShape s_shape = {1};
        float d1[] = {0.000000f};
        float d2[] = {5.059696f};
        float exp_d[] = {0.000000f};  // 0.000000^5.059696 = 0.000000
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_pow(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
