#include "../../include/cten.h"
#include "../test_utils.h"
#include "../test_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void test_div_operator() {
    const char* op_name = "div";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar division (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1};
        const char* tc_name = "div_scalar_basic";

        // Sub-test 1: Basic division - I
        {
            float d1[] = {6.754841f};
            float d2[] = {0.612548f};
            float exp_d[] = {11.027441f};  // 6.754841 / 0.612548 = 11.027441
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_div(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Basic division - II
        {
            float d1[] = {7.628241f};
            float d2[] = {3.545148f};
            float exp_d[] = {2.151741f};  // 7.628241 / 3.545148 = 2.151741
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_div(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector division operations
    {
        const char* tc_name = "div_vector";
        TensorShape v_shape = {3};
        float d1[] = {4.370861f, 9.556429f, 7.587945f};
        float d2[] = {3.193963f, 1.202084f, 1.201975f};
        float exp_d[] = {1.368476f,
                         7.949885f,
                         6.312896f};  // [4.370861/3.193963, 9.556429/1.202084, 7.587945/1.201975]
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix division operations
    {
        const char* tc_name = "div_matrix_2x2";
        TensorShape m_shape = {2, 2};
        float d1[] = {1.522753f, 8.795585f, 6.410035f, 7.372653f};
        float d2[] = {0.592630f, 4.864594f, 4.245992f, 1.455526f};
        float exp_d[] = {2.569482f,
                         1.808082f,
                         1.509667f,
                         5.065284f};  // [1.522753/0.592630, 8.795585/4.864594, 6.410035/4.245992,
                                      // 7.372653/1.455526]
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: 3D tensor division operations
    {
        const char* tc_name = "div_3d_tensor";
        TensorShape t_shape = {2, 2, 2};
        float d1[] = {2.636425f,
                      2.650641f,
                      3.738180f,
                      5.722808f,
                      4.887505f,
                      3.621062f,
                      6.506676f,
                      2.255445f};
        float d2[] = {1.814651f,
                      2.148628f,
                      2.552315f,
                      4.033292f,
                      1.398532f,
                      2.814055f,
                      3.165866f,
                      0.709027f};
        float exp_d[] = {
            1.452855f,
            1.233643f,
            1.464623f,
            1.418893f,
            3.494740f,
            1.286777f,
            2.055260f,
            3.181043f};  // [2.636425/1.814651, 2.650641/2.148628, 3.738180/2.552315, 5.722808/4.033292,
                         // 4.887505/1.398532, 3.621062/2.814055, 6.506676/3.165866, 2.255445/0.709027]
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor t2 = create_test_tensor(t_shape, d2, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Broadcasting (vector divided by scalar)
    {
        const char* tc_name = "div_broadcast_vector_scalar";
        TensorShape vec_shape = {3};
        float vec_data[] = {6.467904f, 2.534717f, 1.585464f};
        TensorShape scalar_shape = {1};
        float scalar_data[] = {4.514808f};

        // Expected: broadcast scalar to vector then apply division
        TensorShape expected_shape = {3};
        float exp_data[] = {
            1.432598f,
            0.561423f,
            0.351170f};  // [6.467904/4.514808, 2.534717/4.514808, 1.585464/4.514808]

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, false);

        Tensor actual_res = Tensor_div(t_vec, t_scalar);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 6: Division by zero
    {
        const char* tc_name = "div_by_zero";
        TensorShape s_shape = {1};
        float d1[] = {10.0f};
        float d2[] = {0.0f};
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);

        Tensor actual_res = Tensor_div(t1, t2);
        // Check if the result is very large (greater than 1e10 in absolute value)
        if(fabs(actual_res.data->flex[0]) < 1e10) {
            fprintf(stderr,
                    "Test %s:%d failed: expected a very large number, got %f\n",
                    tc_name,
                    1,
                    actual_res.data->flex[0]);
            abort();
        }
    }

    // Test Case 7: Division with negative numbers
    {
        const char* tc_name = "div_negative_numbers";
        TensorShape s_shape = {1};
        float d1[] = {-21.854305f};
        float d2[] = {9.556429f};
        float exp_d[] = {-2.286869f};  // -21.854305 / 9.556429 = -2.286869
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 8: Division of negative numbers
    {
        const char* tc_name = "div_both_negative";
        TensorShape s_shape = {1};
        float d1[] = {-37.939727f};
        float d2[] = {-6.387926f};
        float exp_d[] = {5.939287f};  // -37.939727 / -6.387926 = 5.939287
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 9: Broadcasting (matrix divided by vector)
    {
        const char* tc_name = "div_broadcast_matrix_vector";
        TensorShape matrix_shape = {2, 3};  // 2x3 matrix
        float matrix_data[] = {2.1854f, 4.7782f, 3.7940f, 3.1940f, 1.2021f, 1.2020f};
        TensorShape vector_shape = {3};  // vector with 3 elements
        float vector_data[] = {0.6162f, 2.2324f, 1.7022f};

        // Expected: broadcast vector to shape [2,3] then divide
        TensorShape expected_shape = {2, 3};
        float exp_data[] = {3.546576f, 2.140387f, 2.228880f, 5.183382f, 0.538479f, 0.706145f};

        Tensor t_matrix = create_test_tensor(matrix_shape, matrix_data, false);
        Tensor t_vector = create_test_tensor(vector_shape, vector_data, false);

        Tensor actual_res = Tensor_div(t_matrix, t_vector);
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 10: Identity division (dividing by itself)
    {
        const char* tc_name = "div_identity";
        TensorShape v_shape = {3};
        float d[] = {7.340280f, -4.027929f, 1.906585f};
        float exp_d[] = {
            1.000000f,
            1.000000f,
            1.000000f};  // [7.340280/7.340280, -4.027929/-4.027929, 1.906585/1.906585] = [1, 1, 1]
        Tensor t = create_test_tensor(v_shape, d, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t, t);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 11: Very large and very small numbers
    {
        const char* tc_name = "div_extreme_values";
        TensorShape s_shape = {1};
        float d1[] = {1.000000e+07f};
        float d2[] = {1.000000e-07f};
        float exp_d[] = {1.000000e+14f};  // 1.000000e+07f / 1.000000e-07f = 1.000000e+14f
        Tensor t1 = create_test_tensor(s_shape, d1, false);
        Tensor t2 = create_test_tensor(s_shape, d2, false);
        Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res,
                        &expected_res,
                        op_name,
                        tc_name,
                        1,
                        1e6f);  // Using larger tolerance due to floating point precision
    }

    // Test Case 12: 4D tensor division
    {
        const char* tc_name = "div_4d_tensor";
        TensorShape t_shape = {2, 2, 2, 1};
        float d1[] = {64.100351f,
                      73.726532f,
                      11.852604f,
                      97.291887f,
                      84.919838f,
                      29.110520f,
                      26.364247f,
                      26.506406f};
        float d2[] = {7.476360f,
                      11.445616f,
                      9.775010f,
                      7.242125f,
                      13.013352f,
                      4.510889f,
                      7.258604f,
                      8.594513f};
        float exp_d[] = {8.573737f,
                         6.441465f,
                         1.212541f,
                         13.434164f,
                         6.525593f,
                         6.453388f,
                         3.632138f,
                         3.084108f};  // Element-wise division
        Tensor t1 = create_test_tensor(t_shape, d1, false);
        Tensor t2 = create_test_tensor(t_shape, d2, false);
        Tensor expected_res = create_test_tensor(t_shape, exp_d, false);
        Tensor actual_res = Tensor_div(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}
