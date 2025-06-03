#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>
#include <string.h>

void test_matmul_operator() {
    const char* op_name = "matmul";
    PoolId pool_id = 4;

    cten_begin_malloc(pool_id);

    // Test Case 1: Square Matrix Multiplication (2x2 * 2x2)
    {
        const char* tc_name = "matmul_square_2x2";
        TensorShape s1_shape = {2, 2, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorShape s2_shape = {2, 2, 0, 0}; float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
        TensorShape exp_shape = {2, 2, 0, 0}; float exp_d[] = {19.0f, 22.0f, 43.0f, 50.0f};
        
        Tensor t1 = create_test_tensor(s1_shape, d1, false);
        Tensor t2 = create_test_tensor(s2_shape, d2, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_matmul(t1, t2);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 2: Rectangular Matrix Multiplication (2x3 * 3x2)
    {
        const char* tc_name = "matmul_rect_2x3_3x2";
        TensorShape s1_shape = {2, 3, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        TensorShape s2_shape = {3, 2, 0, 0}; float d2[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        TensorShape exp_shape = {2, 2, 0, 0}; float exp_d[] = {58.0f, 64.0f, 139.0f, 154.0f};

        Tensor t1 = create_test_tensor(s1_shape, d1, false);
        Tensor t2 = create_test_tensor(s2_shape, d2, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_matmul(t1, t2);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 3: Matrix-Vector (2x2 * 2x1) (Vector as column matrix)
    {
        const char* tc_name = "matmul_matrix_vector_2x2_2x1";
        TensorShape s_mat_shape = {2, 2, 0, 0}; float d_mat[] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorShape s_vec_shape = {2, 1, 0, 0}; float d_vec[] = {5.0f, 6.0f}; // Column vector
        TensorShape exp_shape = {2, 1, 0, 0}; float exp_d[] = {17.0f, 39.0f};

        Tensor t_mat = create_test_tensor(s_mat_shape, d_mat, false);
        Tensor t_vec = create_test_tensor(s_vec_shape, d_vec, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_matmul(t_mat, t_vec);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 4: Vector-Matrix (1x2 * 2x2) (Vector as row matrix)
    {
        const char* tc_name = "matmul_vector_matrix_1x2_2x2";
        TensorShape s_vec_shape = {1, 2, 0, 0}; float d_vec[] = {1.0f, 2.0f}; // Row vector
        TensorShape s_mat_shape = {2, 2, 0, 0}; float d_mat[] = {3.0f, 4.0f, 5.0f, 6.0f};
        TensorShape exp_shape = {1, 2, 0, 0}; float exp_d[] = {13.0f, 16.0f};

        Tensor t_vec = create_test_tensor(s_vec_shape, d_vec, false);
        Tensor t_mat = create_test_tensor(s_mat_shape, d_mat, false);
        Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
        Tensor actual_res = Tensor_matmul(t_vec, t_mat);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    cten_free(pool_id);
}
