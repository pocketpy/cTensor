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

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    // Test Case 2: Vector element-wise multiplication
    {
        const char* tc_name = "mul_vector_3el";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float d2[] = {4.0f, 5.0f, 0.5f};
        float exp_d[] = {4.0f, 10.0f, 1.5f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_mul(t1, t2);

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
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

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
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

        if (compare_tensors(&actual_res, &expected_res, op_name, tc_name, TEST_FLOAT_TOLERANCE)) {
            csv_reporter_add_entry(op_name, tc_name, true, NULL);
        }
    }

    cten_free(pool_id);
}
