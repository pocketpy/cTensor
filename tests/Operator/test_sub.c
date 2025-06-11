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
        TensorShape s_shape = {1, 0, 0, 0};
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
        TensorShape v_shape = {3, 0, 0, 0};
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
        TensorShape m_shape = {2, 2, 0, 0};
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
        TensorShape mat_shape = {2, 2, 0, 0}; float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; float scalar_data[] = {1.0f};
        
        TensorShape expected_shape = {2, 2, 0, 0}; float exp_data[] = {0.0f, 1.0f, 2.0f, 3.0f};

        Tensor t_mat = create_test_tensor(mat_shape, mat_data, false);
        Tensor t_scalar_original = create_test_tensor(scalar_shape, scalar_data, false);
        
        Tensor actual_res = Tensor_sub(t_mat, t_scalar_original); 
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // TODO: Problem in Broadcasting
    
    // // Test Case 5: Advanced Broadcasting
    // {
    //     const char* tc_name = "sub_advanced_broadcasting";
        
    //     // Sub-test 1: Multi-dimensional broadcasting {3,1} - {1,4} -> {3,4}
    //     {
    //         TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {10.0f, 20.0f, 30.0f};
    //         TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //         TensorShape exp_shape = {3, 4, 0, 0}; 
    //         float exp_d[] = {9.0f, 8.0f, 7.0f, 6.0f,    // 10-[1,2,3,4]
    //                          19.0f, 18.0f, 17.0f, 16.0f, // 20-[1,2,3,4]
    //                          29.0f, 28.0f, 27.0f, 26.0f}; // 30-[1,2,3,4]

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_sub(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 3D broadcasting {2,3,1} - {1,1,4} -> {2,3,4}
    //     {
    //         TensorShape s1_shape = {2, 3, 1, 0}; 
    //         float d1[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    //         TensorShape s2_shape = {1, 1, 4, 0}; 
    //         float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //         TensorShape exp_shape = {2, 3, 4, 0};
    //         float exp_d[] = {
    //             // First 2x3 slice
    //             9.0f, 8.0f, 7.0f, 6.0f,    // 10-[1,2,3,4]
    //             19.0f, 18.0f, 17.0f, 16.0f, // 20-[1,2,3,4]
    //             29.0f, 28.0f, 27.0f, 26.0f, // 30-[1,2,3,4]
    //             // Second 2x3 slice
    //             39.0f, 38.0f, 37.0f, 36.0f, // 40-[1,2,3,4]
    //             49.0f, 48.0f, 47.0f, 46.0f, // 50-[1,2,3,4]
    //             59.0f, 58.0f, 57.0f, 56.0f  // 60-[1,2,3,4]
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_sub(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }
    // }

    // Test Case 6: Order Dependency
    {
        const char* tc_name = "sub_order_dependency";
        
        // Sub-test 1: a - b ≠ b - a verification
        {
            TensorShape v_shape = {2, 0, 0, 0};
            
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
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24], d2[24], exp_d[24];
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 10);
                d2[i] = (float)(i + 1);
                exp_d[i] = d1[i] - d2[i]; // Should be 9.0 for all elements
            }

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_sub(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
