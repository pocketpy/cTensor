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
    
    // TODO: Problem in Broadcasting

    // // Test Case 5: Advanced Broadcasting
    // {
    //     const char* tc_name = "mul_advanced_broadcasting";
        
    //     // Sub-test 1: Multi-dimensional broadcasting {3,1} * {1,4} -> {3,4}
    //     {
    //         TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {2.0f, 3.0f, 4.0f};
    //         TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    //         TensorShape exp_shape = {3, 4, 0, 0}; 
    //         float exp_d[] = {2.0f, 4.0f, 6.0f, 8.0f,    // 2*[1,2,3,4]
    //                          3.0f, 6.0f, 9.0f, 12.0f,   // 3*[1,2,3,4]
    //                          4.0f, 8.0f, 12.0f, 16.0f}; // 4*[1,2,3,4]

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_mul(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 4D broadcasting {1,2,3,4} * {5,1,1,1} -> {5,2,3,4}
    //     {
    //         TensorShape s1_shape = {1, 2, 3, 4}; 
    //         float d1[24];
    //         for(int i = 0; i < 24; i++) d1[i] = (float)(i + 1);
            
    //         TensorShape s2_shape = {5, 1, 1, 1}; 
    //         float d2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            
    //         TensorShape exp_shape = {5, 2, 3, 4};
    //         float exp_d[120]; // 5*2*3*4 = 120
    //         for(int i = 0; i < 5; i++) {
    //             for(int j = 0; j < 24; j++) {
    //                 exp_d[i * 24 + j] = d1[j] * d2[i];
    //             }
    //         }

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_mul(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }
    // }

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
            float d1[24], d2[24], exp_d[24];
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 1);
                d2[i] = 2.0f;
                exp_d[i] = d1[i] * d2[i];
            }

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_mul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
