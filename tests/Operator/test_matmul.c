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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
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

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Edge Matrix Sizes
    {
        const char* tc_name = "matmul_edge_matrix_sizes";
        
        // Sub-test 1: 1x1 matrix multiplication
        {
            TensorShape s1_shape = {1, 1, 0, 0}; float d1[] = {5.0f};
            TensorShape s2_shape = {1, 1, 0, 0}; float d2[] = {3.0f};
            TensorShape exp_shape = {1, 1, 0, 0}; float exp_d[] = {15.0f};

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Single row/column matrices {1,5} * {5,1} -> {1,1}
        {
            TensorShape s1_shape = {1, 5, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            TensorShape s2_shape = {5, 1, 0, 0}; float d2[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            TensorShape exp_shape = {1, 1, 0, 0}; float exp_d[] = {70.0f}; // 1*2+2*3+3*4+4*5+5*6 = 70

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Single column/row matrices {5,1} * {1,5} -> {5,5}
        {
            TensorShape s1_shape = {5, 1, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            TensorShape s2_shape = {1, 5, 0, 0}; float d2[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            TensorShape exp_shape = {5, 5, 0, 0}; 
            float exp_d[] = {
                2.0f, 3.0f, 4.0f, 5.0f, 6.0f,    // 1*[2,3,4,5,6]
                4.0f, 6.0f, 8.0f, 10.0f, 12.0f,  // 2*[2,3,4,5,6]
                6.0f, 9.0f, 12.0f, 15.0f, 18.0f, // 3*[2,3,4,5,6]
                8.0f, 12.0f, 16.0f, 20.0f, 24.0f, // 4*[2,3,4,5,6]
                10.0f, 15.0f, 20.0f, 25.0f, 30.0f // 5*[2,3,4,5,6]
            };

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Large Matrix Operations
    {
        const char* tc_name = "matmul_large_matrix_operations";
        
        // Sub-test 1: Stress test with 10x10 matrices
        {
            TensorShape s1_shape = {10, 10, 0, 0};
            TensorShape s2_shape = {10, 10, 0, 0};
            TensorShape exp_shape = {10, 10, 0, 0};
            
            float d1[100], d2[100], exp_d[100];
            
            // Initialize matrices with simple patterns
            for(int i = 0; i < 100; i++) {
                d1[i] = (float)(i % 10 + 1); // 1,2,3,...,10,1,2,3,...
                d2[i] = (float)(i / 10 + 1); // 1,1,1,...,1,2,2,2,...
            }
            
            // Calculate expected result manually for verification
            for(int i = 0; i < 10; i++) {
                for(int j = 0; j < 10; j++) {
                    float sum = 0;
                    for(int k = 0; k < 10; k++) {
                        sum += d1[i * 10 + k] * d2[k * 10 + j];
                    }
                    exp_d[i * 10 + j] = sum;
                }
            }

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Large rectangular matrix multiplication (reduced size for stack allocation)
        {
            TensorShape s1_shape = {20, 15, 0, 0};
            TensorShape s2_shape = {15, 25, 0, 0};
            TensorShape exp_shape = {20, 25, 0, 0};
            
            // Use stack arrays with reduced size
            float d1[300]; // 20*15 = 300
            float d2[375]; // 15*25 = 375  
            float exp_d[500]; // 20*25 = 500
            
            // Initialize with simple patterns
            for(int i = 0; i < 300; i++) d1[i] = 1.0f;
            for(int i = 0; i < 375; i++) d2[i] = 1.0f;
            for(int i = 0; i < 500; i++) exp_d[i] = 15.0f; // Each element should be 15*1*1 = 15

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Special Matrix Content
    {
        const char* tc_name = "matmul_special_matrix_content";
        
        // Sub-test 1: Matrix with zeros
        {
            TensorShape s1_shape = {2, 2, 0, 0}; float d1[] = {0.0f, 0.0f, 1.0f, 1.0f};
            TensorShape s2_shape = {2, 2, 0, 0}; float d2[] = {1.0f, 2.0f, 3.0f, 4.0f};
            TensorShape exp_shape = {2, 2, 0, 0}; float exp_d[] = {0.0f, 0.0f, 4.0f, 6.0f};

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
