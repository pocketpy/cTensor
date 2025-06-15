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

    // Test Case 7: Larger Matrix Multiplication
    {
        const char* tc_name = "matmul_larger_matrices";
        
        // Sub-test 1: Larger matrix multiplication (4x3 * 3x5)
        {
            TensorShape s1_shape = {4, 3, 0, 0};
            float d1[] = {0.4008f, 0.5596f, 0.1552f, 0.1819f, 0.8618f, 0.9461f, 0.3733f, 0.2707f, 0.6440f, 0.4087f, 0.0254f, 0.1562f};
            TensorShape s2_shape = {3, 5, 0, 0};
            float d2[] = {0.7160f, 0.6589f, 0.0271f, 0.2220f, 0.2311f, 0.6719f, 0.0197f, 0.1041f, 0.7999f, 0.1785f, 0.6527f, 0.2382f, 0.0994f, 0.2432f, 0.7223f};
            TensorShape exp_shape = {4, 5, 0, 0};
            float exp_d[] = {0.7643f, 0.3121f, 0.0846f, 0.5744f, 0.3047f, 1.3269f, 0.3622f, 0.1887f, 0.9598f, 0.8793f, 0.8696f, 0.4047f, 0.1023f, 0.4560f, 0.5997f, 0.4116f, 0.3070f, 0.0292f, 0.1490f, 0.2118f};

            Tensor t1 = create_test_tensor(s1_shape, d1, false);
            Tensor t2 = create_test_tensor(s2_shape, d2, false);
            Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
            Tensor actual_res = Tensor_matmul(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    // TODO : Currently MatMul Doesnt support batch matrix multiplication
    // 
    // // Test Case 8: Batch Matrix Multiplication
    // {
    //     const char* tc_name = "matmul_batch_matrices";
        
    //     // Sub-test 1: Batch matrix multiplication (2x3x4 * 2x4x5)
    //     {
    //         TensorShape s1_shape = {2, 3, 4, 0};
    //         float d1[] = {0.9256f, 0.4219f, 0.3916f, 0.6438f, 0.8790f, 0.0543f, 0.0463f, 0.5632f, 0.7813f, 0.9841f, 0.7979f, 0.8884f, 0.5976f, 0.0739f, 0.8306f, 0.0435f, 0.2653f, 0.7424f, 0.9176f, 0.6326f, 0.2545f, 0.6777f, 0.9430f, 0.4921f};
    //         TensorShape s2_shape = {2, 4, 5, 0};
    //         float d2[] = {0.1146f, 0.8401f, 0.0189f, 0.9417f, 0.9551f, 0.3073f, 0.5162f, 0.6919f, 0.3872f, 0.9831f, 0.8261f, 0.6104f, 0.1850f, 0.4844f, 0.0732f, 0.8003f, 0.3244f, 0.6337f, 0.4984f, 0.1917f, 0.5972f, 0.8280f, 0.1163f, 0.1445f, 0.5281f, 0.3753f, 0.7377f, 0.0097f, 0.0460f, 0.8825f, 0.1283f, 0.3434f, 0.9592f, 0.2614f, 0.8935f, 0.9233f, 0.1056f, 0.1819f, 0.9243f, 0.1263f};
    //         TensorShape exp_shape = {2, 3, 5, 0};
    //         float exp_d[] = {1.0745f, 1.4433f, 0.7899f, 1.5456f, 1.4509f, 0.6064f, 0.9774f, 0.4197f, 1.1520f, 1.0043f, 1.7620f, 1.9396f, 1.4062f, 1.9461f, 1.9424f, 0.5314f, 0.8391f, 0.8748f, 0.3471f, 1.1284f, 1.1388f, 1.1492f, 1.0333f, 0.8970f, 1.6950f, 0.9817f, 1.0865f, 1.0302f, 0.7693f, 1.6373f};

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_matmul(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }
    // }

    // Test Case 9: Special Matrix Content
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
    // TODO: Problem in Broadcasting
    
    // // Test Case 10: Broadcasting
    // {
    //     const char* tc_name = "matmul_broadcasting";
        
    //     // Sub-test 1: Simple matrix multiplication {4,5} @ {5,3} -> {4,3}
    //     {
    //         TensorShape s1_shape = {4, 5, 0, 0};
    //         float d1[] = {
    //             0.3745f, 0.9507f, 0.7320f, 0.5987f, 0.1560f,  // Row 0
    //             0.1560f, 0.0581f, 0.8662f, 0.6011f, 0.7081f,  // Row 1
    //             0.0206f, 0.9699f, 0.8324f, 0.2123f, 0.1818f,  // Row 2
    //             0.1834f, 0.3042f, 0.5248f, 0.4319f, 0.2912f,  // Row 3
    //         };
            
    //         TensorShape s2_shape = {5, 3, 0, 0};
    //         float d2[] = {
    //             0.6119f, 0.1395f, 0.2921f,  // Row 0
    //             0.3664f, 0.4561f, 0.7852f,  // Row 1
    //             0.1997f, 0.5142f, 0.5924f,  // Row 2
    //             0.0465f, 0.6075f, 0.1705f,  // Row 3
    //             0.0651f, 0.9489f, 0.9656f,  // Row 4
    //         };
            
    //         TensorShape exp_shape = {4, 3, 0, 0};
    //         float exp_d[] = {
    //             0.7616f, 1.3740f, 1.5423f,  // Row 0
    //             0.3637f, 1.5308f, 1.3906f,  // Row 1
    //             0.5558f, 1.1748f, 1.4725f,  // Row 2
    //             0.3675f, 0.9730f, 0.9582f,  // Row 3
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_matmul(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 3D Broadcasting {1,3,2} @ {2,2,4} -> {2,3,4}
    //     {
    //         TensorShape s1_shape = {1, 3, 2, 0};
    //         float d1[] = {
    //             0.8084f, 0.3046f,  // [0,0,:]
    //             0.0977f, 0.6842f,  // [0,1,:]
    //             0.4402f, 0.1220f,  // [0,2,:]
    //         };
            
    //         TensorShape s2_shape = {2, 2, 4, 0};
    //         float d2[] = {
    //             // Batch 0
    //             0.4952f, 0.0344f, 0.9093f, 0.2588f,  // [0,0,:]
    //             0.6625f, 0.3117f, 0.5201f, 0.5467f,  // [0,1,:]
    //             // Batch 1
    //             0.1849f, 0.9696f, 0.7751f, 0.9395f,  // [1,0,:]
    //             0.8948f, 0.5979f, 0.9219f, 0.0885f,  // [1,1,:]
    //         };
            
    //         TensorShape exp_shape = {2, 3, 4, 0};
    //         float exp_d[] = {
    //             // Batch 0
    //             0.6021f, 0.1228f, 0.8935f, 0.3757f,  // [0,0,:]
    //             0.5017f, 0.2166f, 0.4447f, 0.3994f,  // [0,1,:]
    //             0.2988f, 0.0532f, 0.4637f, 0.1806f,  // [0,2,:]
    //             // Batch 1
    //             0.4220f, 0.9659f, 0.9074f, 0.7864f,  // [1,0,:]
    //             0.6303f, 0.5038f, 0.7065f, 0.1523f,  // [1,1,:]
    //             0.1906f, 0.4997f, 0.4537f, 0.4243f,  // [1,2,:]
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_matmul(t1, t2);
    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 3: 4D Broadcasting {2,1,2,3} @ {1,1,3,2} -> {2,1,2,2}
    //     {
    //         TensorShape s1_shape = {2, 1, 2, 3};
    //         float d1[] = {
    //             // Batch 0
    //             0.1960f, 0.0452f, 0.3253f,  // [0,0,0,:]
    //             0.3887f, 0.2713f, 0.8287f,  // [0,0,1,:]
    //             // Batch 1
    //             0.3568f, 0.2809f, 0.5427f,  // [1,0,0,:]
    //             0.1409f, 0.8022f, 0.0746f,  // [1,0,1,:]
    //         };
            
    //         TensorShape s2_shape = {1, 1, 3, 2};
    //         float d2[] = {
    //             0.9869f, 0.7722f,  // [0,0,0,:]
    //             0.1987f, 0.0055f,  // [0,0,1,:]
    //             0.8155f, 0.7069f,  // [0,0,2,:]
    //         };
            
    //         TensorShape exp_shape = {2, 1, 2, 2};
    //         float exp_d[] = {
    //             // Batch 0
    //             0.4677f, 0.3816f,  // [0,0,0,:]
    //             1.1133f, 0.8875f,  // [0,0,1,:]
    //             // Batch 1
    //             0.8504f, 0.6607f,  // [1,0,0,:]
    //             0.3593f, 0.1660f,  // [1,0,1,:]
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_matmul(t1, t2);
    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
    //     }     
    // }

    cten_free(pool_id);
}
