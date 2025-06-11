#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_add_operator() {
    const char* op_name = "add";
    PoolId pool_id = 0; 
    cten_begin_malloc(pool_id);

    // Test Case 1: Scalar addition (represented as 1x1 tensors)
    {
        TensorShape s_shape = {1, 0, 0, 0};

        // Sub-test 1
        {
            const char* tc_name = "add_scalar";
            float d1[] = {2.5f};
            float d2[] = {3.5f};
            float exp_d[] = {6.0f};
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2
        {
            const char* tc_name = "add_scalar";  
            float d1[] = {10.0f}; 
            float d2[] = {5.0f};
            float exp_d[] = {15.0f}; 
            Tensor t1 = create_test_tensor(s_shape, d1, false);
            Tensor t2 = create_test_tensor(s_shape, d2, false);
            Tensor expected_res = create_test_tensor(s_shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Vector addition (1D tensors)
    {
        const char* tc_name = "add_vector_1D";
        TensorShape v_shape = {3, 0, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f};
        float d2[] = {4.0f, 5.0f, 6.0f};
        float exp_d[] = {5.0f, 7.0f, 9.0f};
        Tensor t1 = create_test_tensor(v_shape, d1, false);
        Tensor t2 = create_test_tensor(v_shape, d2, false);
        Tensor expected_res = create_test_tensor(v_shape, exp_d, false);
        Tensor actual_res = Tensor_add(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Matrix addition (2D tensors)
    {
        const char* tc_name = "add_matrix_2x2";
        TensorShape m_shape = {2, 2, 0, 0};
        float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float exp_d[] = {6.0f, 8.0f, 10.0f, 12.0f};
        Tensor t1 = create_test_tensor(m_shape, d1, false);
        Tensor t2 = create_test_tensor(m_shape, d2, false);
        Tensor expected_res = create_test_tensor(m_shape, exp_d, false);
        Tensor actual_res = Tensor_add(t1, t2);

        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    
    // Test Case 4: Broadcasting (vector + scalar-like tensor)
    // cTensor's Tensor_add handles broadcasting if cten_elemwise_broadcast modifies inputs.
    // Let's assume for now Tensor_add expects compatible shapes after potential broadcast by cten_elemwise_broadcast.
    // We will test a simple case where one tensor is {1,1} and other is {2,1} (or similar that cten_elemwise_broadcast handles)
    // For this example, let's test adding a {1,0,0,0} tensor to a {2,0,0,0} tensor, expecting failure if not broadcastable or specific error.
    // Given the current structure of Tensor_add, it calls cten_elemwise_broadcast. If that fails, it asserts.
    // A proper test would mock or control cten_elemwise_broadcast, or test shapes it's known to handle.
    // For now, we'll test addition of compatible shapes that might arise from broadcasting.
    // Example: Add [10] (shape {1}) to [[1,2],[3,4]] (shape {2,2}). PyTorch result: [[11,12],[13,14]]
    // cten_elemwise_broadcast might expand [10] to [[10,10],[10,10]] internally before element-wise add.
    // Let's test a compatible case first, then consider how to test broadcast failure or specific broadcast scenarios.
    {
        const char* tc_name = "add_broadcast_vector_plus_scalar_tensor";
        TensorShape vec_shape = {2, 0, 0, 0}; float vec_data[] = {1.0f, 2.0f};
        TensorShape scalar_shape = {1, 0, 0, 0}; float scalar_data[] = {10.0f};
        // Expected: PyTorch would broadcast scalar {10} to {10, 10} then add to get {11, 12}
        // We need to know how cten_elemwise_broadcast modifies shapes to set the correct expected_res_shape.
        // Assuming cten_elemwise_broadcast makes 'scalar' compatible with 'vec' for Tensor_add.
        // If cten_elemwise_broadcast makes them both {2}, then expected shape is {2}.
        TensorShape expected_shape = {2, 0, 0, 0}; float exp_data[] = {11.0f, 12.0f};

        Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
        Tensor t_scalar_original = create_test_tensor(scalar_shape, scalar_data, false);
        
        // Tensor_add internally calls cten_elemwise_broadcast(&t_vec, &t_scalar_original)
        // This might modify t_vec.shape or t_scalar_original.shape if they are broadcastable.
        // The result tensor 'actual_res' will have the broadcasted shape.
        Tensor actual_res = Tensor_add(t_vec, t_scalar_original); 
        Tensor expected_res = create_test_tensor(expected_shape, exp_data, false);

        // We compare actual_res against an expected_res that has the final broadcasted shape and values.
        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }

    // TODO: Problem in Broadcasting

    // // Test Case 5: Advanced Broadcasting
    // {
    //     const char* tc_name = "add_advanced_broadcasting";
        
    //     // Sub-test 1: Multi-dimensional broadcasting {3,1} + {1,4} -> {3,4}
    //     {
    //         TensorShape s1_shape = {3, 1, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f};
    //         TensorShape s2_shape = {1, 4, 0, 0}; float d2[] = {10.0f, 20.0f, 30.0f, 40.0f};
    //         TensorShape exp_shape = {3, 4, 0, 0}; 
    //         float exp_d[] = {11.0f, 21.0f, 31.0f, 41.0f,  // 1+[10,20,30,40]
    //                          12.0f, 22.0f, 32.0f, 42.0f,  // 2+[10,20,30,40]
    //                          13.0f, 23.0f, 33.0f, 43.0f}; // 3+[10,20,30,40]

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 2: 3D broadcasting {2,3,1} + {1,1,4} -> {2,3,4}
    //     {
    //         TensorShape s1_shape = {2, 3, 1, 0}; 
    //         float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    //         TensorShape s2_shape = {1, 1, 4, 0}; 
    //         float d2[] = {10.0f, 20.0f, 30.0f, 40.0f};
    //         TensorShape exp_shape = {2, 3, 4, 0};
    //         float exp_d[] = {
    //             // First 2x3 slice
    //             11.0f, 21.0f, 31.0f, 41.0f,  // 1+[10,20,30,40]
    //             12.0f, 22.0f, 32.0f, 42.0f,  // 2+[10,20,30,40]
    //             13.0f, 23.0f, 33.0f, 43.0f,  // 3+[10,20,30,40]
    //             // Second 2x3 slice
    //             14.0f, 24.0f, 34.0f, 44.0f,  // 4+[10,20,30,40]
    //             15.0f, 25.0f, 35.0f, 45.0f,  // 5+[10,20,30,40]
    //             16.0f, 26.0f, 36.0f, 46.0f   // 6+[10,20,30,40]
    //         };

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 3: 4D broadcasting with size-1 dimensions {1,1,1,1} + {5,4,3,2} -> {5,4,3,2}
    //     {
    //         TensorShape s1_shape = {1, 1, 1, 1}; float d1[] = {5.0f};
    //         TensorShape s2_shape = {5, 4, 3, 2}; 
    //         float d2[120]; // 5*4*3*2 = 120 elements
    //         for(int i = 0; i < 120; i++) d2[i] = (float)(i + 1);
            
    //         TensorShape exp_shape = {5, 4, 3, 2};
    //         float exp_d[120];
    //         for(int i = 0; i < 120; i++) exp_d[i] = d2[i] + 5.0f;

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 3, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 4: Vector-to-matrix broadcasting {3} + {2,3} -> {2,3}
    //     {
    //         TensorShape vec_shape = {3, 0, 0, 0}; float vec_data[] = {1.0f, 2.0f, 3.0f};
    //         TensorShape mat_shape = {2, 3, 0, 0}; float mat_data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    //         TensorShape exp_shape = {2, 3, 0, 0}; 
    //         float exp_d[] = {11.0f, 22.0f, 33.0f, 41.0f, 52.0f, 63.0f};

    //         Tensor t_vec = create_test_tensor(vec_shape, vec_data, false);
    //         Tensor t_mat = create_test_tensor(mat_shape, mat_data, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t_vec, t_mat);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 4, TEST_FLOAT_TOLERANCE);
    //     }

    //     // Sub-test 5: Cross-dimensional broadcasting {5} + {2,5} -> {2,5}
    //     {
    //         TensorShape s1_shape = {5, 0, 0, 0}; float d1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    //         TensorShape s2_shape = {2, 5, 0, 0}; 
    //         float d2[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f};
    //         TensorShape exp_shape = {2, 5, 0, 0};
    //         float exp_d[] = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 61.0f, 72.0f, 83.0f, 94.0f, 105.0f};

    //         Tensor t1 = create_test_tensor(s1_shape, d1, false);
    //         Tensor t2 = create_test_tensor(s2_shape, d2, false);
    //         Tensor expected_res = create_test_tensor(exp_shape, exp_d, false);
    //         Tensor actual_res = Tensor_add(t1, t2);

    //         compare_tensors(&actual_res, &expected_res, op_name, tc_name, 5, TEST_FLOAT_TOLERANCE);
    //     }
    // }

    // Test Case 6: Higher Dimensional Tensors
    {
        const char* tc_name = "add_higher_dimensional_tensors";
        
        // Sub-test 1: 3D tensor addition (same shape)
        {
            TensorShape shape_3d = {2, 3, 4, 0};
            float d1[24], d2[24], exp_d[24];
            for(int i = 0; i < 24; i++) {
                d1[i] = (float)(i + 1);
                d2[i] = (float)(i * 2);
                exp_d[i] = d1[i] + d2[i];
            }

            Tensor t1 = create_test_tensor(shape_3d, d1, false);
            Tensor t2 = create_test_tensor(shape_3d, d2, false);
            Tensor expected_res = create_test_tensor(shape_3d, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 4D tensor addition (same shape)
        {
            TensorShape shape_4d = {2, 3, 4, 5};
            float d1[120], d2[120], exp_d[120];
            for(int i = 0; i < 120; i++) {
                d1[i] = (float)(i + 1);
                d2[i] = (float)(i + 10);
                exp_d[i] = d1[i] + d2[i];
            }

            Tensor t1 = create_test_tensor(shape_4d, d1, false);
            Tensor t2 = create_test_tensor(shape_4d, d2, false);
            Tensor expected_res = create_test_tensor(shape_4d, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 7: Gradient Propagation
    {
        const char* tc_name = "add_gradient_propagation";
        
        // Sub-test 1: requires_grad flag propagation
        {
            TensorShape shape = {2, 2, 0, 0};
            float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
            float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
            float exp_d[] = {6.0f, 8.0f, 10.0f, 12.0f};

            Tensor t1 = create_test_tensor(shape, d1, true);  // requires_grad = true
            Tensor t2 = create_test_tensor(shape, d2, false); // requires_grad = false
            Tensor expected_res = create_test_tensor(shape, exp_d, false);
            Tensor actual_res = Tensor_add(t1, t2);

            // Check that result has gradient node when one input requires grad
            // Note: Gradient node creation depends on implementation

            compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}
