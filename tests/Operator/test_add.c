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
        const char* tc_name = "add_vector_3el";
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

    cten_free(pool_id);
}
