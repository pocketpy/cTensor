#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_sum_backward() {
    const char* op_name = "sum_backward";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Sum all elements backward
    {
        const char* tc_name = "Sum_all_backward";
        // Sub-test 1: Vector sum all
        {
            TensorShape v_shape = {3};
            float data[] = {1.0f, 2.0f, 3.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(v_shape, data, true);
            Tensor z = Tensor_sum(t);  // sum all elements

            Tensor_backward(z, (Tensor){0});

            Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Matrix sum all
        {
            TensorShape m_shape = {2, 3};
            float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor z = Tensor_sum(t);  // sum all elements

            Tensor_backward(z, (Tensor){0});

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: 3D tensor sum all
        {
            TensorShape tensor3d_shape = {2, 2, 2};
            float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(tensor3d_shape, data, true);
            Tensor z = Tensor_sum(t);  // sum all elements

            Tensor_backward(z, (Tensor){0});

            Tensor expected_grad = create_test_tensor(tensor3d_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Sum along dimension 0
    {
        const char* tc_name = "Sum_dim0_backward";
        // Sub-test 1: Matrix sum along dim 0
        {
            TensorShape m_shape = {2, 3};
            float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor z = Tensor_sum(t, 0);  // sum along dim 0. Shape is {3}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 3D tensor sum along dim 0
        {
            TensorShape tensor3d_shape = {2, 3, 4};
            float data[24];
            for(int i = 0; i < 24; i++)
                data[i] = (float)(i + 1);
            float exp_grad[24];
            for(int i = 0; i < 24; i++)
                exp_grad[i] = 1.0f;

            Tensor t = create_test_tensor(tensor3d_shape, data, true);
            Tensor z = Tensor_sum(t, 0);  // sum along dim 0. Shape is {3, 4}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad = create_test_tensor(tensor3d_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 3: Sum along dimension 1
    {
        const char* tc_name = "Sum_dim1_backward";
        // Sub-test 1: Matrix sum along dim 1
        {
            TensorShape m_shape = {2, 3};
            float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor z = Tensor_sum(t, 1);  // sum along dim 1 -> shape {2}
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 3D tensor sum along dim 1
        {
            TensorShape tensor3d_shape = {2, 3, 4};
            float data[24];
            for(int i = 0; i < 24; i++)
                data[i] = (float)(i + 1);
            float exp_grad[24];
            for(int i = 0; i < 24; i++)
                exp_grad[i] = 1.0f;

            Tensor t = create_test_tensor(tensor3d_shape, data, true);
            Tensor z = Tensor_sum(t, 1);  // sum along dim 1 -> shape {2, 4}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad = create_test_tensor(tensor3d_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 4: Sum along dimension 2
    {
        const char* tc_name = "Sum_dim2_backward";
        // Sub-test 1: 3D tensor sum along dim 2
        {
            TensorShape tensor3d_shape = {2, 3, 4};
            float data[24];
            for(int i = 0; i < 24; i++)
                data[i] = (float)(i + 1);
            float exp_grad[24];
            for(int i = 0; i < 24; i++)
                exp_grad[i] = 1.0f;

            Tensor t = create_test_tensor(tensor3d_shape, data, true);
            Tensor z = Tensor_sum(t, 2);  // sum along dim 2 -> shape {2, 3}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad = create_test_tensor(tensor3d_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 5: Random input sum backward
    {
        const char* tc_name = "Random_input_sum_backward";
        // Sub-test 1: Random matrix sum along dim 0
        {
            TensorShape m_shape = {3, 4};
            float data[] = {2.5f, 1.3f, 4.8f, 3.2f, 0.7f, 5.1f, 2.9f, 6.4f, 3.6f, 1.8f, 4.2f, 0.9f};
            float exp_grad[] =
                {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor z = Tensor_sum(t, 0);  // sum along dim 0 -> shape {4}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Random matrix sum along dim 1
        {
            TensorShape m_shape = {3, 4};
            float data[] = {2.5f, 1.3f, 4.8f, 3.2f, 0.7f, 5.1f, 2.9f, 6.4f, 3.6f, 1.8f, 4.2f, 0.9f};
            float exp_grad[] =
                {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor z = Tensor_sum(t, 1);  // sum along dim 1 -> shape {3}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 6: Chained operations with sum
    {
        const char* tc_name = "Chained_operations_with_sum";
        // Sub-test 1: Sum(a*b, dim=0)
        {
            TensorShape m_shape = {2, 3};
            float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float b_data[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
            float exp_grad_a[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
            float exp_grad_b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

            Tensor a = create_test_tensor(m_shape, a_data, true);
            Tensor b = create_test_tensor(m_shape, b_data, true);
            Tensor prod = Tensor_mul(a, b);
            Tensor z = Tensor_sum(prod, 0);  // shape {3}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad_a = create_test_tensor(m_shape, exp_grad_a, false);
            Tensor expected_grad_b = create_test_tensor(m_shape, exp_grad_b, false);

            compare_tensors(&a.node->grad,
                            &expected_grad_a,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&b.node->grad,
                            &expected_grad_b,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Sum(a+b, dim=1)
        {
            TensorShape m_shape = {2, 3};
            float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float b_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
            float exp_grad_a[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_b[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

            Tensor a = create_test_tensor(m_shape, a_data, true);
            Tensor b = create_test_tensor(m_shape, b_data, true);
            Tensor sum_ab = Tensor_add(a, b);
            Tensor z = Tensor_sum(sum_ab, 1);  // shape {2}

            Tensor grad_for_z = Tensor_ones(z.shape, false);
            Tensor_backward(z, grad_for_z);

            Tensor expected_grad_a = create_test_tensor(m_shape, exp_grad_a, false);
            Tensor expected_grad_b = create_test_tensor(m_shape, exp_grad_b, false);

            compare_tensors(&a.node->grad,
                            &expected_grad_a,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&b.node->grad,
                            &expected_grad_b,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}