#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_relu_backward() {
    const char* op_name = "relu_backward";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple ReLU backward
    {
        const char* tc_name = "Simple_relu_backward";
        // Sub-test 1: Scalar ReLU
        {
            TensorShape s_shape = {1};
            float d1[] = {2.0f};         // Positive value
            float exp_grad1[] = {1.0f};  // Gradient passes through for positive values

            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor z = nn_relu(t1);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad1 = create_test_tensor(s_shape, exp_grad1, false);

            compare_tensors(&t1.node->grad,
                            &expected_grad1,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Scalar ReLU with negative input
        {
            TensorShape s_shape = {1};
            float d1[] = {-2.0f};        // Negative value
            float exp_grad1[] = {0.0f};  // Gradient is zero for negative values

            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor z = nn_relu(t1);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad1 = create_test_tensor(s_shape, exp_grad1, false);

            compare_tensors(&t1.node->grad,
                            &expected_grad1,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Vector ReLU
        {
            TensorShape v_shape = {4};
            float d1[] = {-1.0f, 0.0f, 1.0f, 2.0f};
            float exp_grad1[] = {0.0f, 0.0f, 1.0f, 1.0f};  // Gradient is 0 for x <= 0, 1 for x > 0

            Tensor t1 = create_test_tensor(v_shape, d1, true);
            Tensor z = nn_relu(t1);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad1 = create_test_tensor(v_shape, exp_grad1, false);

            compare_tensors(&t1.node->grad,
                            &expected_grad1,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Matrix ReLU backward
    {
        const char* tc_name = "Matrix_relu_backward";
        // Sub-test 1: Matrix with mixed values
        {
            TensorShape m_shape = {2, 3};
            float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -3.0f, 4.0f};
            float exp_grad[] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor z = nn_relu(t);
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
    }

    // Test Case 3: More ReLU backward
    {
        const char* tc_name = "More_relu_backward";
        // Sub-test 1: Random values
        {
            TensorShape v_shape = {6};
            float data[] = {-2.5f, 1.3f, 0.0f, -0.7f, 3.2f, -1.8f};
            float exp_grad[] = {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

            Tensor t = create_test_tensor(v_shape, data, true);
            Tensor z = nn_relu(t);
            Tensor l = Tensor_sum(z);
            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: 3D tensor with random values
        {
            TensorShape tensor3d_shape = {2, 2, 2};
            float data[] = {-1.5f, 2.7f, 0.0f, -3.1f, 4.2f, -0.8f, 1.9f, 0.0f};
            float exp_grad[] = {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

            Tensor t = create_test_tensor(tensor3d_shape, data, true);
            Tensor z = nn_relu(t);
            Tensor l = Tensor_sum(z);
            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad = create_test_tensor(tensor3d_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 4: Custom gradient ReLU backward
    {
        const char* tc_name = "Custom_gradient_relu_backward";
        // Sub-test 1: Non-uniform gradient
        {
            TensorShape v_shape = {4};
            float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
            float grad_data[] = {0.5f, 1.0f, 1.5f, 2.0f};
            float exp_grad[] = {
                0.0f,
                0.0f,
                1.5f,
                2.0f};  // Element-wise product of input gradient and ReLU derivative

            Tensor t = create_test_tensor(v_shape, data, true);
            Tensor z = nn_relu(t);
            Tensor l = Tensor_sum(z);
            TensorShape grad_shape = {4};
            Tensor grad = create_test_tensor(grad_shape, grad_data, false);

            Tensor_backward(l, grad);

            Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 5: Chained operations with ReLU
    {
        const char* tc_name = "Chained_operations_with_relu";
        // Sub-test 1,2,3: Linear -> ReLU -> Sum
        {
            TensorShape input_shape = {2, 3};
            TensorShape weight_shape = {3, 4};
            TensorShape bias_shape = {1, 4};

            float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float weight_data[] =
                {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, 1.0f, -1.1f, 1.2f};
            float bias_data[] = {0.1f, -0.1f, 0.2f, -0.2f};

            float exp_grad_input[] = {0.3f, -0.9f, 3.1f, 0.3f, -0.9f, 3.1f};
            float exp_grad_weight[] =
                {5.0f, 5.0f, 0.0f, 5.0f, 7.0f, 7.0f, 0.0f, 7.0f, 9.0f, 9.0f, 0.0f, 9.0f};
            float exp_grad_bias[] = {2.0f, 2.0f, 0.0f, 2.0f};

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor linear_output = nn_linear(input, weight, bias);
            Tensor relu_output = nn_relu(linear_output);
            Tensor sum_output = Tensor_sum(relu_output);

            Tensor_backward(sum_output, (Tensor){0});

            Tensor expected_grad_input = create_test_tensor(input_shape, exp_grad_input, false);
            Tensor expected_grad_weight = create_test_tensor(weight_shape, exp_grad_weight, false);
            Tensor expected_grad_bias = create_test_tensor(bias_shape, exp_grad_bias, false);

            compare_tensors(&input.node->grad,
                            &expected_grad_input,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&weight.node->grad,
                            &expected_grad_weight,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 4: ReLU -> Mean with mixed values
        {
            TensorShape m_shape = {2, 3};
            float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -3.0f, 4.0f};
            float exp_grad[] = {0.0f,
                                0.0f,
                                1.0f / 6.0f,
                                1.0f / 6.0f,
                                0.0f,
                                1.0f / 6.0f};  // 1/6 for positive values, 0 for negative

            Tensor t = create_test_tensor(m_shape, data, true);
            Tensor relu_output = nn_relu(t);
            Tensor mean_output = Tensor_mean(relu_output);
            Tensor l = Tensor_sum(mean_output);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            4,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}