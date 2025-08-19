#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_linear_backward() {
    const char* op_name = "linear_backward";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple linear backward
    {
        const char* tc_name = "Simple_linear_backward";
        // Sub-test 1: Basic linear layer
        {
            TensorShape input_shape = {1, 3};   // batch_size=1, input_features=3
            TensorShape weight_shape = {3, 2};  // input_features=3, output_features=2
            TensorShape bias_shape = {1, 2};    // output_features=2

            float input_data[] = {1.0f, 2.0f, 3.0f};
            float weight_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
            float bias_data[] = {0.1f, 0.2f};

            // Expected gradients
            float exp_grad_input[] = {0.3f, 0.7f, 1.1f};  // input_grad = weight.T @ grad_output
            float exp_grad_weight[] =
                {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};  // weight_grad = input.T @ grad_output
            float exp_grad_bias[] = {1.0f, 1.0f};      // bias_grad = sum(grad_output, dim=0)

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor output = nn_linear(input, weight, bias);

            // Create a gradient for the output
            TensorShape grad_shape = {1, 2};  // Same as output shape
            float grad_data[] = {1.0f, 1.0f};
            Tensor grad_output = create_test_tensor(grad_shape, grad_data, false);

            Tensor_backward(output, grad_output);

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
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Batch input linear backward
    {
        const char* tc_name = "Batch_linear_backward";
        // Sub-test 1: Batch size > 1
        {
            TensorShape input_shape = {2, 3};   // batch_size=2, input_features=3
            TensorShape weight_shape = {3, 2};  // input_features=3, output_features=2
            TensorShape bias_shape = {1, 2};    // output_features=2

            float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float weight_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
            float bias_data[] = {0.1f, 0.2f};

            // Expected gradients
            float exp_grad_input[] = {0.3f, 0.7f, 1.1f, 0.3f, 0.7f, 1.1f};
            float exp_grad_weight[] = {5.0f, 7.0f, 9.0f, 5.0f, 7.0f, 9.0f};
            float exp_grad_bias[] = {2.0f, 2.0f};  // Sum over batch dimension

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor output = nn_linear(input, weight, bias);

            // Create a gradient for the output
            TensorShape grad_shape = {2, 2};  // Same as output shape
            float grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
            Tensor grad_output = create_test_tensor(grad_shape, grad_data, false);

            Tensor_backward(output, grad_output);

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
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 3: Random input linear backward
    {
        const char* tc_name = "Random_input_linear_backward";
        // Sub-test 1: Random input values
        {
            TensorShape input_shape = {2, 4};   // batch_size=2, input_features=4
            TensorShape weight_shape = {4, 3};  // input_features=4, output_features=3
            TensorShape bias_shape = {1, 3};    // output_features=3

            float input_data[] = {0.5f, 1.3f, 2.7f, 0.8f, 1.9f, 0.4f, 1.2f, 3.1f};
            float weight_data[] =
                {0.2f, 0.1f, 0.3f, 0.5f, 0.4f, 0.2f, 0.1f, 0.7f, 0.6f, 0.3f, 0.2f, 0.8f};
            float bias_data[] = {0.5f, 0.3f, 0.2f};

            // Expected gradients for a gradient of ones at the output
            float exp_grad_bias[] = {2.0f, 2.0f, 2.0f};  // Sum over batch dimension

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor output = nn_linear(input, weight, bias);

            // Create a gradient for the output
            TensorShape grad_shape = {2, 3};  // Same as output shape
            float grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            Tensor grad_output = create_test_tensor(grad_shape, grad_data, false);

            Tensor_backward(output, grad_output);

            Tensor expected_grad_bias = create_test_tensor(bias_shape, exp_grad_bias, false);

            // Focus on bias gradient
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Different gradient values
        {
            TensorShape input_shape = {3, 2};   // batch_size=3, input_features=2
            TensorShape weight_shape = {2, 4};  // input_features=2, output_features=4
            TensorShape bias_shape = {1, 4};    // output_features=4

            float input_data[] = {1.5f, 2.3f, 0.7f, 1.8f, 3.2f, 0.9f};
            float weight_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
            float bias_data[] = {0.1f, 0.2f, 0.3f, 0.4f};

            // Create a non-uniform gradient for the output
            TensorShape grad_shape = {3, 4};  // Same as output shape
            float grad_data[] =
                {0.5f, 1.0f, 1.5f, 2.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 0.8f, 0.6f, 0.4f};

            // Expected bias gradient is the sum of the output gradient across the batch dimension
            float exp_grad_bias[] = {1.6f, 2.0f, 2.4f, 2.8f};

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor output = nn_linear(input, weight, bias);
            Tensor grad_output = create_test_tensor(grad_shape, grad_data, false);

            Tensor_backward(output, grad_output);

            Tensor expected_grad_bias = create_test_tensor(bias_shape, exp_grad_bias, false);

            // Focus on bias gradient
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 4: Chained operations with linear
    {
        const char* tc_name = "Chained_operations_with_linear";
        // Sub-test 1: Linear followed by sum
        {
            TensorShape input_shape = {2, 3};   // batch_size=2, input_features=3
            TensorShape weight_shape = {3, 2};  // input_features=3, output_features=2
            TensorShape bias_shape = {1, 2};    // output_features=2

            float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float weight_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
            float bias_data[] = {0.1f, 0.2f};

            // Expected gradients
            float exp_grad_bias[] = {2.0f, 2.0f};  // For sum reduction

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor output = nn_linear(input, weight, bias);
            Tensor sum_output = Tensor_sum(output);

            Tensor_backward(sum_output, (Tensor){0});

            Tensor expected_grad_bias = create_test_tensor(bias_shape, exp_grad_bias, false);

            // Focus on bias gradient
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Linear followed by mean
        {
            TensorShape input_shape = {2, 3};   // batch_size=2, input_features=3
            TensorShape weight_shape = {3, 2};  // input_features=3, output_features=2
            TensorShape bias_shape = {1, 2};    // output_features=2

            float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float weight_data[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
            float bias_data[] = {0.1f, 0.2f};

            // Expected gradients
            float exp_grad_bias[] = {0.5f, 0.5f};  // For mean reduction (1/2)

            Tensor input = create_test_tensor(input_shape, input_data, true);
            Tensor weight = create_test_tensor(weight_shape, weight_data, true);
            Tensor bias = create_test_tensor(bias_shape, bias_data, true);

            Tensor output = nn_linear(input, weight, bias);
            Tensor mean_output = Tensor_mean(output);

            Tensor_backward(mean_output, (Tensor){0});

            Tensor expected_grad_bias = create_test_tensor(bias_shape, exp_grad_bias, false);

            // Focus on bias gradient
            compare_tensors(&bias.node->grad,
                            &expected_grad_bias,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    cten_free(pool_id);
}