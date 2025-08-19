#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_div_backward() {
    const char* op_name = "div_backward";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple element-wise vector division
    {
        const char* tc_name = "div_vectors_backward";
        TensorShape shape = {3};
        float x_data[] = {6.7548f, 3.4753f, -7.6282f};
        float y_data[] = {4.5687f, 2.6877f, -1.8746f};

        // z = x / y = [1.4785, 1.2930, 4.0692]
        // loss = sum(z) = 6.8407
        float exp_grad_x[] = {0.218881f, 0.372065f, -0.533447f};
        float exp_grad_y[] = {-0.323614f, -0.481095f, 2.170725f};

        Tensor x = create_test_tensor(shape, x_data, true);
        Tensor y = create_test_tensor(shape, y_data, true);

        Tensor z = Tensor_div(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Broadcasting a vector by a scalar
    {
        const char* tc_name = "div_broadcast_vec_scalar_backward";
        TensorShape x_shape = {2};
        TensorShape y_shape = {1};
        float x_data[] = {1.2388f, -6.849f};
        float y_data[] = {-1.8818f};

        // z = x / y = [-0.6583, 3.6396]
        // loss = sum(z) = 2.9813
        float exp_grad_x[] = {-0.531406f, -0.531406f};
        float exp_grad_y[] = {1.584278f};

        Tensor x = create_test_tensor(x_shape, x_data, true);
        Tensor y = create_test_tensor(y_shape, y_data, true);

        Tensor z = Tensor_div(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(x_shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(y_shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Broadcasting a scalar by a vector
    {
        const char* tc_name = "div_broadcast_scalar_vec_backward";
        TensorShape x_shape = {1};
        TensorShape y_shape = {3};
        float x_data[] = {8.2849f};
        float y_data[] = {-4.2233f, 2.361f, 4.8289f};

        // z = x / y = [-1.9617, 3.5091, 1.7157]
        // loss = sum(z) = 3.2631
        float exp_grad_x[] = {0.393854f};
        float exp_grad_y[] = {-0.464498f, -1.486262f, -0.355296f};

        Tensor x = create_test_tensor(x_shape, x_data, true);
        Tensor y = create_test_tensor(y_shape, y_data, true);

        Tensor z = Tensor_div(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(x_shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(y_shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Matrix division with negative values
    {
        const char* tc_name = "div_matrices_neg_vals_backward";
        TensorShape shape = {2, 2};
        float x_data[] = {1.8347f, -8.6274f, -8.2642f, -5.8261f};
        float y_data[] = {-2.5141f, -4.3176f, -4.4468f, 3.8183f};

        // z = x / y = [-0.7298, 1.9982, 1.8585, -1.5258]
        // loss = sum(z) = 1.6011
        float exp_grad_x[] = {-0.397757f, -0.23161f, -0.224881f, 0.261897f};
        float exp_grad_y[] = {-0.290269f, 0.462802f, 0.417932f, 0.399611f};

        Tensor x = create_test_tensor(shape, x_data, true);
        Tensor y = create_test_tensor(shape, y_data, true);

        Tensor z = Tensor_div(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Complex computation graph (z = (a/b) * c)
    {
        const char* tc_name = "div_complex_graph_backward";
        TensorShape shape = {1};
        float a_data[] = {3.0511f};
        float b_data[] = {1.3192f};
        float c_data[] = {1.404f};

        // Let d = a / b. Then z = d * c.
        // Forward: d = 3.0511/1.3192 = 2.3129. z = 2.3129 * 1.404 = 3.2472
        // Backward pass:
        // dz/dc = d = 2.312841
        float exp_grad_c[] = {2.312841f};

        // dz/d(d) = c = 1.404 (This is the upstream gradient for the div op)
        // dz/da = (dz/dd) * (dd/da) = c * (1/b) = 1.404 * (1/1.3192) = 1.064281
        float exp_grad_a[] = {1.064281f};
        // dz/db = (dz/dd) * (dd/db) = c * (-a/b²) = 1.404 * (-3.0511/(1.3192*1.3192)) = -2.461514
        float exp_grad_b[] = {-2.461514f};

        Tensor a = create_test_tensor(shape, a_data, true);
        Tensor b = create_test_tensor(shape, b_data, true);
        Tensor c = create_test_tensor(shape, c_data, true);

        Tensor d = Tensor_div(a, b);
        Tensor z = Tensor_mul(d, c);

        Tensor grad_dummy = {0};
        Tensor_backward(z, grad_dummy);

        Tensor expected_grad_a_tensor = create_test_tensor(shape, exp_grad_a, false);
        Tensor expected_grad_b_tensor = create_test_tensor(shape, exp_grad_b, false);
        Tensor expected_grad_c_tensor = create_test_tensor(shape, exp_grad_c, false);

        compare_tensors(&a.node->grad,
                        &expected_grad_a_tensor,
                        op_name,
                        tc_name,
                        1,
                        TEST_FLOAT_TOLERANCE);
        compare_tensors(&b.node->grad,
                        &expected_grad_b_tensor,
                        op_name,
                        tc_name,
                        2,
                        TEST_FLOAT_TOLERANCE);
        compare_tensors(&c.node->grad,
                        &expected_grad_c_tensor,
                        op_name,
                        tc_name,
                        3,
                        TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}