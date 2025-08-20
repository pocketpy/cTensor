#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>
#include <math.h>

void test_pow_backward() {
    const char* op_name = "pow_backward";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple element-wise vector power
    {
        const char* tc_name = "pow_vectors_backward";
        TensorShape shape = {2};
        float x_data[] = {3.3774f, 0.6125f};
        float y_data[] = {1.4626f, 1.2812f};

        // z = x^y = [3.3774^1.4626, 0.6125^1.2812] = [6.9329, 0.5292]
        // loss = sum(z) = 7.4621
        float exp_grad_x[] = {2.568312f, 1.116224f};
        float exp_grad_y[] = {7.218272f, -0.261589f};

        Tensor x = create_test_tensor(shape, x_data, true);
        Tensor y = create_test_tensor(shape, y_data, true);

        Tensor z = Tensor_pow(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 2: Broadcasting a vector by a scalar exponent
    {
        const char* tc_name = "pow_broadcast_vec_scalar_backward";
        TensorShape x_shape = {2};
        TensorShape y_shape = {1};
        float x_data[] = {3.8141f, 3.5451f};
        float y_data[] = {3.6226f};

        // z = [3.8141^3.6226, 3.5451^3.6226] = [159.5443, 135.3775]
        // loss = sum(z) = 294.9218
        float exp_grad_x[] = {121.277222f, 100.109677f};
        float exp_grad_y[] = {294.921848f};

        Tensor x = create_test_tensor(x_shape, x_data, true);
        Tensor y = create_test_tensor(y_shape, y_data, true);

        Tensor z = Tensor_pow(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(x_shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(y_shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 3: Broadcasting a scalar base by a vector exponent
    {
        const char* tc_name = "pow_broadcast_scalar_vec_backward";
        TensorShape x_shape = {1};
        TensorShape y_shape = {2};
        float x_data[] = {0.8912f};
        float y_data[] = {1.9767f, 0.6043f};

        // z = [0.8912^1.9767, 0.8912^0.6043] = [0.8003, 0.9383]
        // loss = sum(z) = 1.7386
        float exp_grad_x[] = {2.39885f};
        float exp_grad_y[] = {-0.091731f, -0.107441f};

        Tensor x = create_test_tensor(x_shape, x_data, true);
        Tensor y = create_test_tensor(y_shape, y_data, true);

        Tensor z = Tensor_pow(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(x_shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(y_shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 4: Edge cases with x=0 and x<0
    {
        const char* tc_name = "pow_edge_cases_backward";
        TensorShape shape = {2};
        float x_data[] = {0.0f, -2.0f};
        float y_data[] = {2.0f, 3.0f};  // Exponents are integers

        // z = [0^2, (-2)^3] = [0.0, -8.0]
        // loss = sum(z) = -8.0
        // d(loss)/dx:
        // For x=0, y=2: grad is 0 (special case in GradFn_pow)
        // For x=-2, y=3: 3*(-2)^2 = 12.0
        float exp_grad_x[] = {0.0f, 12.0f};
        // d(loss)/dy:
        // For x=0: grad is 0 (special case in GradFn_pow)
        // For x=-2: grad is 0 (ln(-2) is undefined, special case)
        float exp_grad_y[] = {0.0f, 0.0f};

        Tensor x = create_test_tensor(shape, x_data, true);
        Tensor y = create_test_tensor(shape, y_data, true);

        Tensor z = Tensor_pow(x, y);
        Tensor loss = Tensor_sum(z);

        Tensor grad_dummy = {0};
        Tensor_backward(loss, grad_dummy);

        Tensor expected_grad_x = create_test_tensor(shape, exp_grad_x, false);
        Tensor expected_grad_y = create_test_tensor(shape, exp_grad_y, false);

        compare_tensors(&x.node->grad, &expected_grad_x, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
        compare_tensors(&y.node->grad, &expected_grad_y, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }

    // Test Case 5: Complex computation graph (z = (a^b) * c)
    {
        const char* tc_name = "pow_complex_graph_backward";
        TensorShape shape = {1};
        float a_data[] = {1.4839f};
        float b_data[] = {2.2687f};
        float c_data[] = {0.6194f};

        // Let d = a^b. Then z = d * c.
        // Forward: d = 1.4839^2.2687 = 2.4483. z = 2.4483 * 0.6194 = 1.5164
        // Backward pass (upstream grad for d is c=0.6194):
        // dz/da = (dz/dd) * (dd/da) = c * (b*a^(b-1)) = 0.6194 * (2.2687*1.4839^1.2687)
        float exp_grad_a[] = {2.318512f};
        // dz/db = (dz/dd) * (dd/db) = c * (a^b*ln(a)) = 0.6194 * (2.4483*ln(1.4839))
        float exp_grad_b[] = {0.598515f};
        // dz/dc = d = a^b = 2.4483
        float exp_grad_c[] = {2.448306f};

        Tensor a = create_test_tensor(shape, a_data, true);
        Tensor b = create_test_tensor(shape, b_data, true);
        Tensor c = create_test_tensor(shape, c_data, true);

        Tensor d = Tensor_pow(a, b);
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