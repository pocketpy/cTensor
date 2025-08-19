#include "../../include/cten.h"
#include "../test_utils.h"
#include "../csv_reporter.h"
#include "../test_config.h"
#include <stdio.h>

void test_add_backward() {
    const char* op_name = "add_backward";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);

    // Test Case 1: Simple backward (1x1 tensors)
    {
        const char* tc_name = "Simple_backward";
        // Sub-test 1: Scalar backward
        {
            TensorShape s_shape = {1};
            float d1[] = {2.0f};
            float d2[] = {3.0f};
            float exp_grad1[] = {1.0f};  // dz/dx = 1
            float exp_grad2[] = {1.0f};  // dz/dy = 1

            Tensor t1 = create_test_tensor(s_shape, d1, true);
            Tensor t2 = create_test_tensor(s_shape, d2, true);
            Tensor z = Tensor_add(t1, t2);  // z = 5.0

            Tensor_backward(z, (Tensor){0});

            Tensor expected_grad1 = create_test_tensor(s_shape, exp_grad1, false);
            Tensor expected_grad2 = create_test_tensor(s_shape, exp_grad2, false);

            compare_tensors(&t1.node->grad,
                            &expected_grad1,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad,
                            &expected_grad2,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Vector sum backward
        {
            TensorShape v_shape = {3};
            float d1[] = {1.0f, 2.0f, 3.0f};
            float d2[] = {4.0f, 5.0f, 6.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f};

            Tensor t1 = create_test_tensor(v_shape, d1, true);
            Tensor t2 = create_test_tensor(v_shape, d2, true);
            Tensor z = Tensor_add(t1, t2);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad = create_test_tensor(v_shape, exp_grad, false);

            compare_tensors(&t1.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Matrix sum backward
        {
            TensorShape m_shape = {2, 2};
            float d1[] = {1.0f, 2.0f, 3.0f, 4.0f};
            float d2[] = {5.0f, 6.0f, 7.0f, 8.0f};
            float exp_grad[] = {1.0f, 1.0f, 1.0f, 1.0f};

            Tensor t1 = create_test_tensor(m_shape, d1, true);
            Tensor t2 = create_test_tensor(m_shape, d2, true);
            Tensor z = Tensor_add(t1, t2);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad = create_test_tensor(m_shape, exp_grad, false);

            compare_tensors(&t1.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t2.node->grad,
                            &expected_grad,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 2: Broadcasting backward
    {
        const char* tc_name = "Broadcasting_backward";
        // Sub-test 1: Vector + Scalar
        {
            TensorShape vec_shape = {2};
            TensorShape scalar_shape = {1};
            float vec_data[] = {1.0f, 2.0f};
            float scalar_data[] = {3.0f};
            float exp_grad_vec[] = {1.0f, 1.0f};
            float exp_grad_scalar[] = {2.0f};

            Tensor t_vec = create_test_tensor(vec_shape, vec_data, true);
            Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, true);
            Tensor z = Tensor_add(t_vec, t_scalar);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_vec = create_test_tensor(vec_shape, exp_grad_vec, false);
            Tensor expected_grad_scalar = create_test_tensor(scalar_shape, exp_grad_scalar, false);

            compare_tensors(&t_vec.node->grad,
                            &expected_grad_vec,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_scalar.node->grad,
                            &expected_grad_scalar,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Matrix + Row Vector
        {
            TensorShape mat_shape = {2, 3};
            TensorShape row_shape = {1, 3};
            float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float row_data[] = {0.1f, 0.2f, 0.3f};
            float exp_grad_mat[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_row[] = {2.0f, 2.0f, 2.0f};

            Tensor t_mat = create_test_tensor(mat_shape, mat_data, true);
            Tensor t_row = create_test_tensor(row_shape, row_data, true);
            Tensor z = Tensor_add(t_mat, t_row);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_mat = create_test_tensor(mat_shape, exp_grad_mat, false);
            Tensor expected_grad_row = create_test_tensor(row_shape, exp_grad_row, false);

            compare_tensors(&t_mat.node->grad,
                            &expected_grad_mat,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_row.node->grad,
                            &expected_grad_row,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 3: Matrix + Column Vector
        {
            TensorShape mat_shape = {2, 3};
            TensorShape col_shape = {2, 1};
            float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float col_data[] = {10.0f, 20.0f};
            float exp_grad_mat[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_col[] = {3.0f, 3.0f};

            Tensor t_mat = create_test_tensor(mat_shape, mat_data, true);
            Tensor t_col = create_test_tensor(col_shape, col_data, true);
            Tensor z = Tensor_add(t_mat, t_col);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_mat = create_test_tensor(mat_shape, exp_grad_mat, false);
            Tensor expected_grad_col = create_test_tensor(col_shape, exp_grad_col, false);

            compare_tensors(&t_mat.node->grad,
                            &expected_grad_mat,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_col.node->grad,
                            &expected_grad_col,
                            op_name,
                            tc_name,
                            3,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 4: 3D + 2D Tensor
        {
            TensorShape tensor3d_shape = {2, 2, 2};
            TensorShape tensor2d_shape = {1, 2, 2};
            float data3d[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            float data2d[] = {0.1f, 0.2f, 0.3f, 0.4f};
            float exp_grad_3d[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_2d[] = {2.0f, 2.0f, 2.0f, 2.0f};

            Tensor t_3d = create_test_tensor(tensor3d_shape, data3d, true);
            Tensor t_2d = create_test_tensor(tensor2d_shape, data2d, true);
            Tensor z = Tensor_add(t_3d, t_2d);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_3d = create_test_tensor(tensor3d_shape, exp_grad_3d, false);
            Tensor expected_grad_2d = create_test_tensor(tensor2d_shape, exp_grad_2d, false);

            compare_tensors(&t_3d.node->grad,
                            &expected_grad_3d,
                            op_name,
                            tc_name,
                            4,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_2d.node->grad,
                            &expected_grad_2d,
                            op_name,
                            tc_name,
                            4,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 5: Multi-dim broadcast
        {
            TensorShape large_shape = {2, 3, 4};
            TensorShape small_shape = {1, 1, 4};
            float large_data[24];
            for(int i = 0; i < 24; i++)
                large_data[i] = (float)(i + 1);
            float small_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
            float exp_grad_large[24];
            for(int i = 0; i < 24; i++)
                exp_grad_large[i] = 1.0f;
            float exp_grad_small[] = {6.0f, 6.0f, 6.0f, 6.0f};

            Tensor t_large = create_test_tensor(large_shape, large_data, true);
            Tensor t_small = create_test_tensor(small_shape, small_data, true);
            Tensor z = Tensor_add(t_large, t_small);
            Tensor l = Tensor_sum(z);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_large = create_test_tensor(large_shape, exp_grad_large, false);
            Tensor expected_grad_small = create_test_tensor(small_shape, exp_grad_small, false);

            compare_tensors(&t_large.node->grad,
                            &expected_grad_large,
                            op_name,
                            tc_name,
                            5,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_small.node->grad,
                            &expected_grad_small,
                            op_name,
                            tc_name,
                            5,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 3: Chained and Complex Graphs
    {
        const char* tc_name = "Chained_and_Complex_Graphs_backward";
        // Sub-test 1: Complex computation graph (x+y)*w
        {
            TensorShape v_shape = {2};
            TensorShape s_shape = {1};
            float x_data[] = {1.0f, 2.0f};
            float y_data[] = {3.0f};
            float w_data[] = {2.0f, 3.0f};
            float exp_grad_x[] = {2.0f, 3.0f};
            float exp_grad_y[] = {5.0f};
            float exp_grad_w[] = {4.0f, 5.0f};

            Tensor x = create_test_tensor(v_shape, x_data, true);
            Tensor y = create_test_tensor(s_shape, y_data, true);
            Tensor w = create_test_tensor(v_shape, w_data, true);

            Tensor sum_xy = Tensor_add(x, y);
            Tensor prod = Tensor_mul(sum_xy, w);
            Tensor l = Tensor_sum(prod);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_x = create_test_tensor(v_shape, exp_grad_x, false);
            Tensor expected_grad_y = create_test_tensor(s_shape, exp_grad_y, false);
            Tensor expected_grad_w = create_test_tensor(v_shape, exp_grad_w, false);

            compare_tensors(&x.node->grad,
                            &expected_grad_x,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&y.node->grad,
                            &expected_grad_y,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&w.node->grad,
                            &expected_grad_w,
                            op_name,
                            tc_name,
                            1,
                            TEST_FLOAT_TOLERANCE);
        }

        // Sub-test 2: Chain of broadcasting operations (mat + row) + col
        {
            TensorShape mat_shape = {2, 3};
            TensorShape row_shape = {1, 3};
            TensorShape col_shape = {2, 1};
            float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float row_data[] = {0.1f, 0.2f, 0.3f};
            float col_data[] = {10.0f, 20.0f};
            float exp_grad_mat[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float exp_grad_row[] = {2.0f, 2.0f, 2.0f};
            float exp_grad_col[] = {3.0f, 3.0f};

            Tensor t_mat = create_test_tensor(mat_shape, mat_data, true);
            Tensor t_row = create_test_tensor(row_shape, row_data, true);
            Tensor t_col = create_test_tensor(col_shape, col_data, true);

            Tensor z1 = Tensor_add(t_mat, t_row);
            Tensor z2 = Tensor_add(z1, t_col);
            Tensor l = Tensor_sum(z2);

            Tensor_backward(l, (Tensor){0});

            Tensor expected_grad_mat = create_test_tensor(mat_shape, exp_grad_mat, false);
            Tensor expected_grad_row = create_test_tensor(row_shape, exp_grad_row, false);
            Tensor expected_grad_col = create_test_tensor(col_shape, exp_grad_col, false);

            compare_tensors(&t_mat.node->grad,
                            &expected_grad_mat,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_row.node->grad,
                            &expected_grad_row,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
            compare_tensors(&t_col.node->grad,
                            &expected_grad_col,
                            op_name,
                            tc_name,
                            2,
                            TEST_FLOAT_TOLERANCE);
        }
    }

    // Test Case 4: Broadcasting with other ops
    {
        const char* tc_name = "mul_broadcast_scalar_backward";
        TensorShape mat_shape = {2, 2};
        TensorShape scalar_shape = {1};
        float mat_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
        float scalar_data[] = {2.0f};
        float exp_grad_mat[] = {2.0f, 2.0f, 2.0f, 2.0f};
        float exp_grad_scalar[] = {14.0f};

        Tensor t_mat = create_test_tensor(mat_shape, mat_data, true);
        Tensor t_scalar = create_test_tensor(scalar_shape, scalar_data, true);

        Tensor z = Tensor_mul(t_mat, t_scalar);
        Tensor l = Tensor_sum(z);

        Tensor_backward(l, (Tensor){0});

        Tensor expected_grad_mat = create_test_tensor(mat_shape, exp_grad_mat, false);
        Tensor expected_grad_scalar = create_test_tensor(scalar_shape, exp_grad_scalar, false);

        compare_tensors(&t_mat.node->grad,
                        &expected_grad_mat,
                        "mul_backward",
                        tc_name,
                        1,
                        TEST_FLOAT_TOLERANCE);
        compare_tensors(&t_scalar.node->grad,
                        &expected_grad_scalar,
                        "mul_backward",
                        tc_name,
                        1,
                        TEST_FLOAT_TOLERANCE);
    }

    cten_free(pool_id);
}