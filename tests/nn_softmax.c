#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void assert_equal(Tensor a, Tensor b) {
	float epsilon = 1e-3;
	assert(a.data->numel == b.data->numel);

	cten_assert_shape("Shape not equal", a.shape, b.shape);

	for (int i = 0; i < a.data->numel; i++) {
		assert((a.data->flex[i] - b.data->flex[i] < epsilon) &&
		(a.data->flex[i] - b.data->flex[i] > -epsilon));
	}
}

Tensor create_tensor(TensorShape shape, float* data, bool requires_grad) {
	Tensor res = Tensor_new(shape, requires_grad);
	int numel = res.data->numel;
	for (int i = 0; i < numel; i++) {
		res.data->flex[i] = data[i];
	}
	return res;
}


int main()
{
	//dim1 simple softmax
	float a_data[] = { 1,2,3,4 };
	Tensor a = create_tensor((TensorShape) { 4 }, a_data, true);
	Tensor b = nn_softmax(a);
	Tensor c = Tensor_sum(b);
	Tensor_backward(c, (Tensor) {NULL});

	float b_data[] = { 0.0321, 0.0871, 0.2369, 0.6439 };
	Tensor b_answer = create_tensor((TensorShape) { 4 }, b_data, false);
	assert_equal(b, b_answer);

	Tensor a_grad_answer = Tensor_zeros((TensorShape) { 4 }, false);
	assert_equal(a.node->grad, a_grad_answer);

	//dim1 hard softmax
	Tensor a1 = create_tensor((TensorShape) { 4 }, a_data, true);
	Tensor b1 = nn_softmax(a1);
	Tensor z1 = Tensor_mul(a1, b1);
	Tensor c1 = Tensor_sum(z1);
	Tensor_backward(c1, (Tensor) { NULL });

	float b_grad_data[] = { 1,2,3,4 };
	Tensor b1_grad_answer = create_tensor((TensorShape) { 4 }, b_grad_data, false);
	assert_equal(b1.node->grad, b1_grad_answer);

	float a_grad_data[] = { -0.0479, -0.0429, 0.1202, 0.9706 };
	Tensor a1_grad_answer = create_tensor((TensorShape) { 4 }, a_grad_data, false);
	assert_equal(a1.node->grad, a1_grad_answer);

	//dim2 simple softmax
	float dim2_a_data[] = { 1,2,3,4,5,6 };
	a = create_tensor((TensorShape) { 2,3 }, dim2_a_data, true);
	b = nn_softmax(a);
	c = Tensor_sum(b);
	Tensor_backward(c, (Tensor) { NULL });

	float dim2_b_data[] = { 0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652 };
	b_answer = create_tensor((TensorShape) { 2,3 }, dim2_b_data, false);
	assert_equal(b, b_answer);

	a_grad_answer = Tensor_zeros((TensorShape) { 2,3 }, false);
	assert_equal(a.node->grad, a_grad_answer);

	//dim2 hard softmax
	Tensor a2 = create_tensor((TensorShape) { 2,3 }, dim2_a_data, true);
	Tensor b2 = nn_softmax(a2);
	Tensor z2 = Tensor_mul(a2, b2);
	Tensor c2 = Tensor_sum(z2);
	Tensor_backward(c2, (Tensor) { NULL });

	float dim2_b_grad_data[] = { 1,2,3,4,5,6 };
	Tensor b2_grad_answer = create_tensor((TensorShape) { 2,3 }, dim2_b_grad_data, false);
	assert_equal(b2.node->grad, b2_grad_answer);

	float dim2_a_grad_data[] = { -0.0518, 0.1040, 0.9478, -0.0518, 0.1040, 0.9478 };
	Tensor a2_grad_answer = create_tensor((TensorShape) { 2,3 }, dim2_a_grad_data, false);
	assert_equal(a2.node->grad, a2_grad_answer);

	//dim3 simple softmax
	float dim3_a_data[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
	a = create_tensor((TensorShape) { 2,3,4 }, dim3_a_data, true);
	b = nn_softmax(a);
	c = Tensor_sum(b);
	Tensor_backward(c, (Tensor) { NULL });

	float dim3_b_data[] = { 0.0321,0.0871,0.2369,0.6439,0.0321,0.0871,0.2369,0.6439,
									0.0321,0.0871,0.2369,0.6439,0.0321,0.0871,0.2369,0.6439,
									0.0321,0.0871,0.2369,0.6439,0.0321,0.0871,0.2369,0.6439 };
	b_answer = create_tensor((TensorShape) { 2,3,4 }, dim3_b_data, false);
	assert_equal(b, b_answer);

	a_grad_answer = Tensor_zeros((TensorShape) { 2,3,4 }, false);
	assert_equal(a.node->grad, a_grad_answer);

	//dim3 hard softmax
	Tensor a3 = create_tensor((TensorShape) { 2,3,4 }, dim3_a_data, true);
	Tensor b3 = nn_softmax(a3);
	Tensor z3 = Tensor_mul(a3, b3);
	Tensor c3 = Tensor_sum(z3);
	Tensor_backward(c3, (Tensor) { NULL });

	float dim3_b_grad_data[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
	Tensor b3_grad_answer = create_tensor((TensorShape) { 2,3,4 }, dim3_b_grad_data, false);
	assert_equal(b3.node->grad, b3_grad_answer);

	float dim3_a_grad_data[] = { -0.0479,-0.0429,0.1202,0.9706,-0.0479,-0.0429,0.1202,0.9706,
											-0.0479,-0.0429,0.1202,0.9706,-0.0479,-0.0429,0.1202,0.9706,
											-0.0479,-0.0429,0.1202,0.9706,-0.0479,-0.0429,0.1202,0.9706 };
	Tensor a3_grad_answer = create_tensor((TensorShape) { 2,3,4 }, dim3_a_grad_data, false);
	assert_equal(a3.node->grad, a3_grad_answer);


	return 0;
}