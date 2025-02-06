#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define check_answer(input, answer, check_bytes)		\
assert(memcmp(input, answer, check_bytes) == 0)

#define check_array_ambiguous(input, answer, check_elems, tolerate)	\
for(int tocheck = 0; tocheck < check_elems; tocheck++)						\
assert(input[tocheck]-answer[tocheck] <= tolerate || answer[tocheck]-input[tocheck] <= tolerate)

void set_tensor_data(Tensor a, float* data) {
	int numel = a.data->numel;
	for (int i = 0; i < numel; i++) {
		a.data->flex[i] = data[i];
	}
}

float* get_tensor_data(Tensor a) {
	int numel = a.data->numel;
	float* res = (float*)malloc(sizeof(float) * numel);
	assert(res != NULL);
	memcpy(res, a.data->flex, sizeof(float) * numel);
	return res;
}

float* get_tensor_grad(Tensor a) {
	int numel = a.node->grad.data->numel;
	float* res = (float*)malloc(sizeof(float) * numel);
	assert(res != NULL);
	memcpy(res, a.node->grad.data->flex, sizeof(float) * numel);
	return res;
}


int main()
{
	Tensor a = Tensor_new((TensorShape) { 4 }, true);
	float a_data[] = { 1,2,3,4 };
	set_tensor_data(a, a_data);

	//Tensor_add test
	Tensor add_test = Tensor_add(a, a);
	float* add_res = get_tensor_data(add_test);
	float add_ans[] = { 2,4,6,8 };
	check_answer(add_res, add_ans, 4 * sizeof(float));

	//Tensor_mul test
	Tensor mul_test = Tensor_mul(a, a);
	float* mul_res = get_tensor_data(mul_test);
	float mul_ans[] = { 1,4,9,16 };
	check_answer(mul_res, mul_ans, 4 * sizeof(float));

	Tensor single_tensor = Tensor_new((TensorShape) { 0 }, false);
	single_tensor.data->flex[0] = 10;
	mul_test = Tensor_mul(single_tensor, a);
	mul_res = get_tensor_data(mul_test);
	float mul_ans2[] = { 10, 20, 30, 40 };
	check_answer(mul_res, mul_ans2, 4 * sizeof(float));

	//nn_softmax test
	Tensor b = nn_softmax(a);
	float* b_data = get_tensor_data(b);
	float b_data_ans[] = { 0.0321, 0.0871, 0.2369, 0.6439 };
	check_array_ambiguous(b_data, b_data_ans, 4, 0.001);

	Tensor c = Tensor_sum(b);
	float* c_data = get_tensor_data(c);
	float c_data_ans[] = { 1 };
	check_array_ambiguous(c_data, c_data_ans, 1, 0.001);

	Tensor_backward(c, (Tensor) {NULL});

	float* b_grad = get_tensor_grad(b);
	float b_grad_ans[] = { 1,1,1,1 };
	check_array_ambiguous(b_grad, b_grad_ans, 4, 0.001);
	float* a_grad = get_tensor_grad(a);
	float a_grad_ans[] = { 0,0,0,0 };
	check_array_ambiguous(a_grad, a_grad_ans, 4, 0.001);


	return 0;
}